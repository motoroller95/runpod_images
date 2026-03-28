#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
WHITE='\033[0;37m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}================================================${NC}"
    echo -e "${BOLD}$1${NC}"
    echo -e "${CYAN}================================================${NC}"
}
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ $1${NC}"; }

# ── Hardcoded choices ──────────────────────────────────────────────
MODEL_TYPE="z_image_turbo"
MODEL_NAME="Z Image Turbo"
TOML_FILE="z_image_toml.toml"
CAPTION_MODE="skip"
# ──────────────────────────────────────────────────────────────────

clear
print_header "HearmemanAI LoRA Trainer — AUTO MODE (Z Image Turbo)"
echo ""
print_info "Model:        $MODEL_NAME"
print_info "Captioning:   skipped (using existing captions)"
print_info "Config:       default settings"
echo ""

mkdir -p "$NETWORK_VOLUME/logs"

# flash-attn check
if [ -f /tmp/flash_attn_wheel_success ]; then
    print_success "flash-attn is installed and ready (prebuilt wheel)."
elif [ -f /tmp/flash_attn_pid ]; then
    FLASH_ATTN_PID=$(cat /tmp/flash_attn_pid)
    if kill -0 "$FLASH_ATTN_PID" 2>/dev/null; then
        print_warning "flash-attn still compiling (PID: $FLASH_ATTN_PID), waiting..."
        while kill -0 "$FLASH_ATTN_PID" 2>/dev/null; do echo -n "."; sleep 2; done
        echo ""
        wait "$FLASH_ATTN_PID" 2>/dev/null && print_success "flash-attn compiled!" || \
            print_warning "flash-attn compile may have failed. Check: $NETWORK_VOLUME/logs/flash_attn_install.log"
        rm -f /tmp/flash_attn_pid
    else
        rm -f /tmp/flash_attn_pid
        print_success "flash-attn is installed and ready."
    fi
fi

echo ""

# CUDA check
print_header "Checking CUDA Compatibility"
python3 << 'PYTHON_EOF'
import sys, torch
if torch.cuda.is_available():
    x = torch.randn(1, device='cuda') * 2
    print("CUDA compatibility check passed")
else:
    print("CUDA NOT AVAILABLE — deploy with CUDA 12.8")
    sys.exit(1)
PYTHON_EOF
echo ""

# Model download
print_header "Starting Model Download"
mkdir -p "$NETWORK_VOLUME/models"
MODEL_DOWNLOAD_PID=""

# Z Image Turbo toml setup
mkdir -p "$NETWORK_VOLUME/diffusion_pipe/examples"
if [ -f "$NETWORK_VOLUME/diffusion_pipe/examples/z_image_toml.toml" ]; then
    print_info "z_image_toml.toml already exists in examples directory"
elif [ -f "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/z_image_toml.toml" ]; then
    sed -i "s|^output_dir = .*|output_dir = '$NETWORK_VOLUME/output_folder/z_image_lora'|" \
        "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/z_image_toml.toml"
    mv "$NETWORK_VOLUME/runpod-diffusion_pipe/toml_files/z_image_toml.toml" \
        "$NETWORK_VOLUME/diffusion_pipe/examples/"
    print_success "Moved z_image_toml.toml to examples directory"
else
    print_warning "z_image_toml.toml not found — please copy it manually to: $NETWORK_VOLUME/diffusion_pipe/examples/"
fi

# Z Image Turbo model files download
mkdir -p "$NETWORK_VOLUME/models/z_image"
Z_NEED_DOWNLOAD=false
for f in z_image_turbo_bf16.safetensors ae.safetensors qwen_3_4b.safetensors zimage_turbo_training_adapter_v2.safetensors; do
    [ ! -f "$NETWORK_VOLUME/models/z_image/$f" ] && Z_NEED_DOWNLOAD=true && break
done

if [ "$Z_NEED_DOWNLOAD" = true ]; then
    print_info "Starting Z Image Turbo model download in background..."
    (
        hf download Hearme-man/z_image_turbo \
            z_image_turbo_bf16.safetensors ae.safetensors qwen_3_4b.safetensors \
            zimage_turbo_training_adapter_v2.safetensors \
            --local-dir "$NETWORK_VOLUME/models/z_image"
    ) > "$NETWORK_VOLUME/logs/model_download.log" 2>&1 &
    MODEL_DOWNLOAD_PID=$!
else
    print_success "Z Image Turbo model files already present, skipping download."
fi

# Wait for download
if [ -n "$MODEL_DOWNLOAD_PID" ]; then
    print_header "Finalizing Model Download"
    print_info "Waiting for model download to complete..."
    print_info "Monitor: tail -f $NETWORK_VOLUME/logs/model_download.log"
    echo ""
    timeout_counter=0
    max_timeout=10800
    while kill -0 "$MODEL_DOWNLOAD_PID" 2>/dev/null; do
        if tail -n 20 "$NETWORK_VOLUME/logs/model_download.log" 2>/dev/null | grep -qi "error\|failed\|exception\|unauthorized\|403\|404"; then
            print_error "Download error. Check: $NETWORK_VOLUME/logs/model_download.log"
            kill "$MODEL_DOWNLOAD_PID" 2>/dev/null || true
            exit 1
        fi
        echo -n "."
        sleep 3
        timeout_counter=$((timeout_counter + 3))
        [ $timeout_counter -ge $max_timeout ] && print_error "Download timed out." && kill "$MODEL_DOWNLOAD_PID" 2>/dev/null && exit 1
    done
    echo ""
    wait "$MODEL_DOWNLOAD_PID"
    [ $? -ne 0 ] && print_error "Download failed. Check log." && exit 1

    # Verify
    print_info "Verifying model download..."
    missing=""
    for f in z_image_turbo_bf16.safetensors ae.safetensors qwen_3_4b.safetensors zimage_turbo_training_adapter_v2.safetensors; do
        [ ! -f "$NETWORK_VOLUME/models/z_image/$f" ] && missing="$missing $f"
    done
    [ -n "$missing" ] && print_error "Missing files:$missing" && exit 1
    print_success "Model download completed and verified!"
    echo ""
fi

# Dataset config
print_header "Configuring Dataset"
DATASET_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/dataset.toml"
if [ -f "$DATASET_TOML" ]; then
    cp "$DATASET_TOML" "$DATASET_TOML.backup"
    sed -i "s|\$NETWORK_VOLUME/image_dataset_here|$NETWORK_VOLUME/image_dataset_here|g" "$DATASET_TOML" 2>/dev/null || true
    sed -i "s|\$NETWORK_VOLUME/video_dataset_here|$NETWORK_VOLUME/video_dataset_here|g" "$DATASET_TOML" 2>/dev/null || true
    print_success "Dataset configuration updated"
else
    print_warning "dataset.toml not found at $DATASET_TOML"
fi
echo ""

# Training summary
print_header "Training Configuration Summary"
MODEL_TOML="$NETWORK_VOLUME/diffusion_pipe/examples/$TOML_FILE"
RESOLUTION=$(grep "^resolutions = " "$DATASET_TOML" 2>/dev/null | sed 's/resolutions = \[\([0-9]*\)\]/\1/')
[ -z "$RESOLUTION" ] && RESOLUTION="1024"
EPOCHS=$(grep "^epochs = " "$MODEL_TOML" 2>/dev/null | sed 's/epochs = //')
SAVE_EVERY=$(grep "^save_every_n_epochs = " "$MODEL_TOML" 2>/dev/null | sed 's/save_every_n_epochs = //')
RANK=$(grep "^rank = " "$MODEL_TOML" 2>/dev/null | sed 's/rank = //')
LR=$(grep "^lr = " "$MODEL_TOML" 2>/dev/null | sed 's/lr = //')
[ -z "$EPOCHS" ]     && EPOCHS="80"
[ -z "$SAVE_EVERY" ] && SAVE_EVERY="10"
[ -z "$RANK" ]       && RANK="32"
[ -z "$LR" ]         && LR="2e-4"

echo -e "${BOLD}Model:${NC}          $MODEL_NAME"
echo -e "${BOLD}TOML Config:${NC}    examples/$TOML_FILE"
echo -e "${BOLD}Resolution:${NC}     ${RESOLUTION}x${RESOLUTION}"
echo ""
echo -e "${BOLD}Training Parameters:${NC}"
echo "  📊 Epochs:        $EPOCHS"
echo "  💾 Save Every:    $SAVE_EVERY epochs"
echo "  🎛️  LoRA Rank:     $RANK"
echo "  📈 Learning Rate: $LR"
echo "  📂 Dataset:       existing captions"
echo ""

# Start training
print_header "Starting Training"
cd "$NETWORK_VOLUME/diffusion_pipe"

print_info "Upgrading transformers..."
pip install transformers -U -q

print_info "Upgrading peft..."
pip install --upgrade "peft>=0.17.0" -q

echo ""
print_warning "⚠️  Z Image Turbo init can take several minutes — do not interrupt."
sleep 5
echo ""

print_info "Launching DeepSpeed training..."
echo ""
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" deepspeed --num_gpus=1 train.py --deepspeed --config "examples/$TOML_FILE"

print_success "Training completed!"