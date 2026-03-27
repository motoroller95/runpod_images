#!/usr/bin/env bash
set -euo pipefail

echo "==========================================================="
echo "======================klein-9b logs========================"
echo "==========================================================="

# ---------------------------------------------------------------------------
# tcmalloc
# ---------------------------------------------------------------------------
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1 || true)"
if [ -n "${TCMALLOC}" ]; then
    export LD_PRELOAD="${TCMALLOC}"
    echo "worker: tcmalloc enabled: ${TCMALLOC}"
else
    echo "worker: tcmalloc not found, skipping"
fi

# ---------------------------------------------------------------------------
# SageAttention build in background (needs GPU, pre-cloned in Docker)
# ---------------------------------------------------------------------------
if python3 -c "import sageattention" 2>/dev/null; then
    echo "SageAttention already installed, skipping build"
    SAGE_PID=""
else
    echo "Starting SageAttention build..."
    (
        export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32
        cd /tmp/SageAttention
        pip install .
        echo "SageAttention build completed" > /tmp/sage_build_done
    ) > /tmp/sage_build.log 2>&1 &
    SAGE_PID=$!
    echo "SageAttention build started in background (PID: $SAGE_PID)"
fi

# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------
echo "worker: Checking GPU availability..."
GPU_CHECK=""
GPU_OK=0
for attempt in 1 2 3; do
    GPU_CHECK=$(python3 -c "
import torch
try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    print(f'OK: {name}')
except Exception as e:
    print(f'FAIL: {e}')
    exit(1)
" 2>&1) && GPU_OK=1 && break
    echo "worker: GPU check attempt $attempt/3 failed: $GPU_CHECK"
    [ "$attempt" -lt 3 ] && sleep 5
done

if [ "$GPU_OK" -eq 0 ]; then
    echo "worker: torch check failed 3 times, trying nvidia-smi fallback..."
    if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q .; then
        echo "worker: GPU available via nvidia-smi — $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        echo "worker: GPU not available after all checks, aborting"
        exit 1
    fi
else
    echo "worker: GPU available — $GPU_CHECK"
fi

# ---------------------------------------------------------------------------
# Download models directly to /ComfyUI/models/
# ---------------------------------------------------------------------------
COMFYUI_DIR="/ComfyUI"
DIFFUSION_MODELS_DIR="$COMFYUI_DIR/models/diffusion_models"
TEXT_ENCODERS_DIR="$COMFYUI_DIR/models/text_encoders"
VAE_DIR="$COMFYUI_DIR/models/vae"

download_model() {
    local url="$1"
    local full_path="$2"

    local destination_dir=$(dirname "$full_path")
    local destination_file=$(basename "$full_path")

    mkdir -p "$destination_dir"

    if [ -f "$full_path" ]; then
        local size_bytes=$(stat -c%s "$full_path" 2>/dev/null || stat -f%z "$full_path" 2>/dev/null || echo 0)
        local size_mb=$((size_bytes / 1024 / 1024))

        if [ "$size_bytes" -lt 10485760 ]; then
            echo "Deleting corrupted file (${size_mb}MB < 10MB): $full_path"
            rm -f "$full_path"
        else
            echo "$destination_file already exists (${size_mb}MB), skipping download."
            return 0
        fi
    fi

    if [ -f "${full_path}.aria2" ]; then
        echo "Deleting .aria2 control file: ${full_path}.aria2"
        rm -f "${full_path}.aria2"
        rm -f "$full_path"
    fi

    echo "Downloading $destination_file to $destination_dir..."
    aria2c -x 16 -s 16 -k 1M --continue=true -d "$destination_dir" -o "$destination_file" "$url" &
    echo "Download started in background for $destination_file"
}

echo "Starting model downloads..."

download_model "https://huggingface.co/dci05049/flux2-klein-9b/resolve/main/flux-2-klein-9b.safetensors" "$DIFFUSION_MODELS_DIR/flux-2-klein-9b.safetensors"
download_model "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/text_encoders/qwen_3_8b.safetensors" "$TEXT_ENCODERS_DIR/qwen_3_8b.safetensors"
download_model "https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b/resolve/main/split_files/vae/flux2-vae.safetensors" "$VAE_DIR/flux2-vae.safetensors"

# Wait for all aria2c downloads
while pgrep -x "aria2c" > /dev/null; do
    echo "Models are downloading (In Progress)"
    sleep 5
done
echo "All models downloaded successfully"

# ---------------------------------------------------------------------------
# Wait for SageAttention build
# ---------------------------------------------------------------------------
if [ -n "${SAGE_PID:-}" ]; then
    echo "Waiting for SageAttention build to complete..."
    while ! [ -f /tmp/sage_build_done ]; do
        if ps -p $SAGE_PID > /dev/null 2>&1; then
            echo "SageAttention build in progress, this may take up to 5 minutes."
            sleep 5
        else
            if ! [ -f /tmp/sage_build_done ]; then
                echo "SageAttention build process ended unexpectedly. Check logs at /tmp/sage_build.log"
                echo "Continuing with ComfyUI startup..."
                break
            fi
        fi
    done

    if [ -f /tmp/sage_build_done ]; then
        echo "SageAttention build completed successfully!"
    fi
fi

# ---------------------------------------------------------------------------
# Start ComfyUI in background
# ---------------------------------------------------------------------------
COMFY_LOG_LEVEL="${COMFY_LOG_LEVEL:-INFO}"
COMFY_LOG_FILE="/ComfyUI/log.log"
COMFY_PID_FILE="/tmp/comfyui.pid"

echo "worker: Starting ComfyUI (log: $COMFY_LOG_FILE)"
nohup python3 /ComfyUI/main.py \
    --disable-auto-launch \
    --disable-metadata \
    --use-sage-attention \
    --verbose "${COMFY_LOG_LEVEL}" \
    --log-stdout \
    > "$COMFY_LOG_FILE" 2>&1 &
echo $! > "$COMFY_PID_FILE"
echo "worker: ComfyUI started (pid=$(cat $COMFY_PID_FILE))"

# ---------------------------------------------------------------------------
# Start RunPod Handler
# ---------------------------------------------------------------------------
echo "worker: Starting RunPod Handler"
python -u /handler.py
