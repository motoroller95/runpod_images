#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Helper functions
# ============================================================

STARTUP_LOG="/logs/startup.log"
mkdir -p /logs
echo "--- Startup log $(date) ---" > "$STARTUP_LOG"

status_msg() {
    echo ""
    echo "  $1"
}

run_quiet() {
    local label="$1"
    shift

    (
        while true; do
            sleep 10
            echo "       Still working..."
        done
    ) &
    local heartbeat_pid=$!

    "$@" >> "$STARTUP_LOG" 2>&1
    local exit_code=$?

    kill "$heartbeat_pid" 2>/dev/null
    wait "$heartbeat_pid" 2>/dev/null

    if [ $exit_code -ne 0 ]; then
        echo "       Warning: $label may have failed. Check $STARTUP_LOG for details."
    fi

    return $exit_code
}

# ============================================================
# Use libtcmalloc for better memory management
# ============================================================
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1 || true)"
if [ -n "${TCMALLOC}" ]; then
    export LD_PRELOAD="${TCMALLOC}"
fi

# ============================================================
# GPU detection
# ============================================================
detect_cuda_arch() {
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs)
    echo "$gpu_name" > /tmp/detected_gpu

    case "$gpu_name" in
        *B100*|*B200*|*GB200*)
            echo "blackwell" > /tmp/gpu_arch_type; echo "100" ;;
        *5090*|*5080*|*5070*|*5060*|*PRO*6000*Blackwell*)
            echo "blackwell" > /tmp/gpu_arch_type; echo "120" ;;
        *H100*|*H200*)
            echo "hopper" > /tmp/gpu_arch_type; echo "90" ;;
        *L4*|*L40*|*4090*|*4080*|*4070*|*4060*|*PRO*6000*Ada*)
            echo "ada" > /tmp/gpu_arch_type; echo "89" ;;
        *A10*|*A40*|*A6000*|*A5000*|*A4000*|*3090*|*3080*|*3070*|*3060*)
            echo "ampere" > /tmp/gpu_arch_type; echo "86" ;;
        *A100*)
            echo "ampere" > /tmp/gpu_arch_type; echo "80" ;;
        *T4*|*2080*|*2070*|*2060*)
            echo "turing" > /tmp/gpu_arch_type; echo "75" ;;
        *V100*)
            echo "volta" > /tmp/gpu_arch_type; echo "70" ;;
        *)
            echo "unknown" > /tmp/gpu_arch_type; echo "80;86;89;90" ;;
    esac
}

DETECTED_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | xargs)
CUDA_ARCH=$(detect_cuda_arch)

echo ""
echo "================================================"
echo "  lora-trainer starting up..."
echo "  GPU: $DETECTED_GPU"
echo "================================================"

# ============================================================
# [1/3] Flash attention
# ============================================================
status_msg "[1/3] Installing flash attention..."

FLASH_ATTN_WHEEL_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
WHEEL_INSTALLED=false

if [ -n "$FLASH_ATTN_WHEEL_URL" ]; then
    cd /tmp
    WHEEL_NAME=$(basename "$FLASH_ATTN_WHEEL_URL")

    if wget -q -O "$WHEEL_NAME" "$FLASH_ATTN_WHEEL_URL" >> "$STARTUP_LOG" 2>&1; then
        if pip install "$WHEEL_NAME" >> "$STARTUP_LOG" 2>&1; then
            rm -f "$WHEEL_NAME"
            WHEEL_INSTALLED=true
            touch /tmp/flash_attn_wheel_success
        else
            rm -f "$WHEEL_NAME"
        fi
    fi
fi

if [ "$WHEEL_INSTALLED" = false ]; then
    echo "       Building from source in background (this may take a few minutes)..."

    CPU_CORES=$(nproc)
    CPU_JOBS=$(( CPU_CORES - 2 ))
    [ "$CPU_JOBS" -lt 4 ] && CPU_JOBS=4
    AVAILABLE_RAM_GB=$(free -g | awk '/^Mem:/{print $7}')
    RAM_JOBS=$(( AVAILABLE_RAM_GB / 3 ))
    [ "$RAM_JOBS" -lt 4 ] && RAM_JOBS=4
    if [ "$CPU_JOBS" -lt "$RAM_JOBS" ]; then
        OPTIMAL_JOBS=$CPU_JOBS
    else
        OPTIMAL_JOBS=$RAM_JOBS
    fi

    (
        set -e
        CUDA_ARCH=$(detect_cuda_arch)

        pip install ninja packaging -q
        if ! ninja --version > /dev/null 2>&1; then
            pip uninstall -y ninja && pip install ninja
        fi

        cd /tmp
        rm -rf flash-attention
        git clone https://github.com/Dao-AILab/flash-attention.git
        cd flash-attention

        export FLASH_ATTN_CUDA_ARCHS="$CUDA_ARCH"
        export MAX_JOBS=$OPTIMAL_JOBS
        export NVCC_THREADS=4

        python setup.py install

        cd /tmp
        rm -rf flash-attention
    ) > /logs/flash_attn_install.log 2>&1 &
    FLASH_ATTN_PID=$!
    echo "$FLASH_ATTN_PID" > /tmp/flash_attn_pid
fi

# ============================================================
# [2/3] Fetching latest package updates
# ============================================================
status_msg "[2/3] Fetching latest updates..."

pip install torch torchvision torchaudio 2>&1
pip install transformers -U 2>&1
pip install --upgrade "huggingface_hub[cli]" 2>&1
pip install --upgrade "peft>=0.17.0" 2>&1
pip install --upgrade "deepspeed>=0.17.6" 2>&1
pip uninstall -y diffusers && pip install git+https://github.com/huggingface/diffusers 2>&1

# ============================================================
# [3/3] Starting RunPod Handler
# ============================================================
status_msg "[3/3] Starting RunPod Handler..."

echo ""
echo "================================================"
echo "  lora-trainer ready!"
echo "================================================"
echo ""

python -u /handler.py
