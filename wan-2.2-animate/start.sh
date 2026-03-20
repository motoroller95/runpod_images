#!/usr/bin/env bash
set -euo pipefail

echo "==========================================================="
echo "======================motoroller logs======================"
echo "==========================================================="

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1 || true)"
if [ -n "${TCMALLOC}" ]; then
    export LD_PRELOAD="${TCMALLOC}"
    echo "worker-comfyui: tcmalloc enabled: ${TCMALLOC}"
else
    echo "worker-comfyui: tcmalloc not found, skipping"
fi

# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------
echo "worker-comfyui: Checking GPU availability..."
GPU_CHECK=$(python3 -c "
import torch
try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    print(f'OK: {name}')
except Exception as e:
    print(f'FAIL: {e}')
    exit(1)
" 2>&1) || { echo "worker-comfyui: GPU not available: $GPU_CHECK"; exit 1; }
echo "worker-comfyui: GPU available — $GPU_CHECK"

# ---------------------------------------------------------------------------
# Symlink models and custom_nodes from network volume into /ComfyUI
# Leave input/ and output/ as clean local directories
# ---------------------------------------------------------------------------
NETWORK_VOLUME="/runpod-volume/hearmemann/wan-animate"
COMFYUI_DIR="/ComfyUI"

echo "worker-comfyui: Copying models and custom_nodes from network volume..."

for dir in models custom_nodes; do
    src="$NETWORK_VOLUME/$dir"
    dst="$COMFYUI_DIR/$dir"
    if [ ! -d "$src" ]; then
        echo "worker-comfyui: WARNING — $src not found on network volume, skipping"
        continue
    fi
    rm -rf "$dst"
    cp -r "$src" "$dst"
    echo "worker-comfyui: Copied $src -> $dst"
done

# Ensure input/output are clean empty local dirs
rm -rf "$COMFYUI_DIR/input" "$COMFYUI_DIR/output"
mkdir -p "$COMFYUI_DIR/input" "$COMFYUI_DIR/output"

# ---------------------------------------------------------------------------
# Start ComfyUI in background
# ---------------------------------------------------------------------------
COMFY_LOG_LEVEL="${COMFY_LOG_LEVEL:-INFO}"
COMFY_LOG_FILE="/ComfyUI/log.log"
COMFY_PID_FILE="/tmp/comfyui.pid"

echo "worker-comfyui: Starting ComfyUI (log: $COMFY_LOG_FILE)"
nohup python3 /ComfyUI/main.py \
    --disable-auto-launch \
    --disable-metadata \
    --verbose "${COMFY_LOG_LEVEL}" \
    --log-stdout \
    > "$COMFY_LOG_FILE" 2>&1 &
echo $! > "$COMFY_PID_FILE"
echo "worker-comfyui: ComfyUI started (pid=$(cat $COMFY_PID_FILE))"

# ---------------------------------------------------------------------------
# Start RunPod Handler
# ---------------------------------------------------------------------------
echo "worker-comfyui: Starting RunPod Handler"
python -u /handler.py
