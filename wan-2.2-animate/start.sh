#!/usr/bin/env bash

echo "==========================================================="
echo "======================motoroller logs======================"
echo "==========================================================="

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# ---------------------------------------------------------------------------
# GPU check
# ---------------------------------------------------------------------------
echo "worker-comfyui: Checking GPU availability..."
if ! GPU_CHECK=$(python3 -c "
import torch
try:
    torch.cuda.init()
    name = torch.cuda.get_device_name(0)
    print(f'OK: {name}')
except Exception as e:
    print(f'FAIL: {e}')
    exit(1)
" 2>&1); then
    echo "worker-comfyui: GPU is not available: $GPU_CHECK"
    echo "worker-comfyui: Please contact RunPod support and report this machine."
    exit 1
fi
echo "worker-comfyui: GPU available — $GPU_CHECK"

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
NETWORK_VOLUME="/runpod-volume"
COMFYUI_DIR="$NETWORK_VOLUME/ComfyUI"
URL="http://127.0.0.1:8188"

# ---------------------------------------------------------------------------
# Start ComfyUI in background
# ---------------------------------------------------------------------------
echo "worker-comfyui: Starting ComfyUI..."
nohup python3 "$COMFYUI_DIR/main.py" --listen --disable-auto-launch > "$NETWORK_VOLUME/comfyui.log" 2>&1 &

# Wait for ComfyUI to be ready
counter=0
max_wait=300
until curl --silent --fail "$URL" --output /dev/null; do
    if [ $counter -ge $max_wait ]; then
        echo "⚠️  ComfyUI did not start in time. Check $NETWORK_VOLUME/comfyui.log"
        tail -n 100 $NETWORK_VOLUME/comfyui.log
        break
    fi
    echo "🔄 Waiting for ComfyUI... ($counter/$max_wait)"
    sleep 2
    counter=$((counter + 2))
done

if curl --silent --fail "$URL" --output /dev/null; then
    echo "🚀 ComfyUI is UP"
fi

# ---------------------------------------------------------------------------
# Start RunPod Handler
# ---------------------------------------------------------------------------
echo "worker-comfyui: Starting RunPod Handler"
python -u /handler.py
