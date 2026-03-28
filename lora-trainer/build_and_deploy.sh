#!/usr/bin/env bash
set -euo pipefail

IMAGE="motoroller95/lora-trainer"

GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

docker build -f Dockerfile --build-arg GIT_COMMIT="$GIT_COMMIT" -t "$IMAGE" .. && docker push "$IMAGE"
