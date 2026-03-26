FROM aiorbust/flux2-klein-9b:latest

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        aria2 curl \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip install --no-cache-dir runpod requests boto3 onnxruntime-gpu

# SageAttention (pinned commit)
RUN cd /tmp \
    && git clone https://github.com/thu-ml/SageAttention.git \
    && cd SageAttention \
    && git reset --hard 68de379 \
    && EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 pip install . \
    && rm -rf /tmp/SageAttention

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

COPY handler.py /handler.py
COPY comfy_api.py /comfy_api.py
COPY downloader.py /downloader.py
COPY s3_client.py /s3_client.py
COPY workflow.json /workflow.json
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
