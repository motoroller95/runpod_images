FROM aiorbust/flux2-klein-9b:latest

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        aria2 curl \
    && rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip install --no-cache-dir runpod requests boto3 onnxruntime-gpu

# SageAttention: clone at build, build at runtime (needs GPU)
RUN cd /tmp \
    && git clone https://github.com/thu-ml/SageAttention.git \
    && cd SageAttention \
    && git reset --hard 68de379

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

COPY shared/comfy_api.py /comfy_api.py
COPY shared/downloader.py /downloader.py
COPY shared/s3_client.py /s3_client.py
COPY shared/handler_utils.py /handler_utils.py
COPY aiorbust-klein-9b/handler.py /handler.py
COPY aiorbust-klein-9b/workflow.json /workflow.json
COPY aiorbust-klein-9b/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
