FROM hearmeman/diffusion-pipe:v11

ENV NETWORK_VOLUME=/

RUN apt-get update && apt-get install -y --no-install-recommends unzip && rm -rf /var/lib/apt/lists/*

RUN pip install runpod requests boto3

# ============================================================
# Workspace setup (from original start.sh, baked into image)
# ============================================================

# Copy runpod-diffusion_pipe repo into image
COPY lora-trainer/runpod-diffusion_pipe /tmp/runpod-diffusion_pipe

# Move directories to root (NETWORK_VOLUME=/)
RUN mv /tmp/runpod-diffusion_pipe/Captioning / 2>/dev/null || true && \
    mv /tmp/runpod-diffusion_pipe/wan2.2_lora_training / 2>/dev/null || true && \
    mv /tmp/runpod-diffusion_pipe/interactive_start_training.sh / 2>/dev/null && chmod +x /interactive_start_training.sh || true && \
    mv /tmp/runpod-diffusion_pipe/HowToUse.txt / 2>/dev/null || true && \
    if [ -f /tmp/runpod-diffusion_pipe/send_lora.sh ]; then chmod +x /tmp/runpod-diffusion_pipe/send_lora.sh && cp /tmp/runpod-diffusion_pipe/send_lora.sh /usr/local/bin/; fi

# Move diffusion_pipe to root if not already there
RUN if [ -d /diffusion_pipe ] && [ -d /diffusion_pipe/.git ]; then \
        cd /diffusion_pipe && git pull || true; \
    fi

# Copy dataset.toml into diffusion_pipe examples
RUN if [ -d /diffusion_pipe/examples ]; then \
        rm -rf /diffusion_pipe/examples/* && \
        mv /tmp/runpod-diffusion_pipe/dataset.toml /diffusion_pipe/examples/ 2>/dev/null || true; \
    fi

# Patch toml files: with NETWORK_VOLUME=/ paths become /models/, /training_outputs/, etc.
RUN TOML_DIR="/tmp/runpod-diffusion_pipe/toml_files" && \
    if [ -d "$TOML_DIR" ]; then \
        for toml_file in "$TOML_DIR"/*.toml; do \
            if [ -f "$toml_file" ]; then \
                sed -i "s|output_dir = '/data/|output_dir = '/training_outputs/|g" "$toml_file" && \
                sed -i "s|ckpt_path = '/Wan/|ckpt_path = '/models/Wan/|g" "$toml_file"; \
            fi; \
        done && \
        cp "$TOML_DIR"/*.toml /tmp/runpod-diffusion_pipe/toml_files_patched/ 2>/dev/null || \
        mkdir -p /toml_files && cp "$TOML_DIR"/*.toml /toml_files/; \
    fi

# Patch dataset.toml path
RUN if [ -f /diffusion_pipe/examples/dataset.toml ]; then \
        sed -i "s|\\\$NETWORK_VOLUME/image_dataset_here|/image_dataset_here|g" /diffusion_pipe/examples/dataset.toml && \
        sed -i "s|\\\$NETWORK_VOLUME/video_dataset_here|/video_dataset_here|g" /diffusion_pipe/examples/dataset.toml; \
    fi

# Create dataset directories
RUN mkdir -p /image_dataset_here /video_dataset_here /training_outputs /logs

# Keep runpod-diffusion_pipe for reference, clean up
RUN mv /tmp/runpod-diffusion_pipe /runpod-diffusion_pipe

# ============================================================
# Handler and entrypoint
# ============================================================

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

COPY shared/s3_client.py /s3_client.py
COPY shared/downloader.py /downloader.py
COPY lora-trainer/handler.py /handler.py
COPY lora-trainer/train.py /train.py
COPY lora-trainer/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
