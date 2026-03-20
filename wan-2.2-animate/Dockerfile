FROM hearmeman/comfyui-wan-template:v11

RUN pip install runpod requests boto3

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
