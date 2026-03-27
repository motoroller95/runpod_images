FROM hearmeman/comfyui-wan-template:v11

RUN pip install runpod requests boto3

ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=${GIT_COMMIT}

COPY shared/comfy_api.py /comfy_api.py
COPY shared/downloader.py /downloader.py
COPY shared/s3_client.py /s3_client.py
COPY shared/handler_utils.py /handler_utils.py
COPY wan-2.2-animate/handler.py /handler.py
COPY wan-2.2-animate/workflow.json /workflow.json
COPY wan-2.2-animate/start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
