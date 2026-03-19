FROM hearmeman/comfyui-wan-template:v11

RUN pip install runpod requests

COPY handler.py /handler.py
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
