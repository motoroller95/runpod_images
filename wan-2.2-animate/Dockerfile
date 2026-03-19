FROM hearmeman/comfyui-wan-template:v11

RUN pip install runpod requests

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
