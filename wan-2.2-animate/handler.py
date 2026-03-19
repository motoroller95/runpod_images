import runpod
import subprocess
import requests
import time
import base64
import os

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_PATH = "/ComfyUI"

def wait_for_comfyui(timeout=300):
    print("Ожидаем запуска ComfyUI...")
    for i in range(timeout):
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=2)
            if r.status_code == 200:
                print(f"ComfyUI готов за {i} секунд")
                return True
        except:
            time.sleep(1)
    raise Exception(f"ComfyUI не поднялся за {timeout} секунд")

def queue_prompt(workflow: dict) -> str:
    r = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
    r.raise_for_status()
    return r.json()["prompt_id"]

def wait_for_result(prompt_id: str) -> dict:
    while True:
        r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        data = r.json()
        if prompt_id in data:
            if data[prompt_id].get("status", {}).get("status_str") == "error":
                raise Exception(f"ComfyUI ошибка: {data[prompt_id]}")
            return data[prompt_id]["outputs"]
        time.sleep(2)

def handler(job):
    input_data = job["input"]

    # CMD режим
    if "cmd" in input_data:
        result = subprocess.run(
            input_data["cmd"],
            shell=True,
            capture_output=True,
            text=True
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }

    # ComfyUI workflow режим
    workflow = input_data["workflow"]
    prompt_id = queue_prompt(workflow)
    print(f"Промпт поставлен в очередь: {prompt_id}")

    outputs = wait_for_result(prompt_id)

    results = []
    for node_id, node_output in outputs.items():
        for key in ["images", "videos", "gifs"]:
            if key in node_output:
                for item in node_output[key]:
                    filename = item["filename"]
                    subfolder = item.get("subfolder", "")
                    r = requests.get(
                        f"{COMFYUI_URL}/view",
                        params={"filename": filename, "subfolder": subfolder, "type": "output"}
                    )
                    results.append({
                        "filename": filename,
                        "data": base64.b64encode(r.content).decode()
                    })

    return {"outputs": results}


# Запуск ComfyUI при старте воркера
print("Запускаем ComfyUI...")
subprocess.Popen(
    ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--disable-auto-launch"],
    cwd=COMFYUI_PATH
)
wait_for_comfyui(timeout=300)

runpod.serverless.start({"handler": handler})
