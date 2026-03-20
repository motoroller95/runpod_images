import logging
import time

import requests

logger = logging.getLogger("wan_animate.comfy_api")


class ComfyApiClient:
    def __init__(self, base_url: str, poll_interval_seconds: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.poll_interval_seconds = poll_interval_seconds

    def wait_for_ready(self, timeout_seconds: int = 300) -> None:
        logger.info("ComfyUI readiness check started: base_url=%s timeout=%ss", self.base_url, timeout_seconds)
        deadline = time.time() + timeout_seconds
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            try:
                response = requests.get(f"{self.base_url}/system_stats", timeout=2)
                response.raise_for_status()
                logger.info("ComfyUI ready after %d attempts", attempts)
                return
            except requests.RequestException:
                time.sleep(1)
        logger.error("ComfyUI readiness timeout after %d attempts", attempts)
        raise TimeoutError(f"ComfyUI did not become ready within {timeout_seconds} seconds")

    def queue_prompt(self, workflow: dict) -> str:
        logger.info("Queueing prompt in ComfyUI")
        response = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        logger.info("Prompt queued: prompt_id=%s", prompt_id)
        return prompt_id

    def wait_for_result(self, prompt_id: str, timeout_seconds: int = 7200) -> dict:
        logger.info("Waiting for ComfyUI result: prompt_id=%s timeout=%ss", prompt_id, timeout_seconds)
        deadline = time.time() + timeout_seconds
        polls = 0
        last_log_at = time.time()
        while time.time() < deadline:
            polls += 1
            response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=15)
            response.raise_for_status()
            payload = response.json()
            prompt_result = payload.get(prompt_id)
            if prompt_result:
                status = prompt_result.get("status", {}).get("status_str")
                if status == "error":
                    logger.error("ComfyUI returned error status for prompt_id=%s", prompt_id)
                    raise RuntimeError(f"ComfyUI returned an error for prompt {prompt_id}: {prompt_result}")
                outputs = prompt_result.get("outputs", {})
                logger.info("ComfyUI result ready: prompt_id=%s polls=%d output_nodes=%d", prompt_id, polls, len(outputs))
                return outputs
            now = time.time()
            if now - last_log_at >= 30:
                logger.info("Still waiting for ComfyUI result: prompt_id=%s polls=%d", prompt_id, polls)
                last_log_at = now
            time.sleep(self.poll_interval_seconds)
        logger.error("ComfyUI result timeout: prompt_id=%s polls=%d", prompt_id, polls)
        raise TimeoutError(f"ComfyUI result timeout for prompt {prompt_id}")

    def fetch_output_binary(self, item: dict) -> bytes:
        logger.info(
            "Fetching ComfyUI output binary: filename=%s subfolder=%s type=%s",
            item["filename"],
            item.get("subfolder", ""),
            item.get("type", "output"),
        )
        response = requests.get(
            f"{self.base_url}/view",
            params={
                "filename": item["filename"],
                "subfolder": item.get("subfolder", ""),
                "type": item.get("type", "output"),
            },
            timeout=120,
        )
        response.raise_for_status()
        content = response.content
        logger.info("Fetched output binary: filename=%s bytes=%d", item["filename"], len(content))
        return content


def extract_media_items(outputs: dict) -> list[dict]:
    media_items: list[dict] = []
    for node_output in outputs.values():
        for bucket in ("images", "videos", "gifs"):
            for item in node_output.get(bucket, []):
                media_items.append(item)
    logger.info("Extracted media items from ComfyUI outputs: total=%d", len(media_items))
    return media_items
