import logging
import time

import requests

logger = logging.getLogger("wan_animate.comfy_api")


class ComfyApiClient:
    def __init__(self, base_url: str, poll_interval_seconds: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.poll_interval_seconds = poll_interval_seconds

    def wait_for_ready(self, timeout_seconds: int = 300) -> str | None:
        """Returns None on success, error message string on failure."""
        logger.info("ComfyUI readiness check started: base_url=%s timeout=%ss", self.base_url, timeout_seconds)
        deadline = time.time() + timeout_seconds
        attempts = 0
        while time.time() < deadline:
            attempts += 1
            try:
                response = requests.get(f"{self.base_url}/system_stats", timeout=2)
                response.raise_for_status()
                logger.info("ComfyUI ready after %d attempts", attempts)
                return None
            except requests.RequestException:
                time.sleep(1)
        msg = f"ComfyUI did not become ready within {timeout_seconds}s after {attempts} attempts"
        logger.error(msg)
        return msg

    def queue_prompt(self, workflow: dict) -> tuple[str | None, str | None]:
        """Returns (prompt_id, None) on success or (None, error_message) on failure."""
        logger.info("Queueing prompt in ComfyUI")
        try:
            response = requests.post(
                f"{self.base_url}/prompt",
                json={"prompt": workflow},
                timeout=30,
            )
        except requests.RequestException as e:
            msg = f"ComfyUI /prompt request failed: {e}"
            logger.error(msg)
            return None, msg

        if not response.ok:
            msg = f"ComfyUI /prompt rejected (status={response.status_code}): {response.text[:2000]}"
            logger.error(msg)
            return None, msg

        prompt_id = response.json()["prompt_id"]
        logger.info("Prompt queued: prompt_id=%s", prompt_id)
        return prompt_id, None

    def wait_for_result(self, prompt_id: str, timeout_seconds: int = 7200) -> tuple[dict | None, str | None]:
        """Returns (outputs, None) on success or (None, error_message) on failure/timeout."""
        logger.info("Waiting for ComfyUI result: prompt_id=%s timeout=%ss", prompt_id, timeout_seconds)
        deadline = time.time() + timeout_seconds
        polls = 0
        last_log_at = time.time()

        while time.time() < deadline:
            polls += 1
            try:
                response = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=60)
                response.raise_for_status()
            except requests.exceptions.Timeout:
                logger.warning(
                    "ComfyUI /history poll timed out (busy), retrying: prompt_id=%s polls=%d",
                    prompt_id,
                    polls,
                )
                time.sleep(self.poll_interval_seconds)
                continue
            except requests.RequestException as e:
                logger.warning("ComfyUI /history poll error: %s, retrying: polls=%d", e, polls)
                time.sleep(self.poll_interval_seconds)
                continue

            payload = response.json()
            prompt_result = payload.get(prompt_id)
            if prompt_result:
                status = prompt_result.get("status", {}).get("status_str")
                if status == "error":
                    msg = f"ComfyUI returned error for prompt {prompt_id}: {prompt_result.get('status')}"
                    logger.error(msg)
                    return None, msg
                outputs = prompt_result.get("outputs", {})
                logger.info("ComfyUI result ready: prompt_id=%s polls=%d output_nodes=%d", prompt_id, polls, len(outputs))
                return outputs, None

            now = time.time()
            if now - last_log_at >= 30:
                logger.info("Still waiting for ComfyUI result: prompt_id=%s polls=%d", prompt_id, polls)
                last_log_at = now
            time.sleep(self.poll_interval_seconds)

        msg = f"ComfyUI result timeout after {timeout_seconds}s, polls={polls}"
        logger.error(msg)
        return None, msg

    def fetch_output_binary(self, item: dict) -> tuple[bytes | None, str | None]:
        """Returns (bytes, None) on success or (None, error_message) on failure."""
        logger.info(
            "Fetching ComfyUI output binary: filename=%s subfolder=%s type=%s",
            item["filename"],
            item.get("subfolder", ""),
            item.get("type", "output"),
        )
        try:
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
        except requests.RequestException as e:
            msg = f"Failed to fetch output binary {item['filename']}: {e}"
            logger.error(msg)
            return None, msg

        content = response.content
        logger.info("Fetched output binary: filename=%s bytes=%d", item["filename"], len(content))
        return content, None


def extract_media_items(outputs: dict) -> list[dict]:
    media_items: list[dict] = []
    for node_output in outputs.values():
        for bucket in ("images", "videos", "gifs"):
            for item in node_output.get(bucket, []):
                media_items.append(item)
    logger.info("Extracted media items from ComfyUI outputs: total=%d", len(media_items))
    return media_items
