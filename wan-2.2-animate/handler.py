import json
import logging
import mimetypes
import os
import time
from pathlib import Path

import runpod

from comfy_api import ComfyApiClient, extract_media_items
from downloader import download_files
from s3_client import S3Client

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger("wan_animate.handler")

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
WORKFLOW_PATH = os.getenv("WORKFLOW_PATH", "/workflow.json")
COMFYUI_LOG_PATH = os.getenv("COMFYUI_LOG_PATH", "/ComfyUI/log.log")
DOWNLOAD_WORKERS = int(os.getenv("DOWNLOAD_WORKERS", "4"))
COMFY_RESULT_TIMEOUT_SECONDS = int(os.getenv("COMFY_RESULT_TIMEOUT_SECONDS", "7200"))

LORA_EXTENSIONS = {".safetensors"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}


def _dump_comfyui_log(tail_lines: int = 100) -> None:
    log_path = Path(COMFYUI_LOG_PATH)
    if not log_path.exists():
        logger.error("ComfyUI log not found at %s", COMFYUI_LOG_PATH)
        return
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-tail_lines:]
    logger.error("=== ComfyUI log (last %d lines of %d) ===", len(tail), len(lines))
    for line in tail:
        logger.error("[comfyui] %s", line)
    logger.error("=== end of ComfyUI log ===")


def _pick_destination(destinations: list[str], allowed_extensions: set[str]) -> str:
    return next(
        destination
        for destination in destinations
        if Path(destination).suffix.lower() in allowed_extensions
    )


def _load_workflow_template() -> str:
    logger.info("Loading workflow template from %s", WORKFLOW_PATH)
    with open(WORKFLOW_PATH, encoding="utf-8") as fp:
        workflow_text = fp.read()
    logger.info("Workflow template loaded (%d bytes)", len(workflow_text))
    return workflow_text


def _build_workflow(input_data: dict, downloaded_files: list[dict]) -> dict:
    logger.info("Resolving workflow placeholders from downloaded files")
    destinations = [item["destination"] for item in downloaded_files]
    lora_destination = _pick_destination(destinations, LORA_EXTENSIONS)
    image_destination = _pick_destination(destinations, IMAGE_EXTENSIONS)
    video_destination = _pick_destination(destinations, VIDEO_EXTENSIONS)

    video_filename = Path(video_destination).name
    reference_video_format = mimetypes.guess_type(video_filename)[0] or "video/mp4"

    replacements = {
        "{{input_values_lora}}": Path(lora_destination).name,
        "{{input_values_reference_image}}": Path(image_destination).name,
        "{{input_value_reference_video}}": video_filename,
        "{{reference_video_format}}": reference_video_format,
        "{{input_values_positive_prompt}}": input_data["positive_prompt"],
        "{{input_values_negative_prompt}}": input_data["negative_prompt"],
    }

    workflow_text = _load_workflow_template()
    logger.info(
        "Applying placeholders: lora=%s image=%s video=%s video_format=%s positive_len=%d negative_len=%d",
        replacements["{{input_values_lora}}"],
        replacements["{{input_values_reference_image}}"],
        replacements["{{input_value_reference_video}}"],
        replacements["{{reference_video_format}}"],
        len(replacements["{{input_values_positive_prompt}}"]),
        len(replacements["{{input_values_negative_prompt}}"]),
    )
    for placeholder, value in replacements.items():
        workflow_text = workflow_text.replace(placeholder, value)
    workflow = json.loads(workflow_text)
    logger.info("Workflow prepared and parsed successfully")
    return workflow


def _build_output_key(job_id: str, filename: str, seen_names: dict[str, int]) -> tuple[str, str]:
    base_name = Path(filename).name
    counter = seen_names.get(base_name, 0)
    seen_names[base_name] = counter + 1

    if counter == 0:
        final_name = base_name
    else:
        stem = Path(base_name).stem
        suffix = Path(base_name).suffix
        final_name = f"{stem}_{counter}{suffix}"

    return f"run/{job_id}/{final_name}", final_name


def handler(job):
    started_at = time.perf_counter()
    raw_job_id = job.get("id")
    logger.info("Job started: runpod_job_id=%s", raw_job_id)

    try:
        input_data = job["input"]
        downloads = input_data["downloads"]
        logger.info("Stage 1/6: downloading input files (count=%d workers=%d)", len(downloads), DOWNLOAD_WORKERS)
        downloaded_files = download_files(downloads, max_workers=DOWNLOAD_WORKERS)
        logger.info(
            "Stage 1/6 complete: downloaded files -> %s",
            [item["destination"] for item in downloaded_files],
        )

        logger.info("Stage 2/6: building workflow with runtime placeholders")
        workflow = _build_workflow(input_data, downloaded_files)
        logger.info("Stage 2/6 complete")

        logger.info("Stage 3/6: waiting for ComfyUI readiness (url=%s)", COMFYUI_URL)
        comfy = ComfyApiClient(base_url=COMFYUI_URL)
        try:
            comfy.wait_for_ready()
        except TimeoutError:
            _dump_comfyui_log()
            raise
        logger.info("Stage 3/6 complete: ComfyUI is ready")

        logger.info("Stage 4/6: queueing workflow prompt")
        try:
            prompt_id = comfy.queue_prompt(workflow)
        except Exception:
            _dump_comfyui_log(tail_lines=100)
            raise
        logger.info("Stage 4/6 complete: prompt_id=%s", prompt_id)

        logger.info("Stage 5/6: waiting for ComfyUI outputs (timeout=%ss)", COMFY_RESULT_TIMEOUT_SECONDS)
        outputs = comfy.wait_for_result(prompt_id, timeout_seconds=COMFY_RESULT_TIMEOUT_SECONDS)
        media_items = extract_media_items(outputs)
        logger.info("Stage 5/6 complete: media_items=%d", len(media_items))

        logger.info("Stage 6/6: uploading outputs to S3")
        s3 = S3Client()
        job_id = str(raw_job_id or prompt_id)
        seen_names: dict[str, int] = {}
        uploaded_outputs = []

        for index, item in enumerate(media_items, start=1):
            filename = item["filename"]
            logger.info(
                "Uploading output %d/%d: filename=%s subfolder=%s type=%s",
                index,
                len(media_items),
                filename,
                item.get("subfolder", ""),
                item.get("type", "output"),
            )
            media_bytes = comfy.fetch_output_binary(item)
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            object_key, final_name = _build_output_key(job_id, filename, seen_names)
            url = s3.upload_and_presign(media_bytes, object_key, content_type=content_type)
            uploaded_outputs.append({"filename": final_name, "url": url})
            logger.info(
                "Uploaded output %d/%d: s3_key=%s filename=%s bytes=%d",
                index,
                len(media_items),
                object_key,
                final_name,
                len(media_bytes),
            )

        total_seconds = time.perf_counter() - started_at
        logger.info(
            "Job completed successfully: runpod_job_id=%s prompt_id=%s outputs=%d duration=%.2fs",
            raw_job_id,
            prompt_id,
            len(uploaded_outputs),
            total_seconds,
        )
        return {"outputs": uploaded_outputs}
    except Exception:
        total_seconds = time.perf_counter() - started_at
        logger.exception(
            "Job failed: runpod_job_id=%s duration=%.2fs",
            raw_job_id,
            total_seconds,
        )
        raise


runpod.serverless.start({"handler": handler})
