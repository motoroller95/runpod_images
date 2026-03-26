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

logger = logging.getLogger("klein_9b.handler")

GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")
logger.info("=== klein-9b handler starting: commit=%s ===", GIT_COMMIT)

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
WORKFLOW_PATH = os.getenv("WORKFLOW_PATH", "/workflow.json")
COMFYUI_LOG_PATH = os.getenv("COMFYUI_LOG_PATH", "/ComfyUI/log.log")
DOWNLOAD_WORKERS = int(os.getenv("DOWNLOAD_WORKERS", "4"))
COMFY_RESULT_TIMEOUT_SECONDS = int(os.getenv("COMFY_RESULT_TIMEOUT_SECONDS", "600"))

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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


def _error(job_id, stage: str, message: str, started_at: float, dump_logs: bool = False) -> dict:
    total_seconds = time.perf_counter() - started_at
    logger.error("Job failed at %s: runpod_job_id=%s reason=%s duration=%.2fs", stage, job_id, message, total_seconds)
    if dump_logs:
        _dump_comfyui_log()
    return {"error": f"{stage}: {message}"}


def _load_workflow_template() -> str:
    logger.info("Loading workflow template from %s", WORKFLOW_PATH)
    with open(WORKFLOW_PATH, encoding="utf-8") as fp:
        workflow_text = fp.read()
    logger.info("Workflow template loaded (%d bytes)", len(workflow_text))
    return workflow_text


def _build_workflow(prompt: str, face_filename: str, body_filename: str) -> dict:
    logger.info(
        "Building workflow: prompt_len=%d face=%s body=%s",
        len(prompt), face_filename, body_filename,
    )
    workflow_text = _load_workflow_template()

    replacements = {
        "{{input_prompt}}": prompt,
        "{{input_values_target_face}}": face_filename,
        "{{input_values_target_body}}": body_filename,
    }
    for placeholder, value in replacements.items():
        workflow_text = workflow_text.replace(placeholder, value)

    workflow = json.loads(workflow_text)
    logger.info("Workflow prepared and parsed successfully")
    return workflow


def handler(job):
    started_at = time.perf_counter()
    raw_job_id = job.get("id")
    job_id = str(raw_job_id or "unknown")
    logger.info("Job started: runpod_job_id=%s", raw_job_id)

    input_data = job["input"]
    prompt = input_data.get("prompt", "")
    reference_face = input_data["reference_face"]
    target_images = input_data["target_images"]

    # Stage 1: download all input files
    try:
        downloads = [reference_face] + target_images
        logger.info("Stage 1: downloading input files (count=%d workers=%d)", len(downloads), DOWNLOAD_WORKERS)
        download_files(downloads, max_workers=DOWNLOAD_WORKERS)
    except Exception as e:
        return _error(raw_job_id, "stage1/download", str(e), started_at)
    logger.info("Stage 1 complete: all files downloaded")

    comfy = ComfyApiClient(base_url=COMFYUI_URL)

    # Stage 2: wait for ComfyUI ready
    logger.info("Stage 2: waiting for ComfyUI readiness (url=%s)", COMFYUI_URL)
    if err := comfy.wait_for_ready():
        return _error(raw_job_id, "stage2/comfyui_ready", err, started_at, dump_logs=True)
    logger.info("Stage 2 complete: ComfyUI is ready")

    # Stage 3: process each target image
    s3 = S3Client()
    face_filename = Path(reference_face["destination"]).name
    total = len(target_images)
    results = []

    for index, target in enumerate(target_images):
        target_filename = Path(target["destination"]).name
        logger.info("Processing target image %d/%d: %s", index + 1, total, target_filename)

        try:
            # Build workflow
            body_filename = target_filename
            workflow = _build_workflow(prompt, face_filename, body_filename)

            # Queue prompt
            prompt_id, err = comfy.queue_prompt(workflow)
            if err:
                raise RuntimeError(f"queue_prompt failed: {err}")

            # Wait for result
            outputs, err = comfy.wait_for_result(prompt_id, timeout_seconds=COMFY_RESULT_TIMEOUT_SECONDS)
            if err:
                raise RuntimeError(f"wait_for_result failed: {err}")

            media_items = extract_media_items(outputs)
            logger.info("Target %d/%d: got %d media items", index + 1, total, len(media_items))

            # Upload each output
            for item in media_items:
                media_bytes, err = comfy.fetch_output_binary(item)
                if err:
                    raise RuntimeError(f"fetch_output_binary failed: {err}")

                content_type = mimetypes.guess_type(item["filename"])[0] or "application/octet-stream"
                stem = Path(target_filename).stem
                extension = Path(item["filename"]).suffix
                object_key = f"run/{job_id}/{index}_{stem}{extension}"

                url = s3.upload_and_presign(media_bytes, object_key, content_type=content_type)
                results.append({
                    "index": index,
                    "target_filename": target_filename,
                    "filename": f"{index}_{stem}{extension}",
                    "url": url,
                })
                logger.info("Uploaded: s3_key=%s bytes=%d", object_key, len(media_bytes))

        except Exception as e:
            logger.exception("Failed processing target %d/%d (%s)", index + 1, total, target_filename)
            results.append({
                "index": index,
                "target_filename": target_filename,
                "error": str(e),
            })

    total_seconds = time.perf_counter() - started_at
    success_count = sum(1 for r in results if "url" in r)
    error_count = sum(1 for r in results if "error" in r)
    logger.info(
        "Job completed: runpod_job_id=%s total=%d success=%d errors=%d duration=%.2fs",
        raw_job_id, total, success_count, error_count, total_seconds,
    )
    time.sleep(3600)
    return {"outputs": results}


runpod.serverless.start({"handler": handler})
