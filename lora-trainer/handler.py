import logging
import os
import subprocess
import time
from pathlib import Path

import runpod
import requests

from s3_client import S3Client

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

logger = logging.getLogger("lora_trainer.handler")

GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")
logger.info("=== lora-trainer handler starting: commit=%s ===", GIT_COMMIT)

DATASET_DIR = "/image_dataset_here"
OUTPUT_DIR = "/output_folder/z_image_lora"
TRAIN_SCRIPT = "/non-interactive-train.sh"


def _download_archive(url: str, dest_path: str):
    logger.info("Downloading archive from %s", url)
    started_at = time.perf_counter()
    total_bytes = 0
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)
    duration = time.perf_counter() - started_at
    logger.info("Archive downloaded: bytes=%d duration=%.2fs", total_bytes, duration)


def _extract_and_cleanup(archive_path: str, extract_dir: str):
    logger.info("Extracting %s -> %s", archive_path, extract_dir)
    subprocess.run(["unzip", "-o", archive_path, "-d", extract_dir], check=True)
    os.remove(archive_path)
    logger.info("Archive extracted and deleted")

    files = list(Path(extract_dir).rglob("*"))
    logger.info("Dataset contents: %d files", len([f for f in files if f.is_file()]))


def _collect_epoch_files(output_dir: str) -> list[dict]:
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    # Find the run subdirectory (there should be one)
    run_dirs = [d for d in output_path.iterdir() if d.is_dir()]
    if not run_dirs:
        return []

    run_dir = run_dirs[0]
    logger.info("Found training run directory: %s", run_dir)

    epoch_files = []
    for epoch_dir in sorted(run_dir.iterdir()):
        if not epoch_dir.is_dir() or not epoch_dir.name.startswith("epoch"):
            continue
        adapter_file = epoch_dir / "adapter_model.safetensors"
        if adapter_file.exists():
            epoch_files.append({
                "path": str(adapter_file),
                "name": f"{epoch_dir.name}.safetensors",
            })
            logger.info("Found: %s -> %s", adapter_file, epoch_dir.name)

    return epoch_files


def handler(job):
    started_at = time.perf_counter()
    raw_job_id = job.get("id")
    logger.info("Job started: runpod_job_id=%s", raw_job_id)

    input_data = job["input"]
    archive_url = input_data["archive_url"]

    # Stage 0: clean up from previous runs (serverless reuse)
    logger.info("Stage 0: cleaning input/output directories")
    for d in [DATASET_DIR, OUTPUT_DIR]:
        subprocess.run(["rm", "-rf", d], check=True)
        os.makedirs(d, exist_ok=True)

    # Stage 2: download and extract archive
    logger.info("Stage 2: downloading and extracting dataset archive")
    archive_path = os.path.join(DATASET_DIR, "dataset.zip")
    try:
        _download_archive(archive_url, archive_path)
        _extract_and_cleanup(archive_path, DATASET_DIR)
    except Exception as e:
        logger.exception("Failed to download/extract dataset")
        return {"error": f"stage2/dataset: {e}"}

    # Stage 3: run training
    logger.info("Stage 3: starting training")
    try:
        result = subprocess.run(
            ["bash", TRAIN_SCRIPT],
            check=True,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hours max
        )
        logger.info("Training stdout:\n%s", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    except subprocess.TimeoutExpired:
        logger.error("Training timed out after 2 hours")
        return {"error": "stage3/training: timeout after 2 hours"}
    except subprocess.CalledProcessError as e:
        logger.error("Training failed: exit_code=%d stderr=%s", e.returncode, e.stderr[-2000:] if e.stderr else "")
        return {"error": f"stage3/training: exit_code={e.returncode}"}

    # Stage 4: upload results to S3
    logger.info("Stage 4: uploading results")
    try:
        epoch_files = _collect_epoch_files(OUTPUT_DIR)
        if not epoch_files:
            return {"error": "stage4/upload: no epoch files found in output directory"}

        logger.info("Found %d epoch files to upload", len(epoch_files))
        s3 = S3Client()
        job_id = str(raw_job_id or "unknown")
        uploaded = []

        for ef in epoch_files:
            with open(ef["path"], "rb") as f:
                data = f.read()
            object_key = f"lora-training/{job_id}/{ef['name']}"
            url = s3.upload_and_presign(data, object_key, content_type="application/octet-stream")
            uploaded.append({"filename": ef["name"], "url": url})
            logger.info("Uploaded %s (%d bytes)", ef["name"], len(data))

    except Exception as e:
        logger.exception("Upload failed")
        return {"error": f"stage4/upload: {e}"}

    total_seconds = time.perf_counter() - started_at
    logger.info("Job completed: runpod_job_id=%s epochs=%d duration=%.2fs", raw_job_id, len(uploaded), total_seconds)
    return {"outputs": uploaded}


runpod.serverless.start({"handler": handler})
