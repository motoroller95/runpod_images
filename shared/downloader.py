from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import time
from urllib.parse import urlsplit, urlunsplit

import requests

logger = logging.getLogger(__name__)


def _sanitize_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _download_one(url: str, destination: str) -> dict:
    started_at = time.perf_counter()
    safe_url = _sanitize_url(url)
    logger.info("Download started: url=%s destination=%s", safe_url, destination)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    try:
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with destination_path.open("wb") as target:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        target.write(chunk)
                        total_bytes += len(chunk)
    except Exception:
        logger.exception("Download failed: url=%s destination=%s", safe_url, destination)
        raise

    duration = time.perf_counter() - started_at
    logger.info(
        "Download finished: url=%s destination=%s bytes=%d duration=%.2fs",
        safe_url,
        destination,
        total_bytes,
        duration,
    )

    return {"url": url, "destination": str(destination_path)}


def download_files(downloads: list[dict], max_workers: int = 4) -> list[dict]:
    started_at = time.perf_counter()
    logger.info("Batch download started: files=%d workers=%d", len(downloads), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_download_one, item["url"], item["destination"])
            for item in downloads
        ]
        results = [future.result() for future in futures]
    duration = time.perf_counter() - started_at
    logger.info("Batch download finished: files=%d duration=%.2fs", len(results), duration)
    return results
