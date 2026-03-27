import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

COMFYUI_LOG_PATH = os.getenv("COMFYUI_LOG_PATH", "/ComfyUI/log.log")
WORKFLOW_PATH = os.getenv("WORKFLOW_PATH", "/workflow.json")

_INPUT_DIR = Path("/ComfyUI/input")
_OUTPUT_DIR = Path("/ComfyUI/output")


def _cleanup_dirs():
    """Remove stale files from previous runs to prevent ComfyUI caching."""
    for d in [_INPUT_DIR, _OUTPUT_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        logger.info("Cleaned directory: %s", d)


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
