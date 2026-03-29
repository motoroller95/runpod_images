"""
Non-interactive LoRA training script for Z Image Turbo.
Replaces non-interactive-train.sh with proper logging and error handling.
"""

import logging
import os
import re
import subprocess
import time
from pathlib import Path

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("lora_trainer.train")

NETWORK_VOLUME = os.environ.get("NETWORK_VOLUME", "/")
TOML_FILE = "z_image_toml.toml"
MODEL_FILES = [
    "z_image_turbo_bf16.safetensors",
    "ae.safetensors",
    "qwen_3_4b.safetensors",
    "zimage_turbo_training_adapter_v2.safetensors",
]
HF_REPO = "Hearme-man/z_image_turbo"


def check_flash_attn():
    """Wait for flash-attn compilation if still running."""
    if Path("/tmp/flash_attn_wheel_success").exists():
        logger.info("flash-attn: installed (prebuilt wheel)")
        return

    pid_file = Path("/tmp/flash_attn_pid")
    if not pid_file.exists():
        logger.info("flash-attn: no build in progress")
        return

    pid = int(pid_file.read_text().strip())
    try:
        os.kill(pid, 0)
    except OSError:
        pid_file.unlink(missing_ok=True)
        logger.info("flash-attn: build process already finished")
        return

    logger.info("flash-attn: still compiling (PID=%d), waiting...", pid)
    while True:
        try:
            os.kill(pid, 0)
            time.sleep(2)
        except OSError:
            break

    pid_file.unlink(missing_ok=True)
    logger.info("flash-attn: compilation finished")


def check_cuda():
    """Verify CUDA is available."""
    logger.info("Checking CUDA availability...")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA NOT AVAILABLE — deploy with CUDA 12.8")
    x = torch.randn(1, device="cuda") * 2
    logger.info("CUDA check passed: %s", torch.cuda.get_device_name(0))


def setup_toml():
    """Copy and patch z_image_toml.toml into diffusion_pipe/examples/."""
    examples_dir = Path(NETWORK_VOLUME) / "diffusion_pipe" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    dest = examples_dir / TOML_FILE
    if dest.exists():
        logger.info("TOML: %s already in examples", TOML_FILE)
        return

    source = Path(NETWORK_VOLUME) / "runpod-diffusion_pipe" / "toml_files" / TOML_FILE
    if not source.exists():
        raise FileNotFoundError(f"TOML not found: {source}")

    text = source.read_text()
    output_dir = str(Path(NETWORK_VOLUME) / "output_folder" / "z_image_lora")
    text = re.sub(r"^output_dir = .*$", f"output_dir = '{output_dir}'", text, flags=re.MULTILINE)
    dest.write_text(text)
    logger.info("TOML: copied and patched -> %s (output_dir=%s)", dest, output_dir)


def download_models():
    """Download model files from HuggingFace if not already present."""
    models_dir = Path(NETWORK_VOLUME) / "models" / "z_image"
    models_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in MODEL_FILES if not (models_dir / f).exists()]
    if not missing:
        logger.info("Models: all files present, skipping download")
        return

    logger.info("Models: downloading %d missing files from %s", len(missing), HF_REPO)
    from huggingface_hub import hf_hub_download
    for filename in missing:
        logger.info("Models: downloading %s ...", filename)
        hf_hub_download(
            repo_id=HF_REPO,
            filename=filename,
            local_dir=str(models_dir),
        )
        logger.info("Models: downloaded %s", filename)

    # Verify
    still_missing = [f for f in MODEL_FILES if not (models_dir / f).exists()]
    if still_missing:
        raise RuntimeError(f"Model files missing after download: {still_missing}")

    logger.info("Models: download completed and verified")


def setup_dataset_toml():
    """Patch dataset.toml paths."""
    dataset_toml = Path(NETWORK_VOLUME) / "diffusion_pipe" / "examples" / "dataset.toml"
    if not dataset_toml.exists():
        logger.warning("dataset.toml not found at %s", dataset_toml)
        return

    text = dataset_toml.read_text()
    text = text.replace("$NETWORK_VOLUME/image_dataset_here", f"{NETWORK_VOLUME}/image_dataset_here")
    text = text.replace("$NETWORK_VOLUME/video_dataset_here", f"{NETWORK_VOLUME}/video_dataset_here")
    # Also handle already-patched paths from Dockerfile
    text = text.replace("/image_dataset_here", f"{NETWORK_VOLUME}/image_dataset_here".replace("//", "/"))
    text = text.replace("/video_dataset_here", f"{NETWORK_VOLUME}/video_dataset_here".replace("//", "/"))
    dataset_toml.write_text(text)
    logger.info("Dataset TOML: paths patched")


def run_training():
    """Launch DeepSpeed training."""
    diffusion_pipe_dir = Path(NETWORK_VOLUME) / "diffusion_pipe"
    if not diffusion_pipe_dir.exists():
        raise RuntimeError(f"diffusion_pipe not found at {diffusion_pipe_dir}")

    config_path = f"examples/{TOML_FILE}"
    config_full = diffusion_pipe_dir / config_path
    if not config_full.exists():
        raise RuntimeError(f"Training config not found: {config_full}")

    # Log training summary
    text = config_full.read_text()
    epochs = re.search(r"^epochs\s*=\s*(\d+)", text, re.MULTILINE)
    save_every = re.search(r"^save_every_n_epochs\s*=\s*(\d+)", text, re.MULTILINE)
    rank = re.search(r"^rank\s*=\s*(\d+)", text, re.MULTILINE)
    lr = re.search(r"^lr\s*=\s*(.+)$", text, re.MULTILINE)
    logger.info(
        "Training config: epochs=%s save_every=%s rank=%s lr=%s",
        epochs.group(1) if epochs else "?",
        save_every.group(1) if save_every else "?",
        rank.group(1) if rank else "?",
        lr.group(1).strip() if lr else "?",
    )

    logger.info("Launching DeepSpeed training...")
    env = os.environ.copy()
    env["NCCL_P2P_DISABLE"] = "1"
    env["NCCL_IB_DISABLE"] = "1"

    proc = subprocess.run(
        ["deepspeed", "--num_gpus=1", "train.py", "--deepspeed", "--config", config_path],
        cwd=str(diffusion_pipe_dir),
        env=env,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"DeepSpeed training failed (exit_code={proc.returncode})")

    logger.info("Training completed successfully")


def main():
    started_at = time.perf_counter()
    logger.info("=== LoRA Training Script Starting ===")

    check_flash_attn()
    check_cuda()
    setup_toml()
    download_models()
    setup_dataset_toml()
    run_training()

    duration = time.perf_counter() - started_at
    logger.info("=== LoRA Training Script Finished: duration=%.2fs ===", duration)


if __name__ == "__main__":
    main()
