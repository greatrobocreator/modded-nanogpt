"""Run the NanoGPT speedrun on Modal GPUs.

One-time setup:
    uv tool install modal
    modal setup                                   # browser login, writes ~/.modal.toml
    modal run modal_train.py --download-only      # cache FineWeb tokens in a Modal volume

Run the current record (8xH100, ~$4-8 per run):
    modal run --detach modal_train.py

Cheap smoke run — 1xH100, first 20% of training, val eval every 50 steps (~$1):
    NANOGPT_GPU='H100!:1' modal run --detach modal_train.py \
        --stop-frac 0.2 --val-every 50 --run-id baseline-20pct

Variations:
    modal run modal_train.py --script train_gpt_medium.py --chunks 30   # GPT-2 medium track
    NANOGPT_GPU='H100!:2' modal run modal_train.py                      # 1/2/4/8 GPUs
    NANOGPT_TIMEOUT=7200 modal run modal_train.py                       # raise the 1h kill switch

The --stop-frac / --val-every / --run-evals / --run-id knobs are wired into train_gpt.py
via NANOGPT_* env vars (train_gpt_medium.py ignores them). Schedules always span the full
run, so a truncated run reproduces the first N% of a full run's dynamics exactly.

Logs land in the volume; list and fetch them with:
    modal volume ls modded-nanogpt-data logs
    modal volume get modded-nanogpt-data logs/<run_id>.txt
"""

import os
import shutil
import subprocess
from pathlib import Path

import modal

GPU_CONFIG = os.environ.get("NANOGPT_GPU", "H100!:8")
TIMEOUT_S = int(os.environ.get("NANOGPT_TIMEOUT", "3600"))
NPROC = int(GPU_CONFIG.rsplit(":", 1)[1]) if ":" in GPU_CONFIG else 1

REPO_ROOT = Path(__file__).parent
REMOTE_REPO = "/root/modded-nanogpt"
VOL_PATH = "/vol"

app = modal.App("modded-nanogpt")
volume = modal.Volume.from_name("modded-nanogpt-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "DATA_PATH": VOL_PATH,
            "HF_HOME": f"{VOL_PATH}/hf",
            "TORCHINDUCTOR_CACHE_DIR": f"{VOL_PATH}/cache/inductor",
            "TRITON_CACHE_DIR": f"{VOL_PATH}/cache/triton",
        }
    )
    .add_local_dir(
        REPO_ROOT,
        REMOTE_REPO,
        ignore=[".git", ".claude", "img", "logs", "records", "**/__pycache__", "data/fineweb10B"],
    )
)


@app.function(image=image, volumes={VOL_PATH: volume}, timeout=7200)
def download_data(num_chunks: int = 9) -> str:
    from huggingface_hub import hf_hub_download

    target = Path(VOL_PATH) / "data" / "fineweb10B"
    target.mkdir(parents=True, exist_ok=True)

    def get(fname: str) -> None:
        if not (target / fname).exists():
            hf_hub_download(
                repo_id="kjj0/fineweb10B-gpt2",
                filename=fname,
                repo_type="dataset",
                local_dir=str(target),
            )

    get("fineweb_val_%06d.bin" % 0)
    for i in range(1, num_chunks + 1):
        get("fineweb_train_%06d.bin" % i)
    volume.commit()
    return f"{len(list(target.glob('fineweb_train_*.bin')))} train chunks ready in {target}"


@app.function(image=image, gpu=GPU_CONFIG, volumes={VOL_PATH: volume}, timeout=TIMEOUT_S)
def train(
    script: str = "train_gpt.py",
    nproc: int = 8,
    stop_frac: float = 1.0,
    val_every: int = 0,
    run_evals: bool = False,
    run_id: str = "",
) -> str:
    os.chdir(REMOTE_REPO)
    env = os.environ.copy()
    if stop_frac < 1:
        env["NANOGPT_STOP_FRAC"] = str(stop_frac)
    if val_every:
        env["NANOGPT_VAL_EVERY"] = str(val_every)
    if run_evals:
        env["NANOGPT_RUN_EVALS"] = "1"
    if run_id:
        env["NANOGPT_RUN_ID"] = run_id
    subprocess.run(
        ["torchrun", "--standalone", f"--nproc_per_node={nproc}", script],
        check=True,
        env=env,
    )
    saved = []
    for src in sorted(Path("logs").iterdir()):
        dst = Path(VOL_PATH) / "logs" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        saved.append(src.name)
    volume.commit()
    logfiles = [Path("logs") / name for name in saved if name.endswith(".txt")]
    newest = max(logfiles, key=lambda p: p.stat().st_mtime)
    tail = "\n".join(newest.read_text().splitlines()[-8:])
    return f"saved to volume: {saved}\n--- tail of {newest.name} ---\n{tail}"


@app.local_entrypoint()
def main(
    script: str = "train_gpt.py",
    chunks: int = 9,
    download_only: bool = False,
    stop_frac: float = 1.0,
    val_every: int = 0,
    run_evals: bool = False,
    run_id: str = "",
) -> None:
    print(download_data.remote(chunks))
    if download_only:
        return
    print(
        train.remote(
            script=script,
            nproc=NPROC,
            stop_frac=stop_frac,
            val_every=val_every,
            run_evals=run_evals,
            run_id=run_id,
        )
    )
    print("fetch logs with: modal volume get modded-nanogpt-data logs/<run_id>.txt")
