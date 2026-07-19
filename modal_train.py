"""Run the NanoGPT speedrun on Modal GPUs.

One-time setup:
    uv tool install modal
    modal setup                                   # browser login, writes ~/.modal.toml
    modal run modal_train.py --download-only      # cache FineWeb tokens in a Modal volume

Run the current record (8xH100, ~$4-8 per run):
    modal run --detach modal_train.py

Cheap smoke run — 1xH100, first third of training (all of stage 1), sparse val (~$1):
    NANOGPT_GPU='H100!:1' modal run --detach modal_train.py \
        --stop-frac 0.33 --val-every 155 --run-id baseline-33pct

    Keep --val-every sparse on truncated runs: val_tokens is fixed at 10.5M while the
    stage-1 train batch is only ~131K tok/step, so each val eval costs ~30 training steps.
    A few points (start/mid/end) beat every-50. stop_frac 0.33 covers all of stage 1;
    the first 20% (~278 steps) never leaves it.

Multi-token smear experiment (experiments/multi-token-smear/PLAN.md), run naming
smear-k{K}-{smoke|full}-r{repeat}:
    NANOGPT_GPU='H100!:1' modal run --detach modal_train.py \
        --stop-frac 0.33 --val-every 155 --smear-k 3 --run-id smear-k3-smoke-r1

Variations:
    modal run modal_train.py --script train_gpt_medium.py --chunks 30   # GPT-2 medium track
    NANOGPT_GPU='H100!:2' modal run modal_train.py                      # 1/2/4/8 GPUs
    NANOGPT_TIMEOUT=7200 modal run modal_train.py                       # pin the kill switch (else auto)

The kill-switch timeout is computed per run from stop_frac and GPU count (see _timeout_seconds):
a fixed compile+warmup allowance plus training time that scales with the run, instead of a flat 1h.

The --stop-frac / --val-every / --run-evals / --run-id / --smear-k knobs are wired into
train_gpt.py via NANOGPT_* env vars (train_gpt_medium.py ignores them). Schedules always span
the full run, so a truncated run reproduces the first N% of a full run's dynamics exactly.

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
NPROC = int(GPU_CONFIG.rsplit(":", 1)[1]) if ":" in GPU_CONFIG else 1

# Full (stop_frac=1) pure-training wall-clock estimate on ONE H100. Observed stage-1 rate is
# ~241 ms/step; later stages use 2-3x larger batches (~2-4x/step), so a full ~1390-step run lands
# around 15-20 min on one H100. Override the timeout entirely with NANOGPT_TIMEOUT.
FULL_TRAIN_MIN = 18.0

# Every fresh container pays a torch.compile + Triton/inductor autotune + schedule-warmup phase
# before step 0. The volume inductor cache speeds kernel *compilation* but does NOT remove this
# phase (observed 7-13+ min, high variance), so the timeout always budgets for it.
COMPILE_WARMUP_MIN = 15.0


def _timeout_seconds(stop_frac: float, nproc: int) -> int:
    """Compile+warmup allowance plus training time scaled by stop_frac/GPU count, with margin."""
    train_min = (FULL_TRAIN_MIN / max(nproc, 1)) * min(stop_frac, 1.0)
    return int((COMPILE_WARMUP_MIN + train_min) * 1.3 * 60)


TIMEOUT_S = int(os.environ.get("NANOGPT_TIMEOUT", "0")) or _timeout_seconds(1.0, NPROC)

REPO_ROOT = Path(__file__).parent
REMOTE_REPO = "/root/modded-nanogpt"
VOL_PATH = "/vol"

app = modal.App("modded-nanogpt")
volume = modal.Volume.from_name("modded-nanogpt-data", create_if_missing=True)


# triton_kernels.py compiles a custom CE CUDA kernel at import via torch.cuda._compile_kernel,
# which needs nvcc + CUDA headers at /usr/local/cuda. debian_slim + pip-torch has neither, so
# we base on NVIDIA's CUDA devel image (12.8 matches torch 2.10's bundled NVRTC).
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "build-essential")
    .pip_install_from_requirements(str(REPO_ROOT / "requirements.txt"))
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "DATA_PATH": VOL_PATH,
            "HF_HOME": f"{VOL_PATH}/hf",
            "TORCHINDUCTOR_CACHE_DIR": f"{VOL_PATH}/cache/inductor",
            "TRITON_CACHE_DIR": f"{VOL_PATH}/cache/triton",
            "CUDA_HOME": "/usr/local/cuda",
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
    smear_k: int = 0,
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
    if smear_k:
        env["NANOGPT_SMEAR_K"] = str(smear_k)
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
    smear_k: int = 0,
) -> None:
    print(download_data.remote(chunks))
    if download_only:
        return
    timeout_s = (
        int(os.environ["NANOGPT_TIMEOUT"])
        if os.environ.get("NANOGPT_TIMEOUT")
        else _timeout_seconds(stop_frac, NPROC)
    )
    print(f"kill-switch timeout: {timeout_s // 60} min (stop_frac={stop_frac}, nproc={NPROC})")
    print(
        train.with_options(timeout=timeout_s).remote(
            script=script,
            nproc=NPROC,
            stop_frac=stop_frac,
            val_every=val_every,
            run_evals=run_evals,
            run_id=run_id,
            smear_k=smear_k,
        )
    )
    print("fetch logs with: modal volume get modded-nanogpt-data logs/<run_id>.txt")
