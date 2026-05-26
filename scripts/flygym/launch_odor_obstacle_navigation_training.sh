#!/bin/bash -l
#SBATCH --job-name=flygym-v2-training
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=local_data/slurm/out/slurm-%x-%j.out
#SBATCH --error=local_data/slurm/err/slurm-%x-%j.err

set -euo pipefail

resolve_repo_root() {
    local dir="$1"
    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/pyproject.toml" && -f "${dir}/scripts/flygym/train_odor_obstacle_navigation.py" ]]; then
            echo "${dir}"
            return 0
        fi
        dir="$(dirname "${dir}")"
    done
    return 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! REPO_ROOT="$(resolve_repo_root "${SCRIPT_DIR}/../..")"; then
    for start_dir in "${SLURM_SUBMIT_DIR:-}" "${SCRIPT_DIR}"; do
        if [[ -n "${start_dir}" ]] && REPO_ROOT="$(resolve_repo_root "${start_dir}")"; then
            break
        fi
        REPO_ROOT=""
    done
fi
if [[ -z "${REPO_ROOT:-}" ]]; then
    echo "Unable to locate flygym repo root (submit from repo root: sbatch scripts/flygym/launch_odor_obstacle_navigation_training.sh)." >&2
    exit 1
fi
cd "${REPO_ROOT}"

# Ensure the locked CUDA 12.6 PyTorch build is installed before training (avoids PyPI cu13 wheels).
uv sync --frozen --extra warp

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

TIMESTEP=0.0001
DECISION_STEPS=500
EPISODE_TIME_SECONDS=5
N_ENVS=60
ROLLOUT_STEPS=200
BATCH_SIZE=1200
UPDATE_EPOCHS=10
TOTAL_TIMESTEPS=1000000
SUCCESS_BONUS=10.0
PROGRESS_REWARD_WEIGHT=1.0
TIME_PENALTY=0.01
ACTION_SPACE="discrete"
EVAL_INTERVAL=50000
EVAL_EPISODES=5
PROGRESS_INTERVAL_SECONDS=30
DEVICE="cuda"
ENV_DEVICE="cpu"
ENV_RUNNER="subproc"
MP_START_METHOD="spawn"

export WANDB_ENTITY="auto-fly"
export WANDB_PROJECT="autonomous-drosophila"
export WANDB_MODE="online"
WANDB_RUN_NAME="obs_progress_dt005_${ACTION_SPACE}_success_bonus_${SUCCESS_BONUS}_shape_${SLURM_JOB_ID:-local}"
RUN_DIR="${REPO_ROOT}/outputs/odor_obstacle_navigation/${WANDB_RUN_NAME}"

uv run python scripts/flygym/train_odor_obstacle_navigation.py \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --num-envs "${N_ENVS}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --minibatch-size "${BATCH_SIZE}" \
    --update-epochs "${UPDATE_EPOCHS}" \
    --learning-rate 3e-4 \
    --gamma 0.99 \
    --entropy-coef 0.0 \
    --timestep "${TIMESTEP}" \
    --decision-steps "${DECISION_STEPS}" \
    --episode-time "${EPISODE_TIME_SECONDS}" \
    --target-radius 2.0 \
    --success-bonus "${SUCCESS_BONUS}" \
    --progress-reward-weight "${PROGRESS_REWARD_WEIGHT}" \
    --time-penalty "${TIME_PENALTY}" \
    --action-space "${ACTION_SPACE}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --progress-interval-seconds "${PROGRESS_INTERVAL_SECONDS}" \
    --device "${DEVICE}" \
    --env-device "${ENV_DEVICE}" \
    --env-runner "${ENV_RUNNER}" \
    --mp-start-method "${MP_START_METHOD}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    --run-dir "${RUN_DIR}"
