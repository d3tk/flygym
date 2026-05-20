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

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

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
N_ENVS=8
ROLLOUT_STEPS=200
BATCH_SIZE=1200
UPDATE_EPOCHS=10
TOTAL_TIMESTEPS=1000000
SUCCESS_BONUS=10.0
PROGRESS_REWARD_WEIGHT=10.0
EVAL_INTERVAL=50000
EVAL_EPISODES=5
PROGRESS_INTERVAL_SECONDS=30
DEVICE="cuda"
ENV_RUNNER="subproc"
MP_START_METHOD="spawn"

export WANDB_ENTITY="auto-fly"
export WANDB_PROJECT="autonomous-drosophila"
export WANDB_MODE="online"
WANDB_RUN_NAME="obs_progress_dt005_continuous_success_bonus_${SUCCESS_BONUS}_${SLURM_JOB_ID:-local}"
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
    --eval-interval "${EVAL_INTERVAL}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --progress-interval-seconds "${PROGRESS_INTERVAL_SECONDS}" \
    --device "${DEVICE}" \
    --env-runner "${ENV_RUNNER}" \
    --mp-start-method "${MP_START_METHOD}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    --run-dir "${RUN_DIR}"
