#!/bin/bash -l
#SBATCH --job-name=flygym-v2-training
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=local_data/slurm/out/slurm-%x-%j.out
#SBATCH --error=local_data/slurm/err/slurm-%x-%j.err
#
# Submit from repo root:
#   sbatch scripts/flygym/launch_odor_obstacle_navigation_training.sh
#
# Defaults: 842 MuJoCo-Warp envs on CUDA (matches L40 env sweep at n=842).
#
# Environment overrides:
#   N_ENVS=826                    # parallel warp envs (default: 842)
#   BATCH_SIZE=842                # PPO minibatch; must divide rollout_steps * n_envs
#   DEVICE=cpu                    # PPO policy device (ENV_DEVICE=cuda for warp sim)
#   TOTAL_TIMESTEPS=10000000      # SB3 training budget
#   EVAL_FREQ=50000               # eval every N env steps
#   CHECKPOINT_FREQ=100000        # checkpoint every N env steps
#   N_EVAL_ENVS=4                 # small CUDA warp eval pool
#   N_EVAL_EPISODES=13            # episodes per eval env (4×13 ≈ 52 total)
#   ACTION_SPACE=discrete         # discrete | continuous
#   RANDOMIZE_LAYOUT=0            # disable v1-style randomized layout preset
#   SPAWN_XY_JITTER=0.0           # optional; defaults to 0.0 with randomized v1 layouts
#   SPAWN_YAW_JITTER=0.0          # optional; defaults to 0.0 with randomized v1 layouts
#   NO_WANDB=1                    # disable W&B (debug)
#   RESUME=1                      # continue an existing run (see below)
#
# Resume (set RESUME=1 and the original run identifiers):
#   RESUME=1 \
#   WANDB_RUN_NAME="obs_progress_dt005_discrete_warp_842envs_8201234" \
#   WANDB_ID="abc123xyz" \
#   RESUME_CHECKPOINT=".outputs/flygym/sb3/${WANDB_RUN_NAME}/checkpoints/navigation_policy_500000_steps.zip" \
#   sbatch scripts/flygym/launch_odor_obstacle_navigation_training.sh
#
# Local smoke run (4 envs, 5k steps, warp cuda):
#   uv run --extra warp python scripts/flygym/train_odor_obstacle_navigation.py \
#     --total-timesteps 5000 --num-envs 4 --env-device cuda --env-runner warp \
#     --device cuda --no-wandb --run-dir .outputs/flygym/sb3/smoke_test

set -euo pipefail

extra_train_args=("$@")
if [[ ${#extra_train_args[@]} -gt 0 && "${extra_train_args[0]}" == "--" ]]; then
    extra_train_args=("${extra_train_args[@]:1}")
fi

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

mkdir -p local_data/slurm/out local_data/slurm/err

# Ensure the locked CUDA 12.6 PyTorch build is installed before training (avoids PyPI cu13 wheels).
# shellcheck source=ensure_cuda_venv.sh
source "${REPO_ROOT}/scripts/flygym/ensure_cuda_venv.sh"
ensure_cuda_venv "${REPO_ROOT}"
verify_cuda_runtime "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

TIMESTEP="${TIMESTEP:-0.0001}"
DECISION_STEPS="${DECISION_STEPS:-500}"
CCD_ITERATIONS="${CCD_ITERATIONS:-300}"
EPISODE_TIME_SECONDS="${EPISODE_TIME_SECONDS:-5}"
N_ENVS="${N_ENVS:-512}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-64}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-10}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000000}"
SUCCESS_BONUS="${SUCCESS_BONUS:-10.0}"
PROGRESS_REWARD_WEIGHT="${PROGRESS_REWARD_WEIGHT:-1.0}"
TIME_PENALTY="${TIME_PENALTY:-0.01}"
ACTION_SPACE="${ACTION_SPACE:-discrete}"
EVAL_FREQ="${EVAL_FREQ:-50000}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-100000}"
N_EVAL_ENVS="${N_EVAL_ENVS:-5}"
N_EVAL_EPISODES="${N_EVAL_EPISODES:-10}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cpu}"
ENV_DEVICE="${ENV_DEVICE:-cuda}"
ENV_RUNNER="${ENV_RUNNER:-warp}"
MP_START_METHOD="${MP_START_METHOD:-spawn}"
RANDOMIZE_LAYOUT="${RANDOMIZE_LAYOUT:-1}"
NO_WANDB="${NO_WANDB:-0}"

export WANDB_ENTITY="${WANDB_ENTITY:-auto-fly}"
export WANDB_PROJECT="${WANDB_PROJECT:-autonomous-drosophila}"
export WANDB_MODE="${WANDB_MODE:-online}"

RESUME="${RESUME:-0}"
if [[ "${RESUME}" == "1" ]]; then
    if [[ -z "${WANDB_RUN_NAME:-}" ]]; then
        echo "RESUME=1 requires WANDB_RUN_NAME (the original W&B run name)." >&2
        exit 1
    fi
    RUN_DIR="${RUN_DIR:-${REPO_ROOT}/.outputs/flygym/sb3/${WANDB_RUN_NAME}}"
    if [[ -z "${RESUME_CHECKPOINT:-}" && -d "${RUN_DIR}/checkpoints" ]]; then
        RESUME_CHECKPOINT="$(
            ls -1 "${RUN_DIR}"/checkpoints/navigation_policy_*_steps.zip 2>/dev/null \
                | sort -V \
                | tail -1 \
                || true
        )"
    fi
    if [[ -z "${RESUME_CHECKPOINT:-}" ]]; then
        echo "RESUME=1 requires RESUME_CHECKPOINT or an existing ${RUN_DIR}/checkpoints/navigation_policy_*_steps.zip." >&2
        exit 1
    fi
else
    WANDB_RUN_NAME="${WANDB_RUN_NAME:-obs_progress_dt005_${ACTION_SPACE}_warp_${N_ENVS}envs_${SLURM_JOB_ID:-local}}"
    RUN_DIR="${RUN_DIR:-${REPO_ROOT}/.outputs/flygym/sb3/${WANDB_RUN_NAME}}"
    WANDB_ID="${WANDB_ID:-}"
    RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    RESUME_ARGS+=(--resume-from "${RESUME_CHECKPOINT}")
fi
if [[ -n "${WANDB_ID:-}" ]]; then
    RESUME_ARGS+=(--wandb-id "${WANDB_ID}")
fi

WANDB_ARGS=()
if [[ "${NO_WANDB}" == "1" ]]; then
    WANDB_ARGS+=(--no-wandb)
else
    WANDB_ARGS+=(--use-wandb)
fi
if [[ -n "${WANDB_GROUP:-}" ]]; then
    WANDB_ARGS+=(--wandb-group "${WANDB_GROUP}")
fi
if [[ -n "${WANDB_TAGS:-}" ]]; then
    # shellcheck disable=SC2206
    WANDB_TAG_LIST=(${WANDB_TAGS})
    WANDB_ARGS+=(--wandb-tags "${WANDB_TAG_LIST[@]}")
fi

LAYOUT_ARGS=()
if [[ "${RANDOMIZE_LAYOUT}" == "1" ]]; then
    LAYOUT_PRESET="${LAYOUT_PRESET:-v1}"
    SPAWN_XY_JITTER="${SPAWN_XY_JITTER:-0.0}"
    SPAWN_YAW_JITTER="${SPAWN_YAW_JITTER:-0.0}"
    LAYOUT_ARGS+=(
        --randomize-layout
        --layout-preset "${LAYOUT_PRESET}"
        --spawn-xy-jitter "${SPAWN_XY_JITTER}"
        --spawn-yaw-jitter "${SPAWN_YAW_JITTER}"
    )
else
    LAYOUT_PRESET="${LAYOUT_PRESET:-default}"
    LAYOUT_ARGS+=(--layout-preset "${LAYOUT_PRESET}")
    if [[ -n "${SPAWN_XY_JITTER:-}" ]]; then
        LAYOUT_ARGS+=(--spawn-xy-jitter "${SPAWN_XY_JITTER}")
    fi
    if [[ -n "${SPAWN_YAW_JITTER:-}" ]]; then
        LAYOUT_ARGS+=(--spawn-yaw-jitter "${SPAWN_YAW_JITTER}")
    fi
fi
if [[ -n "${OBSTACLE_COUNT_RANGE:-}" ]]; then
    # shellcheck disable=SC2206
    OBSTACLE_COUNT_RANGE_VALUES=(${OBSTACLE_COUNT_RANGE})
    if [[ ${#OBSTACLE_COUNT_RANGE_VALUES[@]} -ne 2 ]]; then
        echo "OBSTACLE_COUNT_RANGE must contain exactly two integers: MIN MAX." >&2
        exit 1
    fi
    LAYOUT_ARGS+=(
        --obstacle-count-range
        "${OBSTACLE_COUNT_RANGE_VALUES[0]}"
        "${OBSTACLE_COUNT_RANGE_VALUES[1]}"
    )
fi

echo "[launch] run_dir=${RUN_DIR} num_envs=${N_ENVS} env_runner=${ENV_RUNNER} layout_preset=${LAYOUT_PRESET} randomize_layout=${RANDOMIZE_LAYOUT} resume=${RESUME}" >&2
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
    echo "[launch] resume_checkpoint=${RESUME_CHECKPOINT}" >&2
fi

uv run --extra warp python scripts/flygym/train_odor_obstacle_navigation.py \
    --total-timesteps "${TOTAL_TIMESTEPS}" \
    --num-envs "${N_ENVS}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --minibatch-size "${BATCH_SIZE}" \
    --update-epochs "${UPDATE_EPOCHS}" \
    --learning-rate 3e-4 \
    --gamma 0.99 \
    --entropy-coef 0.0 \
    --timestep "${TIMESTEP}" \
    --ccd-iterations "${CCD_ITERATIONS}" \
    --decision-steps "${DECISION_STEPS}" \
    --episode-time "${EPISODE_TIME_SECONDS}" \
    --target-radius 2.0 \
    --success-bonus "${SUCCESS_BONUS}" \
    --progress-reward-weight "${PROGRESS_REWARD_WEIGHT}" \
    --time-penalty "${TIME_PENALTY}" \
    --action-space "${ACTION_SPACE}" \
    --eval-freq "${EVAL_FREQ}" \
    --checkpoint-freq "${CHECKPOINT_FREQ}" \
    --n-eval-envs "${N_EVAL_ENVS}" \
    --n-eval-episodes "${N_EVAL_EPISODES}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --env-device "${ENV_DEVICE}" \
    --env-runner "${ENV_RUNNER}" \
    --mp-start-method "${MP_START_METHOD}" \
    "${LAYOUT_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    --wandb-entity "${WANDB_ENTITY}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-mode "${WANDB_MODE}" \
    --wandb-run-name "${WANDB_RUN_NAME}" \
    --run-dir "${RUN_DIR}" \
    "${RESUME_ARGS[@]}" \
    "${extra_train_args[@]}"
