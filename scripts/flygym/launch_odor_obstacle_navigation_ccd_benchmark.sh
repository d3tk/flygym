#!/bin/bash -l
#SBATCH --job-name=flygym-v2-ccd-benchmark
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
        if [[ -f "${dir}/pyproject.toml" && -f "${dir}/scripts/flygym/benchmark_odor_obstacle_navigation_training.py" ]]; then
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
    echo "Unable to locate flygym repo root (submit from repo root: sbatch scripts/flygym/launch_odor_obstacle_navigation_ccd_benchmark.sh)." >&2
    exit 1
fi
cd "${REPO_ROOT}"

# shellcheck source=ensure_cuda_venv.sh
source "${SCRIPT_DIR}/ensure_cuda_venv.sh"
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

TIMESTEP="0.0001"
DECISION_STEPS="500"
EPISODE_TIME_SECONDS="5"
ROLLOUT_STEPS="200"
BATCH_SIZE="1200"
UPDATE_EPOCHS="10"
ROLLOUTS="1"
REPEATS="5"
SUCCESS_BONUS="10.0"
PROGRESS_REWARD_WEIGHT="10.0"
MP_START_METHOD="spawn"
NUM_ENVS="64"
PPO_DEVICES="cpu cuda"
ENV_DEVICES="cuda"
FEATURE_MODES="full"
BENCHMARK_LAYERS="env ppo"
CCD_ITERATIONS="200 250 300 350 400"
BENCHMARK_DIR="${REPO_ROOT}/outputs/odor_obstacle_navigation/benchmarks"
OUTPUT_CSV="${BENCHMARK_DIR}/ccd_training_speed_${SLURM_JOB_ID:-local}.csv"

read -r -a NUM_ENVS_ARGS <<< "${NUM_ENVS}"
read -r -a PPO_DEVICES_ARGS <<< "${PPO_DEVICES}"
read -r -a ENV_DEVICES_ARGS <<< "${ENV_DEVICES}"
read -r -a FEATURE_MODE_ARGS <<< "${FEATURE_MODES}"
read -r -a BENCHMARK_LAYER_ARGS <<< "${BENCHMARK_LAYERS}"
read -r -a CCD_ITERATION_ARGS <<< "${CCD_ITERATIONS}"

mkdir -p "${BENCHMARK_DIR}"

uv run --extra warp python scripts/flygym/benchmark_odor_obstacle_navigation_training.py \
    --runners subproc \
    --ppo-devices "${PPO_DEVICES_ARGS[@]}" \
    --env-devices "${ENV_DEVICES_ARGS[@]}" \
    --num-envs "${NUM_ENVS_ARGS[@]}" \
    --feature-modes "${FEATURE_MODE_ARGS[@]}" \
    --benchmark-layers "${BENCHMARK_LAYER_ARGS[@]}" \
    --ccd-iterations "${CCD_ITERATION_ARGS[@]}" \
    --mp-start-method "${MP_START_METHOD}" \
    --repeats "${REPEATS}" \
    --rollouts "${ROLLOUTS}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --minibatch-size "${BATCH_SIZE}" \
    --update-epochs "${UPDATE_EPOCHS}" \
    --timestep "${TIMESTEP}" \
    --decision-steps "${DECISION_STEPS}" \
    --episode-time "${EPISODE_TIME_SECONDS}" \
    --target-radius 2.0 \
    --success-bonus "${SUCCESS_BONUS}" \
    --progress-reward-weight "${PROGRESS_REWARD_WEIGHT}" \
    --output-csv "${OUTPUT_CSV}" \
    "$@"
