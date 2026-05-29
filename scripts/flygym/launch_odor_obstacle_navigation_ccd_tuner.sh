#!/bin/bash -l
#SBATCH --job-name=flygym-ccd-tuner
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/dkutsche/flygym/local_data/slurm/out/slurm-%x-%j.out
#SBATCH --error=/home/dkutsche/flygym/local_data/slurm/err/slurm-%x-%j.err
#
# Submit:
#   sbatch scripts/flygym/launch_odor_obstacle_navigation_ccd_tuner.sh
#
# Useful overrides:
#   CCD_ITERATIONS="400 600 800 1200 1600"
#   NUM_ENVS=842
#   REPEATS=1
#   ROLLOUT_STEPS=20
#   FEATURE_MODE=full              # full | no-vision | no-olfaction | no-sensors
#   AUTO_REDUCE_ENVS=1             # retry OOMs with fewer envs
#   ENV_REDUCTION_FACTOR=0.75
#   MIN_NUM_ENVS=1

set -euo pipefail

extra_args=("$@")
if [[ ${#extra_args[@]} -gt 0 && "${extra_args[0]}" == "--" ]]; then
    extra_args=("${extra_args[@]:1}")
fi

FLYGYM_ROOT="${FLYGYM_ROOT:-/home/dkutsche/flygym}"
if [[ ! -f "${FLYGYM_ROOT}/scripts/flygym/tune_warp_ccd_iterations.py" ]]; then
    echo "Could not find FlyGym CCD tuner under ${FLYGYM_ROOT}." >&2
    echo "Set FLYGYM_ROOT=/path/to/flygym and retry." >&2
    exit 1
fi

cd "${FLYGYM_ROOT}"
mkdir -p local_data/slurm/out local_data/slurm/err

# shellcheck source=/home/dkutsche/flygym/scripts/flygym/ensure_cuda_venv.sh
source "${FLYGYM_ROOT}/scripts/flygym/ensure_cuda_venv.sh"
ensure_cuda_venv "${FLYGYM_ROOT}"
verify_cuda_runtime "${FLYGYM_ROOT}"

export PYTHONPATH="${FLYGYM_ROOT}/src:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

CCD_ITERATIONS="${CCD_ITERATIONS:-400 600 800 1200 1600}"
NUM_ENVS="${NUM_ENVS:-842}"
REPEATS="${REPEATS:-1}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-20}"
DECISION_STEPS="${DECISION_STEPS:-500}"
TIMESTEP="${TIMESTEP:-0.0001}"
FEATURE_MODE="${FEATURE_MODE:-full}"
SEED="${SEED:-0}"
ACTION="${ACTION:-1.0 1.0}"
AUTO_REDUCE_ENVS="${AUTO_REDUCE_ENVS:-1}"
ENV_REDUCTION_FACTOR="${ENV_REDUCTION_FACTOR:-0.75}"
MIN_NUM_ENVS="${MIN_NUM_ENVS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/odor_obstacle_navigation/benchmarks/ccd_${NUM_ENVS}_${FEATURE_MODE}_${SLURM_JOB_ID:-local}}"

read -r -a CCD_ITERATION_ARGS <<< "${CCD_ITERATIONS}"
read -r -a ACTION_ARGS <<< "${ACTION}"

AUTO_REDUCE_ARGS=(--no-auto-reduce-envs)
if [[ "${AUTO_REDUCE_ENVS}" == "1" ]]; then
    AUTO_REDUCE_ARGS=(--auto-reduce-envs)
fi

uv run --extra warp python scripts/flygym/tune_warp_ccd_iterations.py \
    --num-envs "${NUM_ENVS}" \
    --ccd-iterations "${CCD_ITERATION_ARGS[@]}" \
    --repeats "${REPEATS}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --decision-steps "${DECISION_STEPS}" \
    --timestep "${TIMESTEP}" \
    --feature-mode "${FEATURE_MODE}" \
    --seed "${SEED}" \
    --action "${ACTION_ARGS[@]}" \
    --output-dir "${OUTPUT_DIR}" \
    "${AUTO_REDUCE_ARGS[@]}" \
    --env-reduction-factor "${ENV_REDUCTION_FACTOR}" \
    --min-num-envs "${MIN_NUM_ENVS}" \
    "${extra_args[@]}"
