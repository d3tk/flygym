#!/bin/bash -l
#SBATCH --job-name=flygym-l40-env-sweep
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=local_data/slurm/out/slurm-%x-%j.out
#SBATCH --error=local_data/slurm/err/slurm-%x-%j.err
#
# Probe max Warp envs on the allocated GPU (e.g. L40S), then benchmark env throughput:
#   - CPU subproc envs: CPU_ENV_START..CPU_ENV_END (default 32..256, step 16)
#   - GPU warp envs: (max//2)..max (default step 16)
#
# Submit from repo root:
#   sbatch scripts/flygym/launch_odor_obstacle_navigation_l40_env_sweep.sh

set -euo pipefail

# sbatch forwards script args in "$@"; ignore an optional leading "--" separator.
extra_benchmark_args=("$@")
if [[ ${#extra_benchmark_args[@]} -gt 0 && "${extra_benchmark_args[0]}" == "--" ]]; then
    extra_benchmark_args=("${extra_benchmark_args[@]:1}")
fi

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

# Space-separated integers from start to end (inclusive), stepping by step; always includes end.
build_linear_sweep() {
    local start="$1" end="$2" step="$3"
    if (( start > end )); then
        echo "${end}"
        return 0
    fi
    local -a values=()
    local n="${start}"
    while (( n < end )); do
        values+=("${n}")
        n=$((n + step))
    done
    values+=("${end}")
    printf '%s\n' "${values[@]}" | sort -nu | tr '\n' ' '
}

read_max_verified_gpu_envs() {
    local probe_json="$1"
    uv run --extra warp python - <<'PY' "${probe_json}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
max_verified = int(data.get("max_verified_num_envs", 0))
if max_verified <= 0:
    raise SystemExit(
        f"Invalid max_verified_num_envs={max_verified!r} in {path}. "
        "GPU capacity probe did not find a working env count."
    )
print(max_verified)
PY
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
    echo "Unable to locate flygym repo root (submit from repo root: sbatch scripts/flygym/launch_odor_obstacle_navigation_l40_env_sweep.sh)." >&2
    exit 1
fi
cd "${REPO_ROOT}"

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

JOB_TAG="${SLURM_JOB_ID:-local}"
BENCHMARK_DIR="${REPO_ROOT}/outputs/odor_obstacle_navigation/benchmarks"
PROBE_DIR="${BENCHMARK_DIR}/gpu_env_capacity"
mkdir -p "${PROBE_DIR}" "${BENCHMARK_DIR}"

# --- Phase 1: probe max GPU envs on this node ---------------------------------
CCD_ITERATIONS="${CCD_ITERATIONS:-300}"
FEATURE_MODE="${FEATURE_MODE:-full}"
PROBE_SEARCH_START="${PROBE_SEARCH_START:-64}"
PROBE_SEARCH_STEP="${PROBE_SEARCH_STEP:-16}"
PROBE_RESERVATION_FRACTION="${PROBE_RESERVATION_FRACTION:-0.90}"
PROBE_OUTPUT_JSON="${PROBE_OUTPUT_JSON:-${PROBE_DIR}/probe_${JOB_TAG}.json}"

echo "=== Phase 1: GPU env capacity probe ===" >&2
probe_args=(
    --ccd-iterations "${CCD_ITERATIONS}"
    --feature-mode "${FEATURE_MODE}"
    --search-start "${PROBE_SEARCH_START}"
    --search-step "${PROBE_SEARCH_STEP}"
    --reservation-fraction "${PROBE_RESERVATION_FRACTION}"
    --output-json "${PROBE_OUTPUT_JSON}"
)
if [[ "${PROBE_INCLUDE_PPO:-0}" == "1" ]]; then
    probe_args+=(--include-ppo)
fi
if [[ -n "${PROBE_CALIBRATION_ENVS:-}" ]]; then
    # shellcheck disable=SC2206
    read -r -a _probe_calibration_envs <<< "${PROBE_CALIBRATION_ENVS}"
    probe_args+=(--calibration-envs "${_probe_calibration_envs[@]}")
fi

uv run --extra warp python scripts/flygym/probe_odor_obstacle_navigation_gpu_env_capacity.py \
    "${probe_args[@]}"

MAX_GPU_ENVS="$(read_max_verified_gpu_envs "${PROBE_OUTPUT_JSON}")"
GPU_ENV_START=$((MAX_GPU_ENVS / 2))
GPU_ENV_END="${MAX_GPU_ENVS}"
echo "Max verified GPU envs: ${MAX_GPU_ENVS} (sweep ${GPU_ENV_START}..${GPU_ENV_END})" >&2

# --- Phase 2: training env throughput sweeps ----------------------------------
TIMESTEP="${TIMESTEP:-0.0001}"
DECISION_STEPS="${DECISION_STEPS:-500}"
EPISODE_TIME_SECONDS="${EPISODE_TIME_SECONDS:-5}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-842}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-10}"
ROLLOUTS="${ROLLOUTS:-1}"
REPEATS="${REPEATS:-1}"
SUCCESS_BONUS="${SUCCESS_BONUS:-10.0}"
PROGRESS_REWARD_WEIGHT="${PROGRESS_REWARD_WEIGHT:-1.0}"
TIME_PENALTY="${TIME_PENALTY:-0.01}"
MP_START_METHOD="${MP_START_METHOD:-spawn}"
FEATURE_MODES="${FEATURE_MODES:-full}"
BENCHMARK_LAYERS="${BENCHMARK_LAYERS:-env}"

CPU_ENV_START="${CPU_ENV_START:-32}"
CPU_ENV_END="${CPU_ENV_END:-256}"
CPU_ENV_STEP="${CPU_ENV_STEP:-16}"
GPU_ENV_STEP="${GPU_ENV_STEP:-16}"

CPU_NUM_ENVS="$(build_linear_sweep "${CPU_ENV_START}" "${CPU_ENV_END}" "${CPU_ENV_STEP}")"
GPU_NUM_ENVS="$(build_linear_sweep "${GPU_ENV_START}" "${GPU_ENV_END}" "${GPU_ENV_STEP}")"

read -r -a CPU_NUM_ENVS_ARGS <<< "${CPU_NUM_ENVS}"
read -r -a GPU_NUM_ENVS_ARGS <<< "${GPU_NUM_ENVS}"
read -r -a FEATURE_MODE_ARGS <<< "${FEATURE_MODES}"
read -r -a BENCHMARK_LAYER_ARGS <<< "${BENCHMARK_LAYERS}"
read -r -a CCD_ITERATION_ARGS <<< "${CCD_ITERATIONS}"

CPU_OUTPUT_CSV="${BENCHMARK_DIR}/l40_cpu_env_sweep_${JOB_TAG}.csv"
GPU_OUTPUT_CSV="${BENCHMARK_DIR}/l40_gpu_env_sweep_${JOB_TAG}.csv"

common_benchmark_args=(
    --ccd-iterations "${CCD_ITERATION_ARGS[@]}"
    --feature-modes "${FEATURE_MODE_ARGS[@]}"
    --benchmark-layers "${BENCHMARK_LAYER_ARGS[@]}"
    --mp-start-method "${MP_START_METHOD}"
    --repeats "${REPEATS}"
    --rollouts "${ROLLOUTS}"
    --rollout-steps "${ROLLOUT_STEPS}"
    --minibatch-size "${BATCH_SIZE}"
    --update-epochs "${UPDATE_EPOCHS}"
    --timestep "${TIMESTEP}"
    --decision-steps "${DECISION_STEPS}"
    --episode-time "${EPISODE_TIME_SECONDS}"
    --target-radius 2.0
    --success-bonus "${SUCCESS_BONUS}"
    --progress-reward-weight "${PROGRESS_REWARD_WEIGHT}"
    --time-penalty "${TIME_PENALTY}"
)

echo "=== Phase 2a: CPU subproc env sweep (${CPU_NUM_ENVS}) ===" >&2
uv run --extra warp python scripts/flygym/benchmark_odor_obstacle_navigation_training.py \
    --runners subproc \
    --ppo-devices cpu \
    --env-devices cpu \
    --num-envs "${CPU_NUM_ENVS_ARGS[@]}" \
    --output-csv "${CPU_OUTPUT_CSV}" \
    "${common_benchmark_args[@]}" \
    "${extra_benchmark_args[@]+"${extra_benchmark_args[@]}"}"

echo "=== Phase 2b: GPU warp env sweep (${GPU_NUM_ENVS}) ===" >&2
uv run --extra warp python scripts/flygym/benchmark_odor_obstacle_navigation_training.py \
    --runners warp \
    --ppo-devices cpu \
    --env-devices cuda \
    --num-envs "${GPU_NUM_ENVS_ARGS[@]}" \
    --output-csv "${GPU_OUTPUT_CSV}" \
    "${common_benchmark_args[@]}" \
    "${extra_benchmark_args[@]+"${extra_benchmark_args[@]}"}"

echo "Done. Probe: ${PROBE_OUTPUT_JSON}" >&2
echo "CPU sweep CSV: ${CPU_OUTPUT_CSV}" >&2
echo "GPU sweep CSV: ${GPU_OUTPUT_CSV}" >&2
