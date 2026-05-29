#!/bin/bash -l
# Probe max parallel Warp envs: sweep upward from SEARCH_START until OOM.
# Optional CALIBRATION_ENVS="1 2 4 ..." for a slow low-end VRAM fit.

set -euo pipefail

resolve_repo_root() {
    local dir="$1"
    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/pyproject.toml" && -f "${dir}/scripts/flygym/probe_odor_obstacle_navigation_gpu_env_capacity.py" ]]; then
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
    echo "Unable to locate flygym repo root." >&2
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

# Match launch_odor_obstacle_navigation_benchmark.sh task settings.
CCD_ITERATIONS="${CCD_ITERATIONS:-300}"
FEATURE_MODE="${FEATURE_MODE:-full}"
SEARCH_START="${SEARCH_START:-64}"
SEARCH_STEP="${SEARCH_STEP:-16}"
RESERVATION_FRACTION="${RESERVATION_FRACTION:-0.90}"
INCLUDE_PPO="${INCLUDE_PPO:-0}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/outputs/odor_obstacle_navigation/benchmarks/gpu_env_capacity/probe.json}"

args=(
    --ccd-iterations "${CCD_ITERATIONS}"
    --feature-mode "${FEATURE_MODE}"
    --search-start "${SEARCH_START}"
    --search-step "${SEARCH_STEP}"
    --reservation-fraction "${RESERVATION_FRACTION}"
    --output-json "${OUTPUT_JSON}"
)
if [[ -n "${CALIBRATION_ENVS:-}" ]]; then
    # shellcheck disable=SC2206
    read -r -a _calibration_envs <<< "${CALIBRATION_ENVS}"
    args+=(--calibration-envs "${_calibration_envs[@]}")
fi
if [[ "${INCLUDE_PPO}" == "1" ]]; then
    args+=(--include-ppo)
fi

uv run --extra warp python scripts/flygym/probe_odor_obstacle_navigation_gpu_env_capacity.py \
    "${args[@]}" \
    "$@"
