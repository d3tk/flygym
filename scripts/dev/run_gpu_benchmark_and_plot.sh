#!/bin/bash -l
#SBATCH --job-name=flygym-v2-benchmark
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=local_data/slurm/out/slurm-%x-%j.out
#SBATCH --error=local_data/slurm/err/slurm-%x-%j.err
# Run GPU simulation benchmark and plot scaling results for this machine.
#
# Usage (from flygym repo root or any directory):
#   bash scripts/dev/run_gpu_benchmark_and_plot.sh
#
# Optional environment variables:
#   DEVICE_SUFFIX   Subdirectory tag under gpu_benchmark_results_all/ (default: local)
#   BENCHMARK_BASE_DIR  Parent directory for staged results (default: gpu_benchmark_results_all)

set -euo pipefail

resolve_repo_root() {
    local dir="$1"
    while [[ "${dir}" != "/" ]]; do
        if [[ -f "${dir}/pyproject.toml" && -f "${dir}/scripts/dev/run_gpu_benchmark.py" ]]; then
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
    echo "Unable to locate flygym repo root (submit from repo root: sbatch scripts/dev/run_gpu_benchmark_and_plot.sh)." >&2
    exit 1
fi
cd "${REPO_ROOT}"

# shellcheck source=../flygym/ensure_cuda_venv.sh
source "${REPO_ROOT}/scripts/flygym/ensure_cuda_venv.sh"
ensure_cuda_venv "${REPO_ROOT}"
verify_cuda_runtime "${REPO_ROOT}"

: "${DEVICE_SUFFIX:=local}"
: "${BENCHMARK_BASE_DIR:=gpu_benchmark_results_all}"

echo "=== Running GPU simulation benchmark ===" >&2
uv run --extra warp python "${REPO_ROOT}/scripts/dev/run_gpu_benchmark.py"

STAGE_DIR="${BENCHMARK_BASE_DIR}/gpu_benchmark_results_${DEVICE_SUFFIX}"
mkdir -p "${STAGE_DIR}"
cp -f "${REPO_ROOT}/gpu_benchmark_results/gpu_simulation_benchmark_results.csv" "${STAGE_DIR}/"
cp -f "${REPO_ROOT}/gpu_benchmark_results/gpu_info.csv" "${STAGE_DIR}/"

echo "=== Staged results in ${STAGE_DIR} ===" >&2
echo "=== Plotting (${DEVICE_SUFFIX}) ===" >&2
uv run --extra warp python "${REPO_ROOT}/scripts/dev/plot_benchmark_results.py" \
  --basedir "${BENCHMARK_BASE_DIR}" \
  --devices "${DEVICE_SUFFIX}"

echo "Done. Plot: ${BENCHMARK_BASE_DIR}/scaling_test_simplifygeomfalse.png" >&2
