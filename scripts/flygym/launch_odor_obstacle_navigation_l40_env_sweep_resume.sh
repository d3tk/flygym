#!/bin/bash -l
#SBATCH --job-name=flygym-l40-env-sweep-resume
#SBATCH --partition=general
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --output=local_data/slurm/out/slurm-%x-%j.out
#SBATCH --error=local_data/slurm/err/slurm-%x-%j.err
#
# Resume an interrupted l40 env sweep (skips GPU capacity probe).
#
# Defaults target job 8198431 (probe + CPU sweep through 160 completed).
#
# Submit from repo root:
#   sbatch scripts/flygym/launch_odor_obstacle_navigation_l40_env_sweep_resume.sh
#
# Environment overrides:
#   PRIOR_JOB_ID=8198431          # prior slurm job tag for default paths
#   PROBE_JSON=...                # probe result (default: probe_${PRIOR_JOB_ID}.json)
#   PRIOR_CPU_CSV=...             # completed CPU rows (skip these num_envs)
#   CPU_ENV_START=176             # first CPU num_envs to try (default: 176)
#   CPU_ENV_END=256
#   SKIP_CPU_SWEEP=1              # only run GPU warp sweep (recommended after 8200794)
#   SKIP_GPU_SWEEP=1              # only run remaining CPU sweep
#   RUN_GPU_FIRST=1               # run warp sweep before CPU (default: 1)
#   REPROBE_GPU=1                 # probe max envs on this GPU before sweep (use on A6000 etc.)
#   MERGE_CPU_CSVS=1              # write merged l40_cpu_env_sweep_${PRIOR_JOB_ID}_merged.csv
#
# GPU-only (after CPU resume finished):
#   SKIP_CPU_SWEEP=1 sbatch scripts/flygym/launch_odor_obstacle_navigation_l40_env_sweep_resume.sh
#
# Extra benchmark flags (e.g. --continue-after-worker-failure):
#   sbatch ... --continue-after-worker-failure

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

build_linear_sweep() {
    local start="$1" end="$2" step="$3"
    if (( start > end )); then
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

filter_num_envs_not_in_csv() {
    local csv_path="$1"
    local runner_filter="$2"
    shift 2
    local -a requested=("$@")
    if [[ ! -f "${csv_path}" ]]; then
        printf '%s\n' "${requested[@]}" | tr '\n' ' '
        return 0
    fi
    uv run --extra warp python - <<'PY' "${csv_path}" "${runner_filter}" "${requested[@]}"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
runner_filter = sys.argv[2]
requested = [int(value) for value in sys.argv[3:]]
done: set[int] = set()
with csv_path.open(newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("benchmark_layer") != "env":
            continue
        if row.get("runner") != runner_filter:
            continue
        done.add(int(row["num_envs"]))
remaining = [value for value in requested if value not in done]
print(" ".join(str(value) for value in remaining))
PY
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
        "Point PROBE_JSON at a completed probe output."
    )
print(max_verified)
PY
}

merge_cpu_sweep_csvs() {
    local prior_csv="$1" resume_csv="$2" merged_csv="$3"
    uv run --extra warp python - <<'PY' "${prior_csv}" "${resume_csv}" "${merged_csv}"
import csv
import sys
from pathlib import Path

prior_path = Path(sys.argv[1])
resume_path = Path(sys.argv[2])
merged_path = Path(sys.argv[3])

rows: list[dict[str, str]] = []
seen: set[tuple[str, ...]] = set()
fieldnames: list[str] | None = None

for path in (prior_path, resume_path):
    if not path.is_file():
        continue
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if fieldnames is None:
            fieldnames = list(reader.fieldnames or [])
        for row in reader:
            key = (
                row.get("benchmark_layer", ""),
                row.get("runner", ""),
                row.get("num_envs", ""),
                row.get("repeat", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

if not fieldnames or not rows:
    raise SystemExit("No rows to merge; check PRIOR_CPU_CSV and resume CSV paths.")

rows.sort(key=lambda row: int(row["num_envs"]))
merged_path.parent.mkdir(parents=True, exist_ok=True)
with merged_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {len(rows)} rows to {merged_path}")
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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

PRIOR_JOB_ID="${PRIOR_JOB_ID:-8198431}"
JOB_TAG="${SLURM_JOB_ID:-resume}"
BENCHMARK_DIR="${REPO_ROOT}/outputs/odor_obstacle_navigation/benchmarks"
PROBE_DIR="${BENCHMARK_DIR}/gpu_env_capacity"
mkdir -p "${PROBE_DIR}" "${BENCHMARK_DIR}"

PROBE_JSON="${PROBE_JSON:-${PROBE_DIR}/probe_${PRIOR_JOB_ID}.json}"
PRIOR_CPU_CSV="${PRIOR_CPU_CSV:-${BENCHMARK_DIR}/l40_cpu_env_sweep_${PRIOR_JOB_ID}.csv}"
PRIOR_GPU_CSV="${PRIOR_GPU_CSV:-}"

CCD_ITERATIONS="${CCD_ITERATIONS:-300}"
FEATURE_MODE="${FEATURE_MODE:-full}"
PROBE_SEARCH_START="${PROBE_SEARCH_START:-64}"
PROBE_SEARCH_STEP="${PROBE_SEARCH_STEP:-16}"
PROBE_RESERVATION_FRACTION="${PROBE_RESERVATION_FRACTION:-0.90}"
REPROBE_GPU="${REPROBE_GPU:-0}"

if [[ "${REPROBE_GPU}" == "1" ]]; then
    PROBE_JSON="${PROBE_DIR}/probe_${JOB_TAG}.json"
    echo "=== Re-probe GPU env capacity on this allocation ===" >&2
    probe_args=(
        --ccd-iterations "${CCD_ITERATIONS}"
        --feature-mode "${FEATURE_MODE}"
        --search-start "${PROBE_SEARCH_START}"
        --search-step "${PROBE_SEARCH_STEP}"
        --reservation-fraction "${PROBE_RESERVATION_FRACTION}"
        --output-json "${PROBE_JSON}"
    )
    uv run --extra warp python scripts/flygym/probe_odor_obstacle_navigation_gpu_env_capacity.py \
        "${probe_args[@]}"
elif [[ ! -f "${PROBE_JSON}" ]]; then
    echo "Probe JSON not found: ${PROBE_JSON}" >&2
    echo "Set PROBE_JSON, PRIOR_JOB_ID, or REPROBE_GPU=1." >&2
    exit 1
fi

MAX_GPU_ENVS="$(read_max_verified_gpu_envs "${PROBE_JSON}")"
GPU_ENV_START="${GPU_ENV_START:-$((MAX_GPU_ENVS / 2))}"
GPU_ENV_END="${GPU_ENV_END:-${MAX_GPU_ENVS}}"
GPU_ENV_STEP="${GPU_ENV_STEP:-16}"

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

CPU_ENV_START="${CPU_ENV_START:-176}"
CPU_ENV_END="${CPU_ENV_END:-256}"
CPU_ENV_STEP="${CPU_ENV_STEP:-16}"

SKIP_CPU_SWEEP="${SKIP_CPU_SWEEP:-0}"
SKIP_GPU_SWEEP="${SKIP_GPU_SWEEP:-0}"
RUN_GPU_FIRST="${RUN_GPU_FIRST:-1}"
MERGE_CPU_CSVS="${MERGE_CPU_CSVS:-1}"

read -r -a FEATURE_MODE_ARGS <<< "${FEATURE_MODES}"
read -r -a BENCHMARK_LAYER_ARGS <<< "${BENCHMARK_LAYERS}"
read -r -a CCD_ITERATION_ARGS <<< "${CCD_ITERATIONS}"

CPU_OUTPUT_CSV="${BENCHMARK_DIR}/l40_cpu_env_sweep_${JOB_TAG}.csv"
GPU_OUTPUT_CSV="${BENCHMARK_DIR}/l40_gpu_env_sweep_${JOB_TAG}.csv"
MERGED_CPU_CSV="${BENCHMARK_DIR}/l40_cpu_env_sweep_${PRIOR_JOB_ID}_merged.csv"

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

run_cpu_sweep() {
    if [[ "${SKIP_CPU_SWEEP}" == "1" ]]; then
        echo "SKIP_CPU_SWEEP=1; skipping CPU sweep." >&2
        return 0
    fi
    CPU_NUM_ENVS="$(build_linear_sweep "${CPU_ENV_START}" "${CPU_ENV_END}" "${CPU_ENV_STEP}")"
    read -r -a CPU_NUM_ENVS_REQUESTED <<< "${CPU_NUM_ENVS}"
    if [[ ${#CPU_NUM_ENVS_REQUESTED[@]} -eq 0 ]]; then
        echo "No CPU num_envs in range ${CPU_ENV_START}..${CPU_ENV_END}; skip CPU sweep." >&2
        return 0
    fi
    CPU_NUM_ENVS_REMAINING="$(filter_num_envs_not_in_csv "${PRIOR_CPU_CSV}" subproc "${CPU_NUM_ENVS_REQUESTED[@]}")"
    read -r -a CPU_NUM_ENVS_ARGS <<< "${CPU_NUM_ENVS_REMAINING}"
    if [[ ${#CPU_NUM_ENVS_ARGS[@]} -eq 0 ]]; then
        echo "All requested CPU num_envs already in ${PRIOR_CPU_CSV}; skip CPU sweep." >&2
        return 0
    fi
    echo "=== CPU subproc env sweep (${CPU_NUM_ENVS_ARGS[*]}) ===" >&2
    uv run --extra warp python scripts/flygym/benchmark_odor_obstacle_navigation_training.py \
        --runners subproc \
        --ppo-devices cpu \
        --env-devices cpu \
        --num-envs "${CPU_NUM_ENVS_ARGS[@]}" \
        --output-csv "${CPU_OUTPUT_CSV}" \
        "${common_benchmark_args[@]}" \
        "${extra_benchmark_args[@]+"${extra_benchmark_args[@]}"}"
}

run_gpu_sweep() {
    if [[ "${SKIP_GPU_SWEEP}" == "1" ]]; then
        echo "SKIP_GPU_SWEEP=1; skipping GPU sweep." >&2
        return 0
    fi
    GPU_NUM_ENVS="$(build_linear_sweep "${GPU_ENV_START}" "${GPU_ENV_END}" "${GPU_ENV_STEP}")"
    read -r -a GPU_NUM_ENVS_REQUESTED <<< "${GPU_NUM_ENVS}"
    local gpu_filter_csv="${PRIOR_GPU_CSV}"
    if [[ -z "${gpu_filter_csv}" && -f "${GPU_OUTPUT_CSV}" ]]; then
        gpu_filter_csv="${GPU_OUTPUT_CSV}"
    fi
    if [[ -n "${gpu_filter_csv}" && -f "${gpu_filter_csv}" ]]; then
        GPU_NUM_ENVS_REMAINING="$(filter_num_envs_not_in_csv "${gpu_filter_csv}" warp "${GPU_NUM_ENVS_REQUESTED[@]}")"
        read -r -a GPU_NUM_ENVS_ARGS <<< "${GPU_NUM_ENVS_REMAINING}"
    else
        GPU_NUM_ENVS_ARGS=("${GPU_NUM_ENVS_REQUESTED[@]}")
    fi
    if [[ ${#GPU_NUM_ENVS_ARGS[@]} -eq 0 ]]; then
        echo "All requested GPU num_envs already benchmarked; skip GPU sweep." >&2
        return 0
    fi
    echo "=== GPU warp env sweep (${GPU_NUM_ENVS_ARGS[*]}) ===" >&2
    # env-only layer: warp sim on cuda; --ppo-devices cpu avoids cuda validation noise.
    uv run --extra warp python scripts/flygym/benchmark_odor_obstacle_navigation_training.py \
        --runners warp \
        --ppo-devices cpu \
        --env-devices cuda \
        --num-envs "${GPU_NUM_ENVS_ARGS[@]}" \
        --output-csv "${GPU_OUTPUT_CSV}" \
        "${common_benchmark_args[@]}" \
        "${extra_benchmark_args[@]+"${extra_benchmark_args[@]}"}"
}

echo "=== Resume L40 env sweep ===" >&2
echo "Prior job: ${PRIOR_JOB_ID}" >&2
echo "Probe JSON: ${PROBE_JSON} (max_gpu_envs=${MAX_GPU_ENVS})" >&2
echo "Prior CPU CSV: ${PRIOR_CPU_CSV}" >&2
echo "RUN_GPU_FIRST=${RUN_GPU_FIRST} SKIP_CPU_SWEEP=${SKIP_CPU_SWEEP} SKIP_GPU_SWEEP=${SKIP_GPU_SWEEP}" >&2

if [[ "${RUN_GPU_FIRST}" == "1" ]]; then
    run_gpu_sweep
    run_cpu_sweep
else
    run_cpu_sweep
    run_gpu_sweep
fi

if [[ "${MERGE_CPU_CSVS}" == "1" && -f "${PRIOR_CPU_CSV}" ]]; then
    if [[ -f "${CPU_OUTPUT_CSV}" ]]; then
        echo "=== Merging CPU sweep CSVs ===" >&2
        merge_cpu_sweep_csvs "${PRIOR_CPU_CSV}" "${CPU_OUTPUT_CSV}" "${MERGED_CPU_CSV}"
    else
        echo "No new CPU CSV at ${CPU_OUTPUT_CSV}; merged file not written." >&2
    fi
fi

echo "Done." >&2
echo "Probe (reused): ${PROBE_JSON}" >&2
if [[ -f "${CPU_OUTPUT_CSV}" ]]; then
    echo "CPU resume CSV: ${CPU_OUTPUT_CSV}" >&2
fi
if [[ -f "${GPU_OUTPUT_CSV}" ]]; then
    echo "GPU sweep CSV: ${GPU_OUTPUT_CSV}" >&2
fi
if [[ -f "${MERGED_CPU_CSV}" ]]; then
    echo "CPU merged CSV: ${MERGED_CPU_CSV}" >&2
fi
