#!/usr/bin/env bash
# Sync the locked CUDA 12.6 PyTorch venv, fix incomplete NVIDIA wheels, and
# configure the dynamic linker for CUDA (WSL + venv). Use from launch scripts:
#
#   source "${SCRIPT_DIR}/ensure_cuda_venv.sh"
#   ensure_cuda_venv "${REPO_ROOT}"
#   verify_cuda_runtime "${REPO_ROOT}"

ensure_cuda_venv() {
    local repo_root="$1"
    shift || true
    local extra_flags=("$@")

    cd "${repo_root}"

    if ((${#extra_flags[@]})); then
        uv sync --frozen "${extra_flags[@]}"
    else
        uv sync --frozen --extra warp
    fi

    local py_minor
    py_minor="$("${repo_root}/.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    local site_packages="${repo_root}/.venv/lib/python${py_minor}/site-packages"
    local cudnn_lib="${site_packages}/nvidia/cudnn/lib/libcudnn.so.9"

    if [[ ! -f "${cudnn_lib}" ]]; then
        echo "CUDA wheel libraries incomplete; reinstalling NVIDIA PyTorch dependencies..." >&2
        if ((${#extra_flags[@]})); then
            uv sync --frozen "${extra_flags[@]}" \
                --reinstall-package nvidia-cudnn-cu12 \
                --reinstall-package nvidia-cusparselt-cu12 \
                --reinstall-package nvidia-nccl-cu12 \
                --reinstall-package nvidia-nvshmem-cu12
        else
            uv sync --frozen --extra warp \
                --reinstall-package nvidia-cudnn-cu12 \
                --reinstall-package nvidia-cusparselt-cu12 \
                --reinstall-package nvidia-nccl-cu12 \
                --reinstall-package nvidia-nvshmem-cu12
        fi
    fi

    configure_cuda_ld_path "${repo_root}"
}

# Prepend CUDA library directories so Warp/PyTorch load the right libcuda.so.
# - WSL2: prefer /usr/lib/wsl/lib over the distro stub in /lib
# - All Linux: prepend NVIDIA wheels shipped in the project venv (cuDNN, NCCL, …)
configure_cuda_ld_path() {
    local repo_root="$1"
    local -a paths=()

    if [[ -f /usr/lib/wsl/lib/libcuda.so.1 ]]; then
        paths+=("/usr/lib/wsl/lib")
    fi

    local py_minor
    py_minor="$("${repo_root}/.venv/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    local site_packages="${repo_root}/.venv/lib/python${py_minor}/site-packages/nvidia"
    local pkg lib_dir
    for pkg in "${site_packages}"/*/lib; do
        if [[ -d "${pkg}" ]]; then
            paths+=("${pkg}")
        fi
    done

    local path
    for path in "${paths[@]}"; do
        case ":${LD_LIBRARY_PATH:-}:" in
            *":${path}:"*) ;;
            *) LD_LIBRARY_PATH="${path}:${LD_LIBRARY_PATH:-}" ;;
        esac
    done
    export LD_LIBRARY_PATH
}

# Fail fast when a GPU is present but Warp cannot use it (common WSL libcuda issue).
verify_cuda_runtime() {
    local repo_root="$1"
    configure_cuda_ld_path "${repo_root}"

    if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi -L &>/dev/null 2>&1; then
        return 0
    fi

    if "${repo_root}/.venv/bin/python" -c "
import sys
import torch
import warp as wp

if not torch.cuda.is_available():
    print('PyTorch cannot use CUDA.', file=sys.stderr)
    raise SystemExit(1)
if not any(d.is_cuda for d in wp.get_devices()):
    print('Warp cannot use CUDA (flygym env_device=cuda needs Warp).', file=sys.stderr)
    raise SystemExit(1)
" 2>&1; then
        return 0
    fi

    echo "CUDA runtime check failed. On WSL2, ensure LD_LIBRARY_PATH includes /usr/lib/wsl/lib" >&2
    echo "  export LD_LIBRARY_PATH=\"/usr/lib/wsl/lib:\${LD_LIBRARY_PATH:-}\"" >&2
    echo "Avoid installing nvidia-driver-* inside WSL; use the Windows host driver only." >&2
    exit 1
}
