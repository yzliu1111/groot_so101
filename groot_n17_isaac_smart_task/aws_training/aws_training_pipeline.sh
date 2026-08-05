#!/usr/bin/env bash

# Single AWS H100 entrypoint for the manifest-driven SO101 training batch.
#
# The repository may be checked out anywhere. The one project-local storage
# link is deliberately simple:
#
#   $SMART_PROJECT/outputs -> $AWS_OUTPUTS_TARGET
#   default target: /opt/dlami/nvme/smart_project_outputs
#
# GR00T, caches and temporary files use direct paths below AWS_STORAGE_ROOT, so
# no second project symlink is needed. This script never formats or mounts a
# block device.

set -Eeuo pipefail

readonly GROOT_COMMIT="9c7e746b2cd37a810070a98ef41d290a07e806c2"
readonly UV_VERSION="0.11.29"
readonly BASE_MODEL="nvidia/GR00T-N1.7-3B"
readonly BASE_MODEL_REVISION="2fc962b973bccdd5d8ce4f67cc63b264d6886495"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
DETECTED_SMART_PROJECT="$(cd -- "$SCRIPT_DIR/../../.." && pwd -P)"
AWS_STORAGE_ROOT="${AWS_STORAGE_ROOT:-/opt/dlami/nvme}"
CONFIGURED_SMART_PROJECT="${SMART_PROJECT:-}"
SMART_PROJECT="$DETECTED_SMART_PROJECT"
AWS_OUTPUTS_TARGET="${AWS_OUTPUTS_TARGET:-$AWS_STORAGE_ROOT/smart_project_outputs}"
GROOT_ROOT="${GROOT_ROOT:-$AWS_STORAGE_ROOT/Isaac-GR00T}"
AWS_PREPARED_ROOT="$SMART_PROJECT/outputs"
AWS_TUNING_PREPARED_TARGET="$AWS_OUTPUTS_TARGET/groot_so101_synthetic_datasets/aws_third_training_20260804"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-$SMART_PROJECT/outputs/groot_so101_synthetic_finetune}"
HF_HOME="${HF_HOME:-$AWS_STORAGE_ROOT/cache/huggingface}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$AWS_STORAGE_ROOT/cache/uv}"
UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$AWS_STORAGE_ROOT/cache/uv-python}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$AWS_STORAGE_ROOT/cache/xdg}"
TORCH_HOME="${TORCH_HOME:-$AWS_STORAGE_ROOT/cache/torch}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$AWS_STORAGE_ROOT/cache/torchinductor}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$AWS_STORAGE_ROOT/cache/triton}"
TMPDIR="$AWS_STORAGE_ROOT/tmp"
UV_INSTALL_DIR="$AWS_STORAGE_ROOT/bin"
UV_BIN="$UV_INSTALL_DIR/uv"
# Emergency low-water mark only. This is not an estimate of, or reservation
# for, the whole training batch; retained checkpoints legitimately consume the
# production volume as runs finish.
AWS_MIN_FREE_GIB="${AWS_MIN_FREE_GIB:-200}"
BASE_MODEL_PATH=""

FULL_FINETUNE_DIR="$SCRIPT_DIR/../full_finetune_so101"
BATCH_SCRIPT="$SCRIPT_DIR/aws_training_batch.py"
TRAIN_WRAPPER="$FULL_FINETUNE_DIR/train_so101_synthetic_groot.py"
MANIFEST="$SCRIPT_DIR/aws_tuning_8_manifest.json"
VERSION_CONTRACT="$SCRIPT_DIR/version_contract.py"
GROOT_PYTHON="$GROOT_ROOT/.venv/bin/python"
AWS_DATA_PYTHON="$GROOT_PYTHON"

export AWS_STORAGE_ROOT AWS_OUTPUTS_TARGET SMART_PROJECT
export GROOT_ROOT AWS_PREPARED_ROOT TRAIN_OUTPUT_ROOT
export HF_HOME UV_CACHE_DIR UV_PYTHON_INSTALL_DIR XDG_CACHE_HOME TORCH_HOME TORCHINDUCTOR_CACHE_DIR
export TRITON_CACHE_DIR TMPDIR AWS_DATA_PYTHON

log() {
    printf '[aws-pipeline] %s\n' "$*"
}

die() {
    printf '[aws-pipeline] ERROR: %s\n' "$*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    printf '[aws-pipeline] ERROR: command failed at line %s (exit=%s)\n' \
        "${BASH_LINENO[0]}" "$exit_code" >&2
    exit "$exit_code"
}
trap on_error ERR

usage() {
    cat <<'EOF'
Usage:
  aws_training_pipeline.sh paths
  aws_training_pipeline.sh storage-link
  aws_training_pipeline.sh bootstrap
  aws_training_pipeline.sh auth
  aws_training_pipeline.sh preflight
  aws_training_pipeline.sh audit    [all|DATASET_ID ...] [-- batch options]
  aws_training_pipeline.sh prepare  [all|DATASET_ID ...] [-- batch options]
  aws_training_pipeline.sh verify   [all|DATASET_ID ...] [-- batch options]
  aws_training_pipeline.sh stats    [all|DATASET_ID ...] [-- batch options]
  aws_training_pipeline.sh dry-run  [all|DATASET_ID ...] [-- batch options]
  aws_training_pipeline.sh smoke    [all|DATASET_ID ...] [-- batch options]
  aws_training_pipeline.sh train    [all|DATASET_ID ...] [-- batch options]

Examples:
  AWS_STORAGE_ROOT=/mnt/aws-training ./aws_training_pipeline.sh paths
  AWS_STORAGE_ROOT=/mnt/aws-training ./aws_training_pipeline.sh storage-link
  ./aws_training_pipeline.sh audit all --deep-video
  ./aws_training_pipeline.sh stats real003 --run-tag aws-third-20260804-real003
  ./aws_training_pipeline.sh smoke real003 --run-tag aws-third-20260804-real003
  ./aws_training_pipeline.sh train real003 --run-tag aws-third-20260804-real003

Environment overrides:
  AWS_STORAGE_ROOT   Already-mounted non-root data volume (DLAMI default /opt/dlami/nvme)
  AWS_OUTPUTS_TARGET Physical outputs directory (default /opt/dlami/nvme/smart_project_outputs)
  SMART_PROJECT      Optional consistency check; must resolve to this script's project root
  GROOT_ROOT         Exact Isaac-GR00T checkout
  TRAIN_OUTPUT_ROOT  Checkpoint output root
  HF_HOME            Hugging Face cache
  UV_CACHE_DIR       uv cache
  UV_PYTHON_INSTALL_DIR  uv-managed Python storage
  XDG_CACHE_HOME     compiler/runtime cache
  AWS_MIN_FREE_GIB   Emergency free-space floor on the production mount (default 200)

The script discovers SMART_PROJECT from its own location, so the AWS checkout path
does not need to match the local workstation. Run storage-link once before uploading
prepared data or running bootstrap. It creates only SMART_PROJECT/outputs; GR00T and
caches use direct paths below AWS_STORAGE_ROOT. The script does not run mkfs or mount.
Other path overrides do not bypass storage validation. The standalone `--` separator
is optional and is removed before forwarding.
EOF
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

sanitize_python_runtime_env() {
    local variable
    for variable in PYTHONOPTIMIZE PYTHONPATH PYTHONHOME; do
        if [[ -n "${!variable:-}" ]]; then
            log "clearing inherited $variable for the pinned GR00T runtime"
            unset "$variable"
        fi
    done
    export PYTHONNOUSERSITE=1
}

run_groot_python() {
    env -u PYTHONOPTIMIZE -u PYTHONPATH -u PYTHONHOME \
        PYTHONNOUSERSITE=1 "$GROOT_PYTHON" "$@"
}

find_first_prepared_video() {
    local prepared_root="$1"
    # SMART_PROJECT/outputs is deliberately a command-line symlink. -H follows
    # that entry link without following symlinks nested inside prepared data.
    find -H "$prepared_root" -type f -name '*.mp4' -print -quit
}

uv_version_matches() {
    local output="$1"
    local command_name actual_version ignored_suffix
    read -r command_name actual_version ignored_suffix <<<"$output"
    [[ "$command_name" == "uv" && "$actual_version" == "$UV_VERSION" ]]
}

check_uv_version() {
    local output
    output="$("$UV_BIN" --version)"
    uv_version_matches "$output" || \
        die "expected uv $UV_VERSION, got $output"
    log "uv=$output"
}

validate_project_layout() {
    require_command realpath
    [[ -d "$SMART_PROJECT" ]] || die "project root does not exist: $SMART_PROJECT"

    if [[ -n "$CONFIGURED_SMART_PROJECT" ]]; then
        [[ -d "$CONFIGURED_SMART_PROJECT" ]] || \
            die "configured SMART_PROJECT does not exist: $CONFIGURED_SMART_PROJECT"
        local configured_project
        configured_project="$(realpath -e "$CONFIGURED_SMART_PROJECT")"
        [[ "$configured_project" == "$DETECTED_SMART_PROJECT" ]] || \
            die "SMART_PROJECT must resolve to the project containing this script: configured=$configured_project detected=$DETECTED_SMART_PROJECT"
    fi

    [[ -f "$BATCH_SCRIPT" && -f "$TRAIN_WRAPPER" && -f "$MANIFEST" && \
        -f "$VERSION_CONTRACT" ]] || \
        die "AWS training batch/manifest, version contract, or shared full-finetune wrapper is missing"
}

show_paths() {
    printf 'SMART_PROJECT=%s\n' "$SMART_PROJECT"
    printf 'AWS_STORAGE_ROOT=%s\n' "$AWS_STORAGE_ROOT"
    printf 'AWS_OUTPUTS_LINK=%s\n' "$AWS_PREPARED_ROOT"
    printf 'AWS_OUTPUTS_TARGET=%s\n' "$AWS_OUTPUTS_TARGET"
    printf 'AWS_TUNING_PREPARED_TARGET=%s\n' "$AWS_TUNING_PREPARED_TARGET"
    printf 'GROOT_ROOT=%s\n' "$GROOT_ROOT"
    printf 'TRAIN_OUTPUT_ROOT=%s\n' "$TRAIN_OUTPUT_ROOT"
    printf 'HF_HOME=%s\n' "$HF_HOME"
    printf 'UV_CACHE_DIR=%s\n' "$UV_CACHE_DIR"
    printf 'TORCH_HOME=%s\n' "$TORCH_HOME"
    printf 'TORCHINDUCTOR_CACHE_DIR=%s\n' "$TORCHINDUCTOR_CACHE_DIR"
    printf 'TRITON_CACHE_DIR=%s\n' "$TRITON_CACHE_DIR"
    printf 'TMPDIR=%s\n' "$TMPDIR"
}

assert_dedicated_storage_mount() {
    require_command realpath
    require_command findmnt
    [[ -d "$AWS_STORAGE_ROOT" ]] || \
        die "AWS_STORAGE_ROOT does not exist; mount the large volume first: $AWS_STORAGE_ROOT"
    [[ -w "$AWS_STORAGE_ROOT" ]] || \
        die "AWS_STORAGE_ROOT is not writable by $(id -un): $AWS_STORAGE_ROOT"

    local storage_real storage_device root_device storage_mount
    storage_real="$(realpath -e "$AWS_STORAGE_ROOT")"
    storage_device="$(findmnt -n -o MAJ:MIN -T "$storage_real")"
    root_device="$(findmnt -n -o MAJ:MIN -T /)"
    storage_mount="$(findmnt -n -o TARGET -T "$storage_real")"
    [[ -n "$storage_device" && -n "$root_device" && -n "$storage_mount" ]] || \
        die "cannot identify storage/root mounts with findmnt"
    [[ "$storage_device" != "$root_device" ]] || \
        die "AWS_STORAGE_ROOT is still on the root filesystem ($storage_device, mount=$storage_mount). Mount the large data volume and point AWS_STORAGE_ROOT at it before any setup."
    log "production storage=$storage_real mount=$storage_mount device=$storage_device root_device=$root_device"
}

assert_outputs_target_disjoint() {
    local project_real target_real
    project_real="$(realpath -e "$SMART_PROJECT")"
    target_real="$(realpath -m "$AWS_OUTPUTS_TARGET")"
    case "$target_real" in
        "$project_real"|"$project_real"/*)
            die "AWS_OUTPUTS_TARGET must be outside the project checkout: target=$target_real project=$project_real"
            ;;
    esac
    case "$project_real" in
        "$target_real"|"$target_real"/*)
            die "project checkout must not be inside AWS_OUTPUTS_TARGET: project=$project_real target=$target_real"
            ;;
    esac
}

check_available_storage() {
    require_command df
    require_command awk
    [[ "$AWS_MIN_FREE_GIB" =~ ^[1-9][0-9]*$ ]] || \
        die "AWS_MIN_FREE_GIB must be a positive integer, got $AWS_MIN_FREE_GIB"

    local available_kib required_kib
    available_kib="$(df -Pk "$AWS_STORAGE_ROOT" | awk 'NR == 2 {print $4}')"
    [[ "$available_kib" =~ ^[0-9]+$ ]] || \
        die "could not read free space for $AWS_STORAGE_ROOT"
    required_kib=$((AWS_MIN_FREE_GIB * 1024 * 1024))
    (( available_kib >= required_kib )) || \
        die "production storage is below the emergency floor: available=$((available_kib / 1024 / 1024))GiB floor=${AWS_MIN_FREE_GIB}GiB"
    log "storage free=$((available_kib / 1024 / 1024))GiB emergency_floor=${AWS_MIN_FREE_GIB}GiB"
}

check_symlink_slot() {
    local label="$1"
    local link_path="$2"
    local target_path="$3"

    if [[ -L "$link_path" ]]; then
        [[ -e "$link_path" ]] || \
            die "$label is a dangling symlink: $link_path -> $(readlink "$link_path")"
        [[ -d "$target_path" ]] || \
            die "$label target is not a directory: $target_path"
        local actual_target expected_target
        actual_target="$(realpath -e "$link_path")"
        expected_target="$(realpath -e "$target_path")"
        [[ "$actual_target" == "$expected_target" ]] || \
            die "$label points to the wrong target: actual=$actual_target expected=$expected_target"
    elif [[ -e "$link_path" ]]; then
        die "$label path already exists and is not a symlink: $link_path. This script will not move, delete, or overwrite it."
    fi
}

create_storage_symlink() {
    local label="$1"
    local link_path="$2"
    local target_path="$3"
    if [[ ! -L "$link_path" ]]; then
        ln -s "$(realpath -e "$target_path")" "$link_path"
        log "created $label symlink: $link_path -> $(realpath -e "$target_path")"
    fi
    check_symlink_slot "$label" "$link_path" "$target_path"
}

storage_link() {
    assert_dedicated_storage_mount
    check_available_storage
    assert_outputs_target_disjoint

    # Validate the existing project path before creating anything. Conflicts
    # fail closed; the script never migrates, removes or overwrites user data.
    check_symlink_slot "outputs" "$AWS_PREPARED_ROOT" "$AWS_OUTPUTS_TARGET"

    assert_future_path_on_production_storage "outputs target" "$AWS_OUTPUTS_TARGET"
    mkdir -p "$AWS_OUTPUTS_TARGET"
    assert_path_on_production_storage "outputs target" "$AWS_OUTPUTS_TARGET"
    create_storage_symlink "outputs" "$AWS_PREPARED_ROOT" "$AWS_OUTPUTS_TARGET"
    assert_outputs_link_ready

    log "storage-link PASS"
    log "upload prepared v2.1 data to $AWS_TUNING_PREPARED_TARGET"
    log "logical project path is $AWS_PREPARED_ROOT/groot_so101_synthetic_datasets/aws_third_training_20260804"
}

assert_outputs_link_ready() {
    assert_dedicated_storage_mount
    check_available_storage
    assert_outputs_target_disjoint
    [[ -L "$AWS_PREPARED_ROOT" ]] || \
        die "outputs symlink is missing; run: AWS_STORAGE_ROOT=$AWS_STORAGE_ROOT $0 storage-link"
    check_symlink_slot "outputs" "$AWS_PREPARED_ROOT" "$AWS_OUTPUTS_TARGET"
    assert_path_on_production_storage "project outputs" "$AWS_PREPARED_ROOT"
    assert_future_path_on_production_storage "Isaac-GR00T checkout" "$GROOT_ROOT"
    assert_future_path_on_production_storage "training output" "$TRAIN_OUTPUT_ROOT"
    assert_future_path_on_production_storage "Hugging Face cache" "$HF_HOME"
    assert_future_path_on_production_storage "uv cache" "$UV_CACHE_DIR"
    assert_future_path_on_production_storage "uv Python installs" "$UV_PYTHON_INSTALL_DIR"
    assert_future_path_on_production_storage "XDG cache" "$XDG_CACHE_HOME"
    assert_future_path_on_production_storage "torch cache" "$TORCH_HOME"
    assert_future_path_on_production_storage "torchinductor cache" "$TORCHINDUCTOR_CACHE_DIR"
    assert_future_path_on_production_storage "triton cache" "$TRITON_CACHE_DIR"
    assert_future_path_on_production_storage "temporary files" "$TMPDIR"
    assert_future_path_on_production_storage "uv executable directory" "$UV_INSTALL_DIR"
}

assert_known_host() {
    [[ "$(uname -s)" == "Linux" ]] || die "expected Linux, got $(uname -s)"
    [[ "$(uname -m)" == "x86_64" ]] || die "expected x86_64, got $(uname -m)"
    [[ -r /etc/os-release ]] || die "/etc/os-release is missing"

    local os_id os_version
    os_id="$(sed -n 's/^ID=//p' /etc/os-release | tr -d '"')"
    os_version="$(sed -n 's/^VERSION_ID=//p' /etc/os-release | tr -d '"')"
    [[ "$os_id" == "ubuntu" ]] || die "expected Ubuntu, got ID=$os_id"
    [[ "$os_version" == "24.04" ]] || \
        die "expected the validated Ubuntu 24.04 DLAMI, got VERSION_ID=$os_version"
    log "host: Ubuntu $os_version x86_64"
}

detect_cuda_home() {
    local candidate
    if [[ -n "${CUDA_HOME:-}" ]]; then
        candidate="$CUDA_HOME"
    elif [[ -x /usr/local/cuda-13.2/bin/nvcc ]]; then
        candidate=/usr/local/cuda-13.2
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        candidate=/usr/local/cuda
    else
        die "CUDA toolkit not found; use a CUDA DLAMI. This script will not install or downgrade the driver/toolkit"
    fi
    [[ -x "$candidate/bin/nvcc" ]] || die "CUDA_HOME has no executable nvcc: $candidate"
    CUDA_HOME="$(realpath -e "$candidate")"
    export CUDA_HOME
    case ":$PATH:" in
        *":$CUDA_HOME/bin:"*) ;;
        *) export PATH="$CUDA_HOME/bin:$PATH" ;;
    esac
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$CUDA_HOME/lib64:"*) ;;
        *) export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
    log "CUDA_HOME=$CUDA_HOME"
}

cuda_toolkit_version_from_output() {
    local output="$1"
    local version
    version="$(sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p' <<<"$output" | sed -n '1p')"
    [[ "$version" =~ ^[0-9]+\.[0-9]+$ ]] || return 1
    printf '%s\n' "$version"
}

cuda_toolkit_version() {
    local output version
    output="$("$CUDA_HOME/bin/nvcc" --version 2>&1)"
    version="$(cuda_toolkit_version_from_output "$output")" || \
        die "cannot parse CUDA toolkit version from nvcc output: $output"
    printf '%s\n' "$version"
}

check_cuda13_native_triton_support() {
    local cuda_version="$1"

    if [[ -n "${TRITON_PTXAS_PATH:-}" ]]; then
        log "ignoring inherited TRITON_PTXAS_PATH for the pinned x86 H100 stack: $TRITON_PTXAS_PATH"
        unset TRITON_PTXAS_PATH
    fi

    run_groot_python - "$cuda_version" <<'PY'
import importlib.util
import inspect
from pathlib import Path
import site
import sys

cuda_version = sys.argv[1]
import triton
from triton.backends.nvidia.compiler import get_ptxas, ptx_get_version

if sys.flags.optimize != 0:
    raise RuntimeError(f"Python optimization must be disabled, got optimize={sys.flags.optimize}")

major, minor = map(int, cuda_version.split("."))

# The pinned environment is Triton 3.5.0. It handles CUDA major >=13 natively.
# Reject the repository's legacy PyTorch-2.7/Triton-3.3.1 monkey patch so the
# frozen environment cannot be silently mutated between bootstrap and train.
legacy_files = []
for package_root in site.getsitepackages():
    root = Path(package_root)
    legacy_files.extend(root.glob("triton_cuda13_patch.pth"))
    legacy_files.extend(root.glob("triton_cuda13_patch.py"))
assert importlib.util.find_spec("triton_cuda13_patch") is None, (
    "legacy triton_cuda13_patch module is installed; remove .venv and rerun bootstrap"
)
assert not legacy_files, {
    "legacy_cuda13_patch_files": [str(path) for path in legacy_files]
}

source = inspect.getsource(ptx_get_version)
assert "if major >= 13:" in source, "Triton lacks native CUDA major >=13 support"
assert "if major == 13:" not in source, "legacy CUDA13 source patch detected"

if major >= 13:
    ptx_version = ptx_get_version(cuda_version)
    expected = 90 + (major - 13) * 10 + minor
    assert ptx_version == expected, {
        "cuda_toolkit": cuda_version,
        "expected_ptx": expected,
        "actual_ptx": ptx_version,
    }
    system_mapping = f"toolkit={cuda_version}->PTX{ptx_version}"
else:
    system_mapping = f"toolkit={cuda_version} (<13; CUDA13 mapping not exercised)"

# Triton 3.5 from the pinned cu128 environment uses its bundled ptxas unless an
# override is supplied. Verify that compiler layer independently from system nvcc.
ptxas = get_ptxas()
ptxas_path = Path(ptxas.path).resolve()
triton_root = Path(triton.__file__).resolve().parent
assert ptxas_path.is_relative_to(triton_root), {
    "expected_bundled_triton_ptxas_under": str(triton_root),
    "actual_ptxas": str(ptxas_path),
}
assert ptxas.version == "12.8", {
    "expected_bundled_ptxas": "12.8",
    "actual_bundled_ptxas": ptxas.version,
}
bundled_ptx = ptx_get_version(ptxas.version)
print(
    f"native CUDA mapping OK {system_mapping} "
    f"bundled_ptxas={ptxas.version}->PTX{bundled_ptx} path={ptxas_path}"
)
PY
    log "pinned Triton environment and native CUDA 13+ strategy verified"
}

sudo_prefix() {
    if [[ "$(id -u)" -eq 0 ]]; then
        return 0
    fi
    command -v sudo >/dev/null 2>&1 || die "bootstrap needs root or sudo for apt packages"
    printf '%s' sudo
}

bootstrap() {
    assert_known_host
    require_command nvidia-smi
    detect_cuda_home
    assert_outputs_link_ready

    local sudo_cmd
    sudo_cmd="$(sudo_prefix)"
    if [[ -n "$sudo_cmd" ]]; then
        "$sudo_cmd" apt-get update
        "$sudo_cmd" apt-get install -y --no-install-recommends \
            ca-certificates curl git git-lfs ffmpeg libaio-dev build-essential rsync tmux
    else
        apt-get update
        apt-get install -y --no-install-recommends \
            ca-certificates curl git git-lfs ffmpeg libaio-dev build-essential rsync tmux
    fi
    git lfs install --skip-repo

    mkdir -p \
        "$TRAIN_OUTPUT_ROOT" "$HF_HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" \
        "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" \
        "$TMPDIR" "$UV_INSTALL_DIR" "$UV_PYTHON_INSTALL_DIR"

    if [[ -e "$GROOT_ROOT" && ! -d "$GROOT_ROOT/.git" ]]; then
        die "GROOT_ROOT exists but is not a Git checkout: $GROOT_ROOT"
    fi
    if [[ ! -d "$GROOT_ROOT/.git" ]]; then
        log "cloning Isaac-GR00T without deployment/evaluation submodules"
        GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none \
            https://github.com/NVIDIA/Isaac-GR00T.git "$GROOT_ROOT"
    fi

    local groot_status
    if ! groot_status="$(git -C "$GROOT_ROOT" status --porcelain --untracked-files=no)"; then
        die "cannot inspect Isaac-GR00T worktree before checkout: $GROOT_ROOT"
    fi
    if [[ -n "$groot_status" ]]; then
        die "Isaac-GR00T has tracked local changes; refusing to switch commits: $GROOT_ROOT"
    fi
    git -C "$GROOT_ROOT" fetch origin "$GROOT_COMMIT"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$GROOT_ROOT" checkout --detach "$GROOT_COMMIT"
    [[ "$(git -C "$GROOT_ROOT" rev-parse HEAD)" == "$GROOT_COMMIT" ]] || \
        die "failed to select pinned Isaac-GR00T commit $GROOT_COMMIT"

    log "installing exact uv $UV_VERSION into $UV_INSTALL_DIR"
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | \
        env UV_INSTALL_DIR="$UV_INSTALL_DIR" UV_NO_MODIFY_PATH=1 sh
    [[ -x "$UV_BIN" ]] || die "uv installer did not create $UV_BIN"
    check_uv_version

    (
        cd "$GROOT_ROOT"
        "$UV_BIN" sync --frozen --python 3.12
    )
    [[ -x "$GROOT_PYTHON" ]] || die "GR00T Python was not created: $GROOT_PYTHON"
    run_groot_python -c "import gr00t; print('GR00T import OK')"
    local cuda_version
    cuda_version="$(cuda_toolkit_version)"
    check_cuda13_native_triton_support "$cuda_version"

    log "bootstrap PASS"
    log "next: $0 auth"
    log "then: $0 preflight"
}

huggingface_auth() {
    assert_outputs_link_ready
    [[ -x "$GROOT_ROOT/.venv/bin/hf" ]] || die "run bootstrap first; hf CLI is missing"
    mkdir -p "$HF_HOME"
    assert_path_on_production_storage "Hugging Face token store" "$HF_HOME"
    log "Hugging Face token store=$HF_HOME"
    "$GROOT_ROOT/.venv/bin/hf" auth login
}

assert_path_on_production_storage() {
    local label="$1"
    local path="$2"
    local storage_real path_real storage_mount path_mount

    [[ -e "$path" ]] || die "$label path does not exist: $path"
    storage_real="$(realpath -e "$AWS_STORAGE_ROOT")"
    path_real="$(realpath -e "$path")"
    case "$path_real" in
        "$storage_real"|"$storage_real"/*) ;;
        *) die "$label resolves outside AWS_STORAGE_ROOT: $path_real (root=$storage_real)" ;;
    esac

    storage_mount="$(findmnt -n -o TARGET -T "$storage_real")"
    path_mount="$(findmnt -n -o TARGET -T "$path_real")"
    [[ -n "$storage_mount" && "$path_mount" == "$storage_mount" ]] || \
        die "$label is not on production mount $storage_mount: path=$path_real mount=$path_mount"
    log "$label=$path_real mount=$path_mount"
}

assert_future_path_on_production_storage() {
    local label="$1"
    local path="$2"
    local storage_real path_real

    storage_real="$(realpath -e "$AWS_STORAGE_ROOT")"
    path_real="$(realpath -m "$path")"
    case "$path_real" in
        "$storage_real"|"$storage_real"/*) ;;
        *) die "$label would resolve outside AWS_STORAGE_ROOT before setup writes it: $path_real (root=$storage_real)" ;;
    esac
}

check_storage() {
    assert_outputs_link_ready
    assert_path_on_production_storage "Isaac-GR00T checkout" "$GROOT_ROOT"
    assert_path_on_production_storage "prepared data" "$AWS_PREPARED_ROOT"
    assert_path_on_production_storage "training output" "$TRAIN_OUTPUT_ROOT"
    assert_path_on_production_storage "Hugging Face cache" "$HF_HOME"
    assert_path_on_production_storage "uv cache" "$UV_CACHE_DIR"
    assert_path_on_production_storage "uv Python installs" "$UV_PYTHON_INSTALL_DIR"
    assert_path_on_production_storage "XDG cache" "$XDG_CACHE_HOME"
    assert_path_on_production_storage "torch cache" "$TORCH_HOME"
    assert_path_on_production_storage "torchinductor cache" "$TORCHINDUCTOR_CACHE_DIR"
    assert_path_on_production_storage "triton cache" "$TRITON_CACHE_DIR"
    assert_path_on_production_storage "temporary files" "$TMPDIR"
}

check_gpu_and_system_runtime() {
    require_command nvidia-smi
    require_command ffmpeg
    require_command find
    require_command grep
    detect_cuda_home

    local -a gpu_rows
    mapfile -t gpu_rows < <(
        nvidia-smi --query-gpu=name,memory.total,driver_version \
            --format=csv,noheader,nounits
    )
    [[ "${#gpu_rows[@]}" -eq 1 ]] || \
        die "expected exactly one H100, found ${#gpu_rows[@]} GPUs: ${gpu_rows[*]}"
    [[ "${gpu_rows[0]}" == *H100* ]] || die "expected H100, got: ${gpu_rows[0]}"
    log "gpu=${gpu_rows[0]}"

    local ffmpeg_line ffmpeg_major cuda_version
    ffmpeg_line="$(ffmpeg -version 2>&1 | sed -n '1p')"
    ffmpeg_major="$(sed -E 's/^ffmpeg version ([0-9]+).*/\1/' <<<"$ffmpeg_line")"
    [[ "$ffmpeg_major" =~ ^[0-9]+$ ]] || die "cannot parse FFmpeg version: $ffmpeg_line"
    (( ffmpeg_major >= 4 && ffmpeg_major <= 7 )) || \
        die "torchcodec requires FFmpeg major 4-7, got: $ffmpeg_line"
    local ffmpeg_decoders
    ffmpeg_decoders="$(ffmpeg -hide_banner -decoders 2>&1)"
    grep -qi av1 <<<"$ffmpeg_decoders" || die "FFmpeg has no AV1 decoder"
    log "$ffmpeg_line"
    "$CUDA_HOME/bin/nvcc" --version | sed -n '/release/p'
    cuda_version="$(cuda_toolkit_version)"
    log "system CUDA toolkit=$cuda_version; PyTorch CUDA runtime is checked separately"
}

check_groot_runtime() {
    [[ -d "$GROOT_ROOT/.git" ]] || die "Isaac-GR00T checkout missing: $GROOT_ROOT"
    [[ "$(git -C "$GROOT_ROOT" rev-parse HEAD)" == "$GROOT_COMMIT" ]] || \
        die "Isaac-GR00T must be exactly $GROOT_COMMIT; got $(git -C "$GROOT_ROOT" rev-parse HEAD)"
    local groot_status
    if ! groot_status="$(git -C "$GROOT_ROOT" status --porcelain --untracked-files=all)"; then
        die "cannot inspect Isaac-GR00T worktree during preflight: $GROOT_ROOT"
    fi
    [[ -z "$groot_status" ]] || \
        die "Isaac-GR00T worktree is dirty; formal stats/train require exact pinned code"
    [[ -x "$GROOT_PYTHON" ]] || die "GR00T Python missing: $GROOT_PYTHON"
    [[ -x "$UV_BIN" ]] || die "pinned uv missing: $UV_BIN"
    check_uv_version
    log "Isaac-GR00T commit=$GROOT_COMMIT"
    local cuda_version
    cuda_version="$(cuda_toolkit_version)"
    check_cuda13_native_triton_support "$cuda_version"

    env -u PYTHONOPTIMIZE -u PYTHONPATH -u PYTHONHOME \
        PYTHONNOUSERSITE=1 PYTHONPATH="$SCRIPT_DIR" "$GROOT_PYTHON" - <<'PY'
import sys
import deepspeed
import flash_attn
import pyarrow
import torch
import torchcodec
import transformers
import triton
import gr00t  # noqa: F401
from version_contract import version_mismatches

if sys.flags.optimize != 0:
    raise RuntimeError(f"Python optimization must be disabled, got optimize={sys.flags.optimize}")

expected = {
    "torch": "2.9.0",
    "torch_cuda": "12.8",
    "torchcodec": "0.8.0",
    "triton": "3.5.0",
    "deepspeed": "0.17.6",
    "flash_attn": "2.8.3",
    "transformers": "4.57.3",
    "pyarrow": "23.0.1",
}
actual = {
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchcodec": torchcodec.__version__,
    "triton": triton.__version__,
    "deepspeed": deepspeed.__version__,
    "flash_attn": flash_attn.__version__,
    "transformers": transformers.__version__,
    "pyarrow": pyarrow.__version__,
}
assert sys.version_info[:2] == (3, 12), sys.version
mismatches = version_mismatches(expected, actual)
assert not mismatches, {"version_mismatches": mismatches}
assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
assert "H100" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print("python=", sys.version.split()[0])
print("versions=", actual)
print("torch_gpu=", torch.cuda.get_device_name(0))

compiled = torch.compile(lambda x: torch.sin(x) + 1)
value = compiled(torch.randn(1024, device="cuda"))
torch.cuda.synchronize()
assert torch.isfinite(value).all()
print("torch.compile OK")
PY

    local sample_video
    sample_video="${AWS_PREFLIGHT_VIDEO:-}"
    if [[ -z "$sample_video" ]]; then
        sample_video="$(find_first_prepared_video "$AWS_PREPARED_ROOT")"
    fi
    [[ -n "$sample_video" && -f "$sample_video" ]] || \
        die "no prepared MP4 found for torchcodec decode under $AWS_PREPARED_ROOT"
    run_groot_python - "$sample_video" <<'PY'
from pathlib import Path
import sys
import torch
from torchcodec.decoders import VideoDecoder

path = Path(sys.argv[1])
if sys.flags.optimize != 0:
    raise RuntimeError(f"Python optimization must be disabled, got optimize={sys.flags.optimize}")
decoder = VideoDecoder(str(path))
assert len(decoder) > 0, path
frame = decoder[0]
assert frame.ndim == 3 and frame.dtype == torch.uint8, (path, frame.shape, frame.dtype)
print(f"torchcodec decode OK path={path} frames={len(decoder)} shape={tuple(frame.shape)}")
PY
}

check_pipeline_contract() {
    [[ -f "$BATCH_SCRIPT" ]] || die "batch runner missing: $BATCH_SCRIPT"
    [[ -f "$TRAIN_WRAPPER" ]] || die "training wrapper missing: $TRAIN_WRAPPER"
    [[ -f "$MANIFEST" ]] || die "training manifest missing: $MANIFEST"
    [[ -f "$GROOT_ROOT/gr00t/experiment/launch_finetune.py" ]] || \
        die "upstream launch_finetune.py missing"

    local manifest_commit
    manifest_commit="$(
        run_groot_python -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["groot_commit"])' \
            "$MANIFEST"
    )"
    [[ "$manifest_commit" == "$GROOT_COMMIT" ]] || \
        die "manifest commit $manifest_commit does not match environment commit $GROOT_COMMIT"

    local manifest_model_revision
    manifest_model_revision="$(
        run_groot_python -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["base_model_revision"])' \
            "$MANIFEST"
    )"
    [[ "$manifest_model_revision" == "$BASE_MODEL_REVISION" ]] || \
        die "manifest model revision $manifest_model_revision does not match $BASE_MODEL_REVISION"

    local help_output
    help_output="$(
        run_groot_python "$GROOT_ROOT/gr00t/experiment/launch_finetune.py" --help 2>&1
    )"
    grep -q -- '--state-dropout-prob' <<<"$help_output" || \
        die "upstream launch_finetune.py lacks --state-dropout-prob"
    grep -q -- '--color-jitter-params' <<<"$help_output" || \
        die "upstream launch_finetune.py lacks --color-jitter-params"
    run_groot_python "$BATCH_SCRIPT" --help >/dev/null
    run_groot_python "$BATCH_SCRIPT" verify all \
        --manifest "$MANIFEST" >/dev/null
    run_groot_python "$BATCH_SCRIPT" dry-run all \
        --manifest "$MANIFEST" \
        --data-python "$AWS_DATA_PYTHON" \
        --groot-root "$GROOT_ROOT" \
        --groot-python "$GROOT_PYTHON" \
        --output-root "$TRAIN_OUTPUT_ROOT" \
        --run-tag preflight-contract >/dev/null
    log "all eight prepared dataset contracts OK"
    log "pipeline contract OK"
}

check_huggingface_access() {
    BASE_MODEL_PATH="$(run_groot_python - "$BASE_MODEL" "$BASE_MODEL_REVISION" <<'PY'
from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
import sys

repo_id = sys.argv[1]
revision = sys.argv[2]
info = HfApi().model_info(repo_id, revision=revision)
if info.sha != revision:
    raise RuntimeError(f"Hugging Face revision mismatch: expected={revision} actual={info.sha}")
path = snapshot_download(repo_id=repo_id, revision=revision)
print(Path(path).resolve())
PY
    )"
    [[ -d "$BASE_MODEL_PATH" ]] || die "pinned base-model snapshot missing: $BASE_MODEL_PATH"
    assert_path_on_production_storage "base model" "$BASE_MODEL_PATH"
    log "Hugging Face model=$BASE_MODEL revision=$BASE_MODEL_REVISION path=$BASE_MODEL_PATH"
}

preflight() {
    assert_known_host
    check_storage
    check_gpu_and_system_runtime
    check_groot_runtime
    check_pipeline_contract
    check_huggingface_access
    log "preflight PASS"
}

forward_batch() {
    local action="$1"
    shift

    local -a forwarded=()
    local separator_removed=0
    local argument
    for argument in "$@"; do
        if [[ "$argument" == "--" && "$separator_removed" -eq 0 ]]; then
            separator_removed=1
            continue
        fi
        forwarded+=("$argument")
    done

    # Every AWS batch action must resolve all mutable paths onto the large
    # volume. Formal stats/smoke/train additionally run the full GPU/model/data
    # preflight; the lighter actions still cannot create data on the root disk.
    if [[ "$action" == "stats" || "$action" == "smoke" || "$action" == "train" ]]; then
        preflight
    else
        assert_outputs_link_ready
    fi

    [[ -x "$GROOT_PYTHON" ]] || die "run bootstrap first; missing $GROOT_PYTHON"
    [[ -f "$BATCH_SCRIPT" ]] || die "batch runner missing: $BATCH_SCRIPT"
    local -a model_args=()
    if [[ "$action" == "smoke" || "$action" == "train" ]]; then
        model_args=(--base-model-path "$BASE_MODEL_PATH")
    fi
    exec env -u PYTHONOPTIMIZE -u PYTHONPATH -u PYTHONHOME \
        PYTHONNOUSERSITE=1 "$GROOT_PYTHON" "$BATCH_SCRIPT" "$action" \
        "${forwarded[@]}" \
        "${model_args[@]}" \
        --manifest "$MANIFEST" \
        --data-python "$AWS_DATA_PYTHON" \
        --groot-root "$GROOT_ROOT" \
        --groot-python "$GROOT_PYTHON" \
        --output-root "$TRAIN_OUTPUT_ROOT"
}

main() {
    local command="${1:-}"
    case "$command" in
        -h|--help|help|"")
            usage
            return
            ;;
    esac

    sanitize_python_runtime_env
    validate_project_layout
    case "$command" in
        paths)
            [[ "$#" -eq 1 ]] || die "paths takes no positional arguments"
            show_paths
            ;;
        storage-link)
            [[ "$#" -eq 1 ]] || die "storage-link takes no positional arguments"
            storage_link
            ;;
        bootstrap)
            [[ "$#" -eq 1 ]] || die "bootstrap takes no positional arguments"
            bootstrap
            ;;
        auth)
            [[ "$#" -eq 1 ]] || die "auth takes no positional arguments"
            huggingface_auth
            ;;
        preflight)
            [[ "$#" -eq 1 ]] || die "preflight takes no positional arguments"
            preflight
            ;;
        audit|prepare|verify|stats|dry-run|smoke|train)
            shift
            forward_batch "$command" "$@"
            ;;
        *)
            usage >&2
            die "unknown command: $command"
            ;;
    esac
}

main "$@"
