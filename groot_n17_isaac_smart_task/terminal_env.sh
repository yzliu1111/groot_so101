#!/usr/bin/env bash

# Source this file at the start of a new terminal.  It exports project paths
# only; each README still activates the LeRobot, GR00T, or LeIsaac runtime that
# belongs to the current step.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run with: source ${BASH_SOURCE[0]} <local|target>" >&2
    exit 2
fi

profile="${1:-}"
case "$profile" in
    local)
        groot_root="$HOME/Isaac-GR00T-py312"
        ;;
    target)
        groot_root="$HOME/Isaac-GR00T"
        ;;
    *)
        echo "Usage: source ${BASH_SOURCE[0]} <local|target>" >&2
        return 2
        ;;
esac

experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
smart_project="$(cd -- "$experiment_root/../.." && pwd)"

export SMART_PROJECT="$smart_project"
export EXPERIMENT_ROOT="$experiment_root"
export GROOT_ROOT="$groot_root"
export LEISAAC_ROOT="$SMART_PROJECT/leisaac"
export LEISAAC_ASSETS_ROOT="$LEISAAC_ROOT/assets"
export LEISAAC_ENV="${LEISAAC_ENV:-leisaac}"
export LEROBOT_ENV="${LEROBOT_ENV:-lerobot}"
export OMNI_KIT_ACCEPT_EULA=YES

printf '[terminal-env] profile=%s\n' "$profile"
printf '[terminal-env] SMART_PROJECT=%s\n' "$SMART_PROJECT"
printf '[terminal-env] GROOT_ROOT=%s\n' "$GROOT_ROOT"
printf '[terminal-env] LEISAAC_ENV=%s LEROBOT_ENV=%s\n' "$LEISAAC_ENV" "$LEROBOT_ENV"
