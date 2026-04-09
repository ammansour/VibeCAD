#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/vibecad"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Error: source directory not found: ${SRC_DIR}" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync is required but was not found on PATH." >&2
  exit 1
fi

detect_kicad_plugin_dir() {
  if [[ -n "${KICAD_PLUGIN_DIR:-}" ]]; then
    printf "%s\n" "${KICAD_PLUGIN_DIR}"
    return 0
  fi

  local os_name
  os_name="$(uname -s | tr '[:upper:]' '[:lower:]')"

  if [[ "${os_name}" == "darwin" ]]; then
    local prefs_root="${HOME}/Library/Preferences/kicad"
    local app_support_root="${HOME}/Library/Application Support/kicad"

    for ver in 9.0 8.0 7.0 6.0 5.1; do
      if [[ -d "${prefs_root}/${ver}" ]]; then
        printf "%s\n" "${prefs_root}/${ver}/scripting/plugins"
        return 0
      fi
    done

    if [[ -d "${app_support_root}" ]]; then
      printf "%s\n" "${app_support_root}/scripting/plugins"
      return 0
    fi

    printf "%s\n" "${prefs_root}/9.0/scripting/plugins"
    return 0
  fi

  if [[ "${os_name}" == "linux" ]]; then
    local linux_root="${HOME}/.local/share/kicad"
    for ver in 9.0 8.0 7.0 6.0 5.1; do
      if [[ -d "${linux_root}/${ver}" ]]; then
        printf "%s\n" "${linux_root}/${ver}/scripting/plugins"
        return 0
      fi
    done
    printf "%s\n" "${linux_root}/9.0/scripting/plugins"
    return 0
  fi

  echo "Error: unsupported OS for automatic KiCad plugin path detection (${os_name})." >&2
  echo "Set KICAD_PLUGIN_DIR to the full plugins directory path." >&2
  exit 1
}

resolve_target_package_dir() {
  local plugin_dir="$1"
  if [[ -d "${plugin_dir}/VibeCAD/vibecad" ]]; then
    printf "%s\n" "${plugin_dir}/VibeCAD/vibecad"
    return 0
  fi
  if [[ -d "${plugin_dir}/vibecad" ]]; then
    printf "%s\n" "${plugin_dir}/vibecad"
    return 0
  fi
  if [[ -d "${plugin_dir}/VibeCAD" ]]; then
    printf "%s\n" "${plugin_dir}/VibeCAD/vibecad"
    return 0
  fi
  printf "%s\n" "${plugin_dir}/vibecad"
}

TARGET_PLUGIN_DIR="$(detect_kicad_plugin_dir)"
TARGET_PACKAGE_DIR="$(resolve_target_package_dir "${TARGET_PLUGIN_DIR}")"

mkdir -p "${TARGET_PACKAGE_DIR}"

rsync -a --delete \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude ".DS_Store" \
  "${SRC_DIR}/" "${TARGET_PACKAGE_DIR}/"

echo "VibeCAD deployed successfully:"
echo "  source: ${SRC_DIR}"
echo "  target: ${TARGET_PACKAGE_DIR}"
echo

echo "Tip: set KICAD_PLUGIN_DIR to override target location."
