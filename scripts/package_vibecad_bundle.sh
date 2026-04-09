#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${REPO_ROOT}/vibecad"
DEFAULT_OUTPUT="${REPO_ROOT}/dist/vibecad-shareable.zip"
OUTPUT_ZIP="${1:-${DEFAULT_OUTPUT}}"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Error: source directory not found: ${SRC_DIR}" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync is required but was not found on PATH." >&2
  exit 1
fi

if ! command -v zip >/dev/null 2>&1; then
  echo "Error: zip is required but was not found on PATH." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${TMP_DIR}/vibecad"
rsync -a --delete \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  --exclude "*.swp" \
  --exclude "*.swo" \
  --exclude "*.swx" \
  --exclude ".*.swp" \
  --exclude ".*.swo" \
  --exclude ".*.swx" \
  --exclude ".DS_Store" \
  --exclude "debug/" \
  "${SRC_DIR}/" "${TMP_DIR}/vibecad/"

mkdir -p "$(dirname "${OUTPUT_ZIP}")"
rm -f "${OUTPUT_ZIP}"
(
  cd "${TMP_DIR}"
  zip -qr "${OUTPUT_ZIP}" vibecad
)

echo "Created shareable bundle:"
echo "  ${OUTPUT_ZIP}"
echo
echo "Inside the zip, the top-level folder is: vibecad/"
