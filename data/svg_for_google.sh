#!/usr/bin/env bash
set -euo pipefail

REPEATS="${REPEATS:-6}"       # times to repeat selection-ungroup
KEEP_TEXT="${KEEP_TEXT:-0}"   # 1 = keep text editable (no object-to-path)
KEEP_STROKE="${KEEP_STROKE:-0}" # 1 = keep strokes (no object-stroke-to-path)

# Temp clean profile to *disable* user extensions/config
PROFILE_DIR="$(mktemp -d)"
trap 'rm -rf "$PROFILE_DIR"' EXIT

# Minimal, clean environment (prevents Snap/Conda leaks) + keep GUI basics quiet
run_ink() {
  env -i \
    INKSCAPE_PROFILE_DIR="$PROFILE_DIR" \
    HOME="$HOME" USER="${USER:-$LOGNAME}" LOGNAME="${LOGNAME:-$USER}" \
    LANG="${LANG:-en_US.UTF-8}" LC_ALL="${LC_ALL:-}" \
    DISPLAY="${DISPLAY:-}" WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}" \
    XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" XDG_DATA_DIRS="${XDG_DATA_DIRS:-}" \
    PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    inkscape "$@"
}

build_actions() {
  local out="$1"
  local act="select-all:all;"
  for _ in $(seq 1 "$REPEATS"); do act+="selection-ungroup;"; done
  if [ "$KEEP_TEXT" != "1" ];   then act+="object-to-path;"; fi
  if [ "$KEEP_STROKE" != "1" ]; then act+="object-stroke-to-path;"; fi
  act+="export-filename:${out};export-do"
  printf '%s' "$act"
}

convert_one() {
  local in="$1"; local out="$2"
  # First try full pipeline
  if run_ink "$in" --actions "$(build_actions "$out")"; then
    echo "✓ $out"
    return 0
  fi
  # Fallback: no ungroup, no conversions — just export as EMF
  echo "⚠️  Fallback (no ungroup) for: $in"
  run_ink "$in" --actions "export-filename:${out};export-do"
  echo "✓ $out (fallback)"
}

# Process SVGs in current dir (safe with spaces)
while IFS= read -r -d '' f; do
  convert_one "$f" "${f%.svg}.emf"
done < <(find . -maxdepth 1 -type f -name '*.svg' -print0)
