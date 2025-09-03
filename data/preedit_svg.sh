#!/usr/bin/env bash
set -euo pipefail

REPEATS="${REPEATS:-6}"
BACKUP="${BACKUP:-1}"
RECURSE="${RECURSE:-0}"

run_ink() {
  env -i \
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

  # expand selectors for common white fills
  local -a sel=(
    "rect[fill='#ffffff']" "rect[fill='#fff']" "rect[fill='#FFFFFF']"
    "rect[fill='white']"
    "*[style*='fill:#ffffff']" "*[style*='fill:#fff']" "*[style*='fill:#FFFFFF']"
    "*[style*='fill:rgb(255,255,255)']" "*[style*='fill: rgb(255, 255, 255)']"
  )
  for s in "${sel[@]}"; do
    act+="select-clear;select-by-selector:${s};delete-selection;"
  done

  # always export to plain SVG and write to 'out'
  act+="export-plain-svg;export-filename:${out};export-do"
  printf '%s' "$act"
}

# gather files
declare -a files
if [[ "$RECURSE" == "1" ]]; then
  while IFS= read -r -d '' f; do files+=("$f"); done < <(find . -type f -name '*.svg' -print0)
else
  shopt -s nullglob
  files=( *.svg )
fi
[[ ${#files[@]} -gt 0 ]] || { echo "No .svg files found."; exit 0; }

echo "Processing ${#files[@]} SVG(s): ungroup ×${REPEATS}, remove white backgrounds, overwrite originals."
echo

for f in "${files[@]}"; do
  [[ "$BACKUP" == "1" && ! -e "${f}.bak" ]] && cp -p -- "$f" "${f}.bak" || true

  tmp="$(mktemp --suffix=.svg)"
  if run_ink "$f" --actions "$(build_actions "$tmp")" >/dev/null 2>&1; then
    mv -f -- "$tmp" "$f"
    echo "✓ saved: $f"
  else
    rm -f -- "$tmp"
    echo "❌ failed: $f"
  fi
done
