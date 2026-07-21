#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor"

declare -A URLS=(
  [PDFK]="https://github.com/colaudiolab/PDFK.git"
  [budgeted-cl]="https://github.com/snumprlab/budgeted-cl.git"
  [OnlineDeepLearning]="https://github.com/phquang/OnlineDeepLearning.git"
  [GLOBEM]="https://github.com/UW-EXP/GLOBEM.git"
  [OLI2DS]="https://github.com/youdianlong/OLI2DS.git"
  [KOIL]="https://github.com/JunjieHu/koil.git"
)

declare -A COMMITS=(
  [PDFK]="ff4f58a320eeab35e76bb75999c75c6a862b967d"
  [budgeted-cl]="fb4d79cd928d5f00b013d3a2c525a5a1e374b337"
  [OnlineDeepLearning]="90456299654b742b45374c83f8923faf895db13c"
  [GLOBEM]="4f140fc5290dc97204298cca28b956165aa0a29f"
  [OLI2DS]="119e7c1764d2c4f592f7582e0857a351a2819dfe"
  [KOIL]="5b181412795a2c9777e9d54eef9f94233c01d9c0"
)

ALL=(PDFK budgeted-cl OnlineDeepLearning GLOBEM OLI2DS KOIL)
if (( $# > 0 )); then
  SELECTED=("$@")
else
  SELECTED=("${ALL[@]}")
fi

mkdir -p "${VENDOR_DIR}"

for name in "${SELECTED[@]}"; do
  if [[ -z "${URLS[$name]+x}" ]]; then
    echo "Unknown repository: ${name}" >&2
    echo "Available: ${ALL[*]}" >&2
    exit 2
  fi

  destination="${VENDOR_DIR}/${name}"
  if [[ -e "${destination}" && ! -d "${destination}/.git" ]]; then
    echo "Refusing to replace non-Git path: ${destination}" >&2
    exit 3
  fi

  if [[ ! -d "${destination}/.git" ]]; then
    git clone --filter=blob:none --no-checkout "${URLS[$name]}" "${destination}"
  fi

  git -C "${destination}" fetch --filter=blob:none origin "${COMMITS[$name]}"
  git -C "${destination}" checkout --detach "${COMMITS[$name]}"
  echo "${name}: $(git -C "${destination}" rev-parse HEAD)"
done
