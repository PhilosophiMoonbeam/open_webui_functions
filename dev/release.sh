#!/bin/bash

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ROOT=$(git rev-parse --show-toplevel)
cd "${ROOT}"

GEMINI_MANIFEST="release/gemini-suite-v3.0.0.yaml"
GEMINI_TAG="gemini-suite/v3.0.0"
GEMINI_MEMBERS=(
  gemini_manifold
  gemini_manifold_companion
  gemini_reasoning_toggle
  gemini_map_grounding_toggle
  gemini_url_context_toggle
  gemini_paid_api
  gemini_enterprise_toggle
)

die() {
  echo -e "${RED}Error: $*${NC}" >&2
  exit 1
}

confirm_tag() {
  local tag=$1
  git rev-parse "refs/tags/${tag}" >/dev/null 2>&1 && die "Tag '${tag}' already exists locally."
  echo -e "${BLUE}Ready to create and push:${NC} ${YELLOW}${tag}${NC}"
  read -r -p "Continue? (y/N) " reply
  [[ "${reply}" =~ ^[Yy]$ ]] || die "Aborted."
  git tag "${tag}"
  git push origin "${tag}"
  echo -e "${GREEN}Pushed ${tag}; verify the draft release before publication.${NC}"
}

assert_clean_and_current() {
  git diff --quiet || die "Working tree has unstaged changes."
  git diff --cached --quiet || die "Index has staged changes."
  [[ -z "$(git ls-files --others --exclude-standard)" ]] || die "Working tree has untracked files."
  local upstream
  upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null) ||
    die "Current branch has no upstream."
  git fetch --quiet "${upstream%%/*}"
  [[ "$(git rev-parse HEAD)" == "$(git rev-parse '@{upstream}')" ]] ||
    die "Current branch is not exactly up to date with ${upstream}."
}

release_gemini_suite() {
  assert_clean_and_current
  make check
  mkdir -p dist/gemini-suite-release-check
  uv run python .github/scripts/build_gemini_suite.py \
    --manifest "${GEMINI_MANIFEST}" --tag "${GEMINI_TAG}" \
    --output dist/gemini-suite-v3.0.0.tar.gz
  uv run python .github/scripts/build_gemini_suite.py \
    --manifest "${GEMINI_MANIFEST}" --tag "${GEMINI_TAG}" \
    --output dist/gemini-suite-release-check/gemini-suite-v3.0.0.tar.gz
  cmp dist/gemini-suite-v3.0.0.tar.gz \
    dist/gemini-suite-release-check/gemini-suite-v3.0.0.tar.gz
  cp "${GEMINI_MANIFEST}" dist/gemini-suite-v3.0.0.yaml
  (cd dist && sha256sum gemini-suite-v3.0.0.tar.gz gemini-suite-v3.0.0.yaml > SHA256SUMS)
  confirm_tag "${GEMINI_TAG}"
}

if [[ "${1:-}" == "gemini-suite" ]]; then
  release_gemini_suite
  exit 0
fi
[[ $# -eq 0 ]] || die "Usage: dev/release.sh [gemini-suite]"

mapfile -t PLUGIN_FILES < <(
  find plugins/filters plugins/pipes -path '*/__pycache__/*' -prune -o \
    -name '*.py' ! -name '__init__.py' -print | sort
)
SAFE_PLUGIN_FILES=()
for file in "${PLUGIN_FILES[@]}"; do
  name=$(basename "${file}" .py)
  is_gemini_member=false
  for member in "${GEMINI_MEMBERS[@]}"; do
    [[ "${name}" == "${member}" ]] && is_gemini_member=true
  done
  [[ "${is_gemini_member}" == false ]] && SAFE_PLUGIN_FILES+=("${file}")
done
[[ ${#SAFE_PLUGIN_FILES[@]} -gt 0 ]] || die "No standalone plugins found."

echo -e "${BLUE}Gemini suite members are excluded; use 'dev/release.sh gemini-suite'.${NC}"
PS3=$'\nSelect a standalone plugin to release: '
select selected_path in "${SAFE_PLUGIN_FILES[@]}"; do
  [[ -n "${selected_path}" ]] && break
  echo "Invalid selection."
done

plugin_name=$(basename "${selected_path}" .py)
version=$(sed -nE 's/^version:[[:space:]]*([^[:space:]]+).*$/\1/p' "${selected_path}" | head -1)
[[ "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$ ]] ||
  die "Invalid or missing frontmatter version in ${selected_path}."
confirm_tag "${plugin_name}/v${version}"
