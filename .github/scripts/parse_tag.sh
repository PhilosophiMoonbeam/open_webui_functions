#!/bin/bash

set -euo pipefail

TAG="${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"
TAG_PATTERN='^[a-z0-9][a-z0-9_-]*/v[0-9]+\.[0-9]+\.[0-9]+((a|b|rc)[0-9]+)?$'
if [[ ! "${TAG}" =~ ${TAG_PATTERN} ]]; then
  echo "::error::Malformed release tag '${TAG}'; expected <name>/v<semver>."
  exit 1
fi

PLUGIN_NAME="${TAG%/*}"
VERSION="${TAG##*/}"
VERSION_NO_V="${VERSION#v}"
IS_PRERELEASE="false"
if [[ "${VERSION_NO_V}" =~ (a|b|rc)[0-9]+$ ]]; then
  IS_PRERELEASE="true"
fi

echo "Parsed tag: plugin=${PLUGIN_NAME}, version=${VERSION}, prerelease=${IS_PRERELEASE}"
echo "plugin_name=${PLUGIN_NAME}" >> "${GITHUB_OUTPUT:?GITHUB_OUTPUT is required}"
echo "version=${VERSION}" >> "${GITHUB_OUTPUT}"
echo "is_prerelease=${IS_PRERELEASE}" >> "${GITHUB_OUTPUT}"
