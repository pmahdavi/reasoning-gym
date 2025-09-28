#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   REGISTRY=ghcr.io OWNER_REPO=open-thought/reasoning-gym TAG=v0.1.23 \
#   ./scripts/build_full_image.sh
#
# Optional overrides:
#   BASE_IMAGE=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#   PLATFORMS=linux/amd64

REGISTRY="${REGISTRY:-ghcr.io}"
OWNER_REPO="${OWNER_REPO:?set OWNER_REPO, e.g. open-thought/reasoning-gym}"
TAG="${TAG:?set TAG, e.g. v0.1.23}"
BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04}"
PLATFORMS="${PLATFORMS:-linux/amd64}"

IMAGE="${REGISTRY}/${OWNER_REPO}:${TAG}"
FULL_IMAGE="${REGISTRY}/${OWNER_REPO}:full-${TAG}"

# Enable Buildx
if ! docker buildx version >/dev/null 2>&1; then
  echo "Docker Buildx is required" >&2
  exit 1
fi

echo "Building FULL image: ${IMAGE} (platforms=${PLATFORMS})"

docker buildx build \
  --push \
  --platform "${PLATFORMS}" \
  --build-arg "INCLUDE_TRAINING_STACK=1" \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --provenance=false \
  -f Dockerfile \
  -t "${IMAGE}" \
  -t "${FULL_IMAGE}" \
  .

echo "Pushed: ${IMAGE} and ${FULL_IMAGE}" 