#!/usr/bin/env bash
# ============================================================================
# Install ICICLE (https://github.com/ingonyama-zk/icicle) so we can link
# against the BabyBear field + CUDA backend in tests/icicle_compare.cu.
#
# Strategy: download the official pre-built Linux release. The CUDA backend
# itself lives in a private repo, so a from-source build needs SSH access
# you almost certainly don't have. The release tarball ships:
#
#   include/icicle/...                      <- headers
#   lib/libicicle_device.so                 <- runtime / device API
#   lib/libicicle_field_babybear.so         <- BabyBear scalar + frontend NTT
#   lib/backend/cuda/libicicle_backend_cuda_*.so  <- prebuilt CUDA backend
#
# ICICLE BabyBear uses prime q = 15 * 2^27 + 1 = 2013265921, the *same*
# DEFAULT_PRIME we use, so a head-to-head NTT comparison is meaningful.
#
# Usage:
#   ./scripts/build_icicle.sh                      # default install dir
#   ICICLE_INSTALL_DIR=/path ./scripts/build_icicle.sh
#   ICICLE_VERSION=4.0.0 ./scripts/build_icicle.sh
# ============================================================================
set -euo pipefail

ICICLE_INSTALL_DIR="${ICICLE_INSTALL_DIR:-$HOME/.local/icicle}"
ICICLE_VERSION="${ICICLE_VERSION:-4.0.0}"
# Two tarballs are needed: frontend (headers + libicicle_*.so) and the CUDA
# backend plugins (libicicle_backend_cuda_*.so under lib/backend/).
ICICLE_FE_FLAVOUR="${ICICLE_FE_FLAVOUR:-ubuntu22}"
ICICLE_BE_FLAVOUR="${ICICLE_BE_FLAVOUR:-ubuntu22-cuda122}"

ICICLE_VER_TAG="$(echo "$ICICLE_VERSION" | tr '.' '_')"
FE_TARBALL="icicle_${ICICLE_VER_TAG}-${ICICLE_FE_FLAVOUR}.tar.gz"
BE_TARBALL="icicle_${ICICLE_VER_TAG}-${ICICLE_BE_FLAVOUR}.tar.gz"
BASE_URL="https://github.com/ingonyama-zk/icicle/releases/download/v${ICICLE_VERSION}"

echo "==> ICICLE version    : $ICICLE_VERSION"
echo "==> Frontend tarball  : $FE_TARBALL"
echo "==> CUDA backend tar  : $BE_TARBALL"
echo "==> Install dir       : $ICICLE_INSTALL_DIR"

WORKDIR="$(mktemp -d -t icicle.XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT
cd "$WORKDIR"

echo "==> Downloading frontend (~16 MB) ..."
curl -L --fail --progress-bar -o "$FE_TARBALL" "$BASE_URL/$FE_TARBALL"
echo "==> Downloading CUDA backend (~370 MB) ..."
curl -L --fail --progress-bar -o "$BE_TARBALL" "$BASE_URL/$BE_TARBALL"

echo "==> Extracting frontend ..."
mkdir -p staged
tar -xzf "$FE_TARBALL" -C staged
echo "==> Extracting CUDA backend (overlays lib/backend/) ..."
tar -xzf "$BE_TARBALL" -C staged

if [ ! -d staged/icicle/include ] || [ ! -d staged/icicle/lib ]; then
    echo "ERROR: unexpected archive layout under staged/icicle/:" >&2
    find staged -maxdepth 3 -type d >&2
    exit 1
fi

SRC=staged/icicle

echo "==> Installing into $ICICLE_INSTALL_DIR ..."
mkdir -p "$ICICLE_INSTALL_DIR"
cp -a "$SRC/include" "$ICICLE_INSTALL_DIR/"
cp -a "$SRC/lib"     "$ICICLE_INSTALL_DIR/"

echo
echo "==> Installed:"
find "$ICICLE_INSTALL_DIR/lib" -maxdepth 3 -name 'libicicle*' | sort
echo
echo "==> Header sanity:"
ls "$ICICLE_INSTALL_DIR/include/icicle/ntt.h" \
   "$ICICLE_INSTALL_DIR/include/icicle/runtime.h" \
   "$ICICLE_INSTALL_DIR/include/icicle/fields/field_config.h" 2>/dev/null || true

cat <<EOF

==> Next steps:

  cd build
  cmake -DICICLE_ENABLED=ON \\
        -DICICLE_INSTALL_DIR=$ICICLE_INSTALL_DIR \\
        ..
  make -j icicle_compare

  ICICLE_BACKEND_INSTALL_DIR=$ICICLE_INSTALL_DIR/lib/backend ./icicle_compare

The ICICLE_BACKEND_INSTALL_DIR env var tells icicle_load_backend_from_env_or_default()
where to find libicicle_backend_cuda_*.so at runtime.
EOF
