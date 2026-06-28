#!/bin/bash
# ============================================================
#  Turbulence Realm — SINDy  Linux Installer Builder
#
#  Uses Makeself to create a self-extracting .run installer
#  from the PyInstaller build output.
#
#  Usage:
#    ./scripts/make_linux_installer.sh
#
#  Prerequisites:
#    - PyInstaller build in build/dist/TurbulenceRealmSINDy/
#    - makeself installed (apt install makeself)
# ============================================================

set -e

APP_NAME="Turbulence Realm SINDy"
APP_NAME_SHORT="TurbulenceRealmSINDy"
VERSION="2.2.0"
BUILD_DIR="build/dist/TurbulenceRealmSINDy"
STAGING_DIR="build/linux-staging"
OUTPUT_DIR="installer"
INSTALLER_NAME="${APP_NAME_SHORT}-${VERSION}-Linux-Installer.run"

# Check makeself
if ! command -v makeself &>/dev/null; then
    echo "Error: makeself is not installed."
    echo "  Install with: sudo apt install makeself"
    exit 1
fi

# Check build output
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: PyInstaller build not found at ${BUILD_DIR}"
    echo "  Run 'pyinstaller TurbulenceRealmSINDy.spec' first."
    exit 1
fi

echo "==> Preparing staging directory..."
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

# Copy the PyInstaller output
cp -r "$BUILD_DIR"/* "$STAGING_DIR"/

# Copy the install script into the staging area
cp linux/install.sh "$STAGING_DIR"/install.sh
chmod +x "$STAGING_DIR"/install.sh

# Copy the disclaimer
cp DISCLAIMER.txt "$STAGING_DIR"/DISCLAIMER.txt

# Ensure the logo is present for the desktop icon
cp logo.png "$STAGING_DIR"/logo.png 2>/dev/null || true

echo "==> Creating self-extracting installer with Makeself..."
mkdir -p "$OUTPUT_DIR"

makeself \
    --notemp \
    --current \
    --tar-quietly \
    "$STAGING_DIR" \
    "${OUTPUT_DIR}/${INSTALLER_NAME}" \
    "${APP_NAME} v${VERSION} Installer" \
    ./install.sh

echo ""
echo "==> Installer created: ${OUTPUT_DIR}/${INSTALLER_NAME}"
echo "    Size: $(du -h "${OUTPUT_DIR}/${INSTALLER_NAME}" | cut -f1)"
echo ""
echo "    To install:  ./${OUTPUT_DIR}/${INSTALLER_NAME}"
echo "    To uninstall: /opt/${APP_NAME_SHORT}/install.sh --uninstall"
