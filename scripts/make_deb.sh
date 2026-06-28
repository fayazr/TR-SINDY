#!/bin/bash
# ============================================================
#  Turbulence Realm — SINDy  .deb Package Builder
#
#  Creates a .deb package from the PyInstaller build output
#  for Debian/Ubuntu-based distributions.
#
#  Usage:
#    ./scripts/make_deb.sh
#
#  Prerequisites:
#    - PyInstaller build in build/dist/TurbulenceRealmSINDy/
#    - dpkg-deb installed (standard on Debian/Ubuntu)
# ============================================================

set -e

APP_NAME_SHORT="TurbulenceRealmSINDy"
VERSION="2.2.0"
BUILD_DIR="build/dist/TurbulenceRealmSINDy"
DEB_ROOT="build/deb-root"
OUTPUT_DIR="installer"
DEB_NAME="${APP_NAME_SHORT}-${VERSION}-amd64.deb"

# Check build output
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: PyInstaller build not found at ${BUILD_DIR}"
    echo "  Run 'pyinstaller TurbulenceRealmSINDy.spec' first."
    exit 1
fi

echo "==> Preparing .deb directory structure..."
rm -rf "$DEB_ROOT"
mkdir -p "$DEB_ROOT/DEBIAN"
mkdir -p "$DEB_ROOT/opt/${APP_NAME_SHORT}"
mkdir -p "$DEB_ROOT/usr/share/applications"
mkdir -p "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps"

# Copy control file
cp linux/deb/DEBIAN/control "$DEB_ROOT/DEBIAN/control"

# Copy PyInstaller output to /opt/TurbulenceRealmSINDy/
cp -r "$BUILD_DIR"/* "$DEB_ROOT/opt/${APP_NAME_SHORT}"/
chmod +x "$DEB_ROOT/opt/${APP_NAME_SHORT}/${APP_NAME_SHORT}" 2>/dev/null || true

# Copy .desktop file
cp linux/deb/usr/share/applications/turbulencerealm-sindy.desktop \
   "$DEB_ROOT/usr/share/applications/"

# Copy icon
cp logo.png "$DEB_ROOT/usr/share/icons/hicolor/256x256/apps/turbulencerealm-sindy.png"

# Set permissions
find "$DEB_ROOT" -type d -exec chmod 755 {} \;
chmod 755 "$DEB_ROOT/DEBIAN/control"

echo "==> Building .deb package..."
mkdir -p "$OUTPUT_DIR"
dpkg-deb --build --root-owner-group "$DEB_ROOT" "${OUTPUT_DIR}/${DEB_NAME}"

echo ""
echo "==> .deb package created: ${OUTPUT_DIR}/${DEB_NAME}"
echo "    Size: $(du -h "${OUTPUT_DIR}/${DEB_NAME}" | cut -f1)"
echo ""
echo "    To install:  sudo dpkg -i ${OUTPUT_DIR}/${DEB_NAME}"
echo "    To uninstall: sudo dpkg -r ${APP_NAME_SHORT}"
