#!/bin/bash
# ============================================================
#  Turbulence Realm — SINDy  Linux Post-Install Script
#  Runs after the Makeself archive extracts its contents.
#  Creates desktop/menu shortcuts and optional desktop icon.
# ============================================================

set -e

APP_NAME="Turbulence Realm SINDy"
APP_NAME_SHORT="TurbulenceRealmSINDy"
EXEC_NAME="TurbulenceRealmSINDy"
ICON_NAME="turbulencerealm-sindy"
CATEGORY="Science;Physics;DataVisualization;"

# The extracted app directory (where makeself extracted to)
APP_DIR="$(cd "$(dirname "$0")" && pwd)"

# Default install location
DEFAULT_PREFIX="/opt/${APP_NAME_SHORT}"

# ------------------------------------------------------------------
# Show disclaimer
# ------------------------------------------------------------------
show_disclaimer() {
    cat <<'DISCLAIMER'
========================================================================
  TURBULENCE REALM — SINDy  —  DISCLAIMER OF LIABILITY
========================================================================

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.

The user assumes all responsibility for the use of this software.

========================================================================
DISCLAIMER

    if [ -t 0 ]; then
        read -p "Do you accept this disclaimer? [y/N] " accept
        case "$accept" in
            y|Y|yes|YES) ;;
            *) echo "Installation cancelled."; exit 1 ;;
        esac
    fi
}

# ------------------------------------------------------------------
# Ask a yes/no question
# ------------------------------------------------------------------
ask_yes_no() {
    local prompt="$1"
    local default="$2"
    if [ "$default" = "y" ]; then
        read -p "$prompt [Y/n] " answer
        case "$answer" in
            n|N|no|NO) return 1 ;;
            *) return 0 ;;
        esac
    else
        read -p "$prompt [y/N] " answer
        case "$answer" in
            y|Y|yes|YES) return 0 ;;
            *) return 1 ;;
        esac
    fi
}

# ------------------------------------------------------------------
# Create a .desktop file for application menu integration
# ------------------------------------------------------------------
create_desktop_entry() {
    local prefix="$1"
    local desktop_dir="${HOME}/.local/share/applications"
    local icon_dir="${HOME}/.local/share/icons/hicolor/256x256/apps"
    local desktop_file="${desktop_dir}/${APP_NAME_SHORT}.desktop"

    mkdir -p "$desktop_dir" "$icon_dir"

    # Copy icon
    if [ -f "${prefix}/logo.png" ]; then
        cp "${prefix}/logo.png" "${icon_dir}/${ICON_NAME}.png"
    fi

    cat > "$desktop_file" <<DESKTOP
[Desktop Entry]
Type=Application
Name=${APP_NAME}
Comment=Video-based fluid-flow analysis with SINDy
Exec=${prefix}/${EXEC_NAME}
Icon=${ICON_NAME}
Terminal=false
Categories=${CATEGORY}
DESKTOP

    chmod +x "$desktop_file"
    echo "  Created: ${desktop_file}"
}

# ------------------------------------------------------------------
# Create desktop shortcut
# ------------------------------------------------------------------
create_desktop_shortcut() {
    local prefix="$1"
    local desktop_dir="${HOME}/Desktop"
    local desktop_file="${desktop_dir}/${APP_NAME_SHORT}.desktop"

    # Handle locales where Desktop folder has a different name
    if [ -f "${HOME}/.config/user-dirs.dirs" ]; then
        source "${HOME}/.config/user-dirs.dirs"
        if [ -n "$XDG_DESKTOP_DIR" ]; then
            desktop_dir=$(eval echo "$XDG_DESKTOP_DIR")
        fi
    fi

    mkdir -p "$desktop_dir"

    cat > "$desktop_file" <<DESKTOP
[Desktop Entry]
Type=Application
Name=${APP_NAME}
Comment=Video-based fluid-flow analysis with SINDy
Exec=${prefix}/${EXEC_NAME}
Icon=${ICON_NAME}
Terminal=false
Categories=${CATEGORY}
DESKTOP

    chmod +x "$desktop_file"
    echo "  Created: ${desktop_file}"
}

# ------------------------------------------------------------------
# Uninstall function
# ------------------------------------------------------------------
uninstall() {
    echo "Uninstalling ${APP_NAME}..."

    local prefix="${DEFAULT_PREFIX}"
    local desktop_file="${HOME}/.local/share/applications/${APP_NAME_SHORT}.desktop"
    local icon_file="${HOME}/.local/share/icons/hicolor/256x256/apps/${ICON_NAME}.png"
    local desktop_shortcut="${HOME}/Desktop/${APP_NAME_SHORT}.desktop"

    # Remove installed files
    if [ -d "$prefix" ]; then
        sudo rm -rf "$prefix"
        echo "  Removed: ${prefix}"
    fi

    # Remove desktop entry
    rm -f "$desktop_file" 2>/dev/null && echo "  Removed: ${desktop_file}"
    rm -f "$icon_file" 2>/dev/null && echo "  Removed: ${icon_file}"
    rm -f "$desktop_shortcut" 2>/dev/null && echo "  Removed: ${desktop_shortcut}"

    # Update desktop database
    update-desktop-database "${HOME}/.local/share/applications" 2>/dev/null || true
    gtk-update-icon-cache -f 2>/dev/null || true

    echo "Uninstall complete."
    exit 0
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
main() {
    # Handle --uninstall flag
    if [ "$1" = "--uninstall" ]; then
        uninstall
    fi

    echo ""
    echo "  ============================================"
    echo "    ${APP_NAME}  v2.2.0  —  Linux Installer"
    echo "  ============================================"
    echo ""

    # Show disclaimer
    show_disclaimer

    echo ""

    # Ask for install location
    echo "Install directory: ${DEFAULT_PREFIX}"
    if ask_yes_no "  Use this location?" "y"; then
        PREFIX="$DEFAULT_PREFIX"
    else
        read -p "  Enter install directory: " PREFIX
        if [ -z "$PREFIX" ]; then
            echo "  No directory specified. Using default: ${DEFAULT_PREFIX}"
            PREFIX="$DEFAULT_PREFIX"
        fi
    fi
    echo ""

    # Install files
    echo "Installing to ${PREFIX}..."
    sudo mkdir -p "$PREFIX"
    sudo cp -r "$APP_DIR"/* "$PREFIX"/
    sudo chmod +x "${PREFIX}/${EXEC_NAME}" 2>/dev/null || true
    echo "  Files copied."
    echo ""

    # Create application menu entry
    echo "Creating application menu entry..."
    create_desktop_entry "$PREFIX"
    update-desktop-database "${HOME}/.local/share/applications" 2>/dev/null || true
    echo ""

    # Ask for desktop shortcut
    if ask_yes_no "  Create a desktop shortcut?" "n"; then
        create_desktop_shortcut "$PREFIX"
    fi
    echo ""

    # Done
    echo "  ============================================"
    echo "    Installation complete!"
    echo "  ============================================"
    echo ""
    echo "  You can now launch ${APP_NAME} from:"
    echo "    - Application menu (search for 'Turbulence Realm')"
    echo "    - Command line: ${PREFIX}/${EXEC_NAME}"
    echo ""
    echo "  To uninstall, run:"
    echo "    ${PREFIX}/install.sh --uninstall"
    echo ""

    # Ask to launch
    if ask_yes_no "  Launch ${APP_NAME} now?" "n"; then
        "${PREFIX}/${EXEC_NAME}" &
    fi
}

main "$@"
