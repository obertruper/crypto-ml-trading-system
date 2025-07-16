#!/bin/bash

# Cursor IDE Installation Script for Linux
# Compatible with Ubuntu 22.04, 24.04 and other Linux distributions
# Author: LLM Transform Project
# Date: 2025-07-16

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables
CURSOR_DOWNLOAD_URL="https://www.cursor.com/api/download?platform=linux-x64&releaseTrack=stable"
INSTALL_DIR="$HOME/Applications"
CURSOR_APPIMAGE="$INSTALL_DIR/cursor.AppImage"
DESKTOP_FILE="$HOME/.local/share/applications/cursor.desktop"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to detect Ubuntu version
detect_ubuntu_version() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo "$VERSION_ID"
    else
        echo "unknown"
    fi
}

# Function to download Cursor
download_cursor() {
    print_info "Downloading Cursor IDE..."
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd "$TEMP_DIR"
    
    # Download with Firefox user agent to avoid 403 errors
    wget --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/115.0" \
         -O cursor.AppImage \
         "$CURSOR_DOWNLOAD_URL" \
         --show-progress
    
    if [ $? -eq 0 ]; then
        print_success "Download completed successfully"
        return 0
    else
        print_error "Failed to download Cursor"
        return 1
    fi
}

# Function to install dependencies
install_dependencies() {
    local ubuntu_version=$(detect_ubuntu_version)
    
    if [[ "$ubuntu_version" < "24.04" ]]; then
        print_info "Installing libfuse2 for Ubuntu < 24.04..."
        sudo apt update
        sudo apt install -y libfuse2
    else
        print_warning "Ubuntu 24.04 detected. Skipping libfuse2 installation."
        print_info "If you encounter issues, try the extraction method instead."
    fi
}

# Function to setup Cursor
setup_cursor() {
    print_info "Setting up Cursor IDE..."
    
    # Create Applications directory if it doesn't exist
    mkdir -p "$INSTALL_DIR"
    
    # Move AppImage to installation directory
    mv cursor.AppImage "$CURSOR_APPIMAGE"
    
    # Make executable
    chmod +x "$CURSOR_APPIMAGE"
    
    print_success "Cursor AppImage installed to: $CURSOR_APPIMAGE"
}

# Function to create desktop entry
create_desktop_entry() {
    print_info "Creating desktop entry..."
    
    # Create local applications directory if it doesn't exist
    mkdir -p "$HOME/.local/share/applications"
    
    # Download Cursor icon
    wget -q -O "$INSTALL_DIR/cursor.png" \
         "https://raw.githubusercontent.com/getcursor/cursor/main/resources/app/resources/linux/code.png" \
         2>/dev/null || true
    
    # Create desktop entry
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Name=Cursor
Comment=The AI-powered code editor
Exec=$CURSOR_APPIMAGE --no-sandbox %F
Icon=$INSTALL_DIR/cursor.png
Type=Application
Categories=Development;IDE;TextEditor;
StartupNotify=true
MimeType=text/plain;text/x-source;
EOF
    
    # Make desktop entry executable
    chmod +x "$DESKTOP_FILE"
    
    # Update desktop database
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
    
    print_success "Desktop entry created"
}

# Function to add shell alias
add_shell_alias() {
    print_info "Adding shell alias..."
    
    # Detect shell
    SHELL_RC=""
    if [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
    elif [ -n "$BASH_VERSION" ]; then
        SHELL_RC="$HOME/.bashrc"
    fi
    
    if [ -n "$SHELL_RC" ]; then
        # Check if alias already exists
        if ! grep -q "alias cursor=" "$SHELL_RC" 2>/dev/null; then
            echo "" >> "$SHELL_RC"
            echo "# Cursor IDE alias" >> "$SHELL_RC"
            echo "alias cursor='$CURSOR_APPIMAGE --no-sandbox'" >> "$SHELL_RC"
            print_success "Added cursor alias to $SHELL_RC"
            print_info "Run 'source $SHELL_RC' or restart your terminal to use the alias"
        else
            print_info "Cursor alias already exists in $SHELL_RC"
        fi
    fi
}

# Function to extract AppImage (alternative for Ubuntu 24.04)
extract_appimage() {
    print_info "Extracting AppImage for Ubuntu 24.04 compatibility..."
    
    cd "$INSTALL_DIR"
    
    # Extract AppImage
    "$CURSOR_APPIMAGE" --appimage-extract
    
    if [ -d "squashfs-root" ]; then
        # Rename extracted directory
        mv squashfs-root cursor-extracted
        
        # Create wrapper script
        cat > "$INSTALL_DIR/cursor" << EOF
#!/bin/bash
exec "$INSTALL_DIR/cursor-extracted/cursor" "\$@"
EOF
        
        chmod +x "$INSTALL_DIR/cursor"
        
        # Update desktop entry to use extracted version
        sed -i "s|Exec=.*|Exec=$INSTALL_DIR/cursor %F|" "$DESKTOP_FILE"
        
        print_success "AppImage extracted successfully"
        print_info "You can now run Cursor using: $INSTALL_DIR/cursor"
        
        return 0
    else
        print_error "Failed to extract AppImage"
        return 1
    fi
}

# Function to test Cursor installation
test_cursor() {
    print_info "Testing Cursor installation..."
    
    if [ -f "$CURSOR_APPIMAGE" ]; then
        # Try to run Cursor with version flag
        if "$CURSOR_APPIMAGE" --version &>/dev/null; then
            print_success "Cursor is working correctly"
            return 0
        else
            print_warning "Cursor AppImage exists but may not be working properly"
            return 1
        fi
    else
        print_error "Cursor AppImage not found"
        return 1
    fi
}

# Main installation function
main() {
    echo -e "${GREEN}================================${NC}"
    echo -e "${GREEN}  Cursor IDE Installation Script${NC}"
    echo -e "${GREEN}================================${NC}"
    echo
    
    # Check if running on Linux
    if [[ "$OSTYPE" != "linux-gnu"* ]]; then
        print_error "This script is designed for Linux systems only"
        exit 1
    fi
    
    # Check if Cursor is already installed
    if [ -f "$CURSOR_APPIMAGE" ]; then
        print_warning "Cursor appears to be already installed at: $CURSOR_APPIMAGE"
        read -p "Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Installation cancelled"
            exit 0
        fi
    fi
    
    # Download Cursor
    if ! download_cursor; then
        exit 1
    fi
    
    # Install dependencies
    install_dependencies
    
    # Setup Cursor
    setup_cursor
    
    # Clean up temp directory
    cd "$HOME"
    rm -rf "$TEMP_DIR"
    
    # Create desktop entry
    create_desktop_entry
    
    # Add shell alias
    add_shell_alias
    
    # Test installation
    if ! test_cursor; then
        ubuntu_version=$(detect_ubuntu_version)
        if [[ "$ubuntu_version" == "24.04" ]]; then
            print_warning "Standard installation may not work on Ubuntu 24.04"
            read -p "Do you want to try the extraction method? (Y/n): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                if extract_appimage; then
                    print_success "Alternative installation completed"
                fi
            fi
        fi
    fi
    
    echo
    print_success "Cursor IDE installation completed!"
    echo
    print_info "You can start Cursor using one of these methods:"
    echo "  1. Click on Cursor in your applications menu"
    echo "  2. Run: cursor (after restarting terminal)"
    echo "  3. Run: $CURSOR_APPIMAGE --no-sandbox"
    echo
    print_info "For GPU acceleration issues, you may need to add --disable-gpu flag"
    echo
}

# Run main function
main "$@"