# Cursor IDE Installation Guide for Linux

## Overview
Cursor is an AI-powered code editor built on top of VS Code, designed to help developers write better code faster with AI assistance.

## Official Website
- **Main site**: https://cursor.com
- **Direct download**: https://cursor.com/download

## Quick Installation

### Method 1: Using the Installation Script (Recommended)
```bash
# Run the installation script
./install_cursor_ide.sh
```

This script will:
- Download the latest stable version of Cursor
- Install necessary dependencies
- Set up desktop integration
- Add command-line alias
- Handle Ubuntu 24.04 compatibility issues

### Method 2: Manual Installation

1. **Download Cursor AppImage**
   ```bash
   wget --user-agent="Mozilla/5.0" -O cursor.AppImage \
        "https://www.cursor.com/api/download?platform=linux-x64&releaseTrack=stable"
   ```

2. **Make it executable**
   ```bash
   chmod +x cursor.AppImage
   ```

3. **Install dependencies (Ubuntu < 24.04)**
   ```bash
   sudo apt update
   sudo apt install libfuse2
   ```

4. **Run Cursor**
   ```bash
   ./cursor.AppImage --no-sandbox
   ```

## Ubuntu 24.04 Special Instructions

Ubuntu 24.04 has compatibility issues with AppImage. If you encounter problems:

1. **Extract the AppImage**
   ```bash
   ./cursor.AppImage --appimage-extract
   ```

2. **Run from extracted directory**
   ```bash
   ./squashfs-root/cursor
   ```

## System Requirements

- **OS**: Linux (Ubuntu 20.04+, Fedora, Debian, etc.)
- **Architecture**: x86_64 (AMD64) or ARM64
- **RAM**: 4GB minimum, 8GB recommended
- **Disk Space**: 500MB

## Features

- AI-powered code completion
- Chat with AI about your code
- Automatic bug detection and fixes
- Code refactoring suggestions
- Multi-file editing with context awareness
- Built on VS Code, supports all VS Code extensions

## Troubleshooting

### "FUSE" Error
If you see an error about FUSE:
```bash
# For Ubuntu/Debian:
sudo apt install libfuse2

# For Fedora:
sudo dnf install fuse-libs
```

### GPU Acceleration Issues
If Cursor has rendering problems:
```bash
./cursor.AppImage --no-sandbox --disable-gpu
```

### Ubuntu 24.04 "Not Responding"
Use the extraction method described above instead of running the AppImage directly.

### Permission Denied
Ensure the AppImage is executable:
```bash
chmod +x cursor.AppImage
```

## Command Line Usage

After installation with the script, you can use:
```bash
# Start Cursor
cursor

# Open a specific file
cursor myfile.py

# Open a directory
cursor /path/to/project
```

## Updates

Cursor has built-in auto-update functionality. You can also manually check for updates:
- Go to Help → Check for Updates
- Or download the latest version and replace the AppImage

## Uninstallation

To remove Cursor:
```bash
# Remove AppImage
rm ~/Applications/cursor.AppImage

# Remove desktop entry
rm ~/.local/share/applications/cursor.desktop

# Remove alias from shell config
# Edit ~/.bashrc or ~/.zshrc and remove the cursor alias line
```

## Additional Resources

- **Official Documentation**: https://cursor.com/docs
- **Community Forum**: https://forum.cursor.com
- **GitHub Issues**: https://github.com/getcursor/cursor/issues

## License

Cursor is proprietary software. Check their website for licensing details.

---

**Note**: This installation guide was created on 2025-07-16 and may need updates as Cursor evolves.