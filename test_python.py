#!/usr/bin/env python3
import sys
print(f"Python executable: {sys.executable}", file=sys.stderr)
print(f"Python version: {sys.version}", file=sys.stderr)
print("Test output to stdout")
sys.stderr.write("Test output to stderr\n")
sys.stderr.flush()
sys.stdout.flush()