#!/usr/bin/env python3
import sys
import os

# Принудительно выводим все в stderr
sys.stdout = sys.stderr

print("=== FORCE RUN DOWNLOAD ===", file=sys.stderr)
print(f"Python: {sys.executable}", file=sys.stderr)
print(f"Working dir: {os.getcwd()}", file=sys.stderr)

try:
    # Настраиваем логирование чтобы выводилось в stderr
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        stream=sys.stderr,
        force=True
    )
    
    print("\nImporting download_data...", file=sys.stderr)
    import download_data
    
    print("\nRunning main()...", file=sys.stderr)
    download_data.main()
    
except Exception as e:
    print(f"\nERROR: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)