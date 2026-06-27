#!/usr/bin/env python3
"""
Simple launcher script for Turbulence Realm - SINDy (v2.0 modular package).
"""
import sys
import os

# Handle frozen (PyInstaller / Nuitka) environment
if getattr(sys, 'frozen', False):
    sys.path.insert(0, getattr(sys, '_MEIPASS', os.path.dirname(sys.argv[0])))
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tr_sindy_app import main

if __name__ == "__main__":
    main()
