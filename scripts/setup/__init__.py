"""
Setup Scripts Package
=====================

This package contains scripts for setting up the GENESIS environment and dependencies.

Scripts:
- setup.py: Main package setup script
- setup_environment.py: Environment setup utilities
- setup_micromamba.sh: Micromamba environment setup script
- getting_started.py: Getting started verification script
"""

import os
import sys

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

__all__ = [
    "setup",
    "setup_environment", 
    "getting_started",
]
