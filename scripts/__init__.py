"""
GENESIS Scripts Package
=======================

This package contains all the main scripts for running GENESIS.

Subpackages:
- analysis: Analysis and evaluation scripts
- setup: Environment setup scripts  
- visualization: Data and model visualization scripts

Main Scripts:
- train.py: Main training script
- sample.py: Event generation script
"""

import os
import sys

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

__all__ = [
    "train",
    "sample",
    "analysis",
    "setup", 
    "visualization",
]
