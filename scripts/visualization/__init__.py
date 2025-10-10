"""
Visualization Scripts Package
=============================

This package contains scripts for visualizing data and model outputs.

Scripts:
- visualize_data.py: Visualize input data distributions
- visualize_diffusion.py: Visualize diffusion process and generated events
"""

import os
import sys

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

__all__ = [
    "visualize_data",
    "visualize_diffusion",
]
