"""
Analysis Scripts Package
========================

This package contains scripts for analyzing and comparing different model architectures
and evaluating model performance.

Scripts:
- compare_architectures.py: Compare different model architectures
- evaluate.py: Evaluate trained models
"""

from .compare_architectures import main as compare_architectures
from .evaluate import main as evaluate

__all__ = [
    "compare_architectures",
    "evaluate",
]
