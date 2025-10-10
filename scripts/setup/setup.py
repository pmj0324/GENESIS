#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for GENESIS IceCube diffusion model.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="genesis-icecube",
    version="1.0.0",
    author="Minje Park",
    author_email="pmj032400@naver.com",
    description="A diffusion model for generating IceCube muon neutrino events",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/GENESIS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=5.0.0",
            "pytest>=7.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
            "seaborn>=0.11.0",
        ],
        "distributed": [
            "deepspeed>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "genesis-train=train:main",
            "genesis-sample=sample:main",
            "genesis-evaluate=evaluate:main",
            "genesis-compare=compare_architectures:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.csv"],
    },
    zip_safe=False,
)