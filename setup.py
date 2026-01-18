#!/usr/bin/env python3
"""
CatBP: Categorical Belief Propagation

Setup script for installation.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="catbp",
    version="1.0.0",
    author="Juan Zambrano, Enrique ter Horst, Sridhar Mahadevan",
    author_email="jd.yokim@gmail.com",
    description="Categorical Belief Propagation with holonomy-aware compilation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/catbp",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GPL-3.0",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "catbp=main:main",
        ],
    },
)
