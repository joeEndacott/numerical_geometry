"""
Setup
=====
"""

import os
from setuptools import setup, find_packages


def read_readme():
    """
    Read README
    ===========

    Reads the README.md file if it exists.
    """
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return (
        "A numerical geometry project in Python, with a focus on mesh deformations using "
        "neural-networks."
    )


setup(
    name="numerical_geometry",
    version="0.1.0",
    author="Joseph Endacott",
    author_email="joseph.endacott@gmail.com",
    description=(
        "A package for numerical geometry computations including mesh generation, deformation, and "
        "neural network-based shape transformations."
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pyvista",
        "matplotlib",
        "pyvista",
        "scipy",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
