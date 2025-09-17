#!/usr/bin/env python3
"""
Setup script for G2C Framework
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read version from g2c module
def read_version():
    version_file = os.path.join("src", "g2c.py")
    with open(version_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "0.1.0"

setup(
    name="g2c",
    version=read_version(),
    author="EarthOnlinePlayer5732",
    description="A lightweight framework for console-based games and applications",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/EarthOnlinePlayer5732/G2C",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.6",
    keywords="console games framework terminal cli",
    project_urls={
        "Bug Reports": "https://github.com/EarthOnlinePlayer5732/G2C/issues",
        "Source": "https://github.com/EarthOnlinePlayer5732/G2C",
    },
)