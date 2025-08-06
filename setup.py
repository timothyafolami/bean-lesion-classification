"""
Setup script for Bean Lesion Classification system.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Read README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Bean Lesion Classification System"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
else:
    requirements = []

setup(
    name="bean-lesion-classification",
    version="1.0.0",
    description="End-to-end ML system for bean leaf disease classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Bean Classification Team",
    author_email="team@example.com",
    url="https://github.com/example/bean-lesion-classification",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "onnxruntime-gpu>=1.15.0",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine-learning deep-learning computer-vision classification pytorch fastapi",
    entry_points={
        "console_scripts": [
            "bean-train=training.train:main",
            "bean-api=api.main:main",
            "bean-convert=inference.convert:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)