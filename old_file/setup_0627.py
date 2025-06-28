#!/usr/bin/env python3
"""
COVID-19変異予測パッケージのセットアップスクリプト
"""

from setuptools import setup, find_packages
from pathlib import Path

# README.mdの読み込み
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# requirements.txtの読み込み
def read_requirements():
    requirements_file = this_directory / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file) as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="covid-mutation-prediction",
    version="1.0.0",
    author="AI Research Team",
    author_email="research@example.com",
    description="COVID-19変異予測のための深層学習パッケージ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/covid-mutation-prediction",
    project_urls={
        "Bug Tracker": "https://github.com/example/covid-mutation-prediction/issues",
        "Documentation": "https://covid-mutation-prediction.readthedocs.io/",
        "Source Code": "https://github.com/example/covid-mutation-prediction",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "covid-mutation-predict=covid_mutation_prediction.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "covid_mutation_prediction": [
            "config/*.yaml",
            "data/templates/*.csv",
        ],
    },
    zip_safe=False,
    keywords=[
        "covid-19",
        "mutation prediction", 
        "deep learning",
        "transformer",
        "bioinformatics",
        "machine learning",
        "pytorch"
    ],
)
