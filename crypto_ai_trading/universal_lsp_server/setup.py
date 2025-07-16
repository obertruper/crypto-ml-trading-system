#!/usr/bin/env python3
"""
Setup script for Universal LSP Server
"""

from setuptools import setup, find_packages
from pathlib import Path

# Читаем README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Читаем requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines() 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="universal-lsp-server",
    version="1.0.0",
    author="Universal LSP Server Contributors",
    author_email="",
    description="Универсальный LSP сервер для улучшения работы с кодом в AI-ассистентах",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/universal-lsp-server",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lsp-server=lsp_server.cli:main",
            "universal-lsp=lsp_server.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "lsp_server": [
            "config/*.yaml",
            "config/*.yml",
            "templates/*",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
            "pre-commit>=3.0.0",
        ],
        "all": [
            "jedi>=0.19.0",
            "rope>=1.9.0",
            "pylint>=3.0.0",
            "flake8>=6.0.0",
            "autopep8>=2.0.0",
            "mypy>=1.5.0",
        ],
    },
)