#!/usr/bin/env python3
"""
Universal LSP Server - Language Server Protocol для Python проектов
Оптимизирован для работы с AI ассистентами
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="universal-lsp-server",
    version="1.0.0",
    author="AI Assistant",
    description="Универсальный LSP сервер для Python с поддержкой AI контекста",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/universal-lsp-server",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pygls>=1.0.0",
        "jedi>=0.18.0",
        "pyyaml>=6.0",
        "click>=8.0",
        "watchdog>=2.0",
        "aiofiles>=0.8.0",
        "colorama>=0.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lsp-server=lsp_server.cli:cli",
            "universal-lsp=lsp_server.cli:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "lsp_server": ["config/*.yaml", "templates/*"],
    },
)