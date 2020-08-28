"""
Installer for NGKCS module.

This is very much proof-of-concept code.
"""
from setuptools import setup  # , find_packages

setup(
    name="ngkcs",
    version="0.1",
    packages=["ngkcs"],
    install_requres=[
        "aiokatcp==0.8.0",  # Note, that this version will give deprecation warnings in Python 3.8.
        "ipaddress",
        "pytest-asyncio",
        "docker",
    ],
)
