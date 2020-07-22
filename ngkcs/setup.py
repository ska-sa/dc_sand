from setuptools import setup, find_packages

setup(
    name="ngcks",
    version="0.1",
    packages=["ngkcs",],
    install_requres=["aiokatcp==0.8.0",], # Note, that this version will give deprecation warnings in Python 3.8.
)