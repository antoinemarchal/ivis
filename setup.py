# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="deconv",
    version="0.1",
    description="A package for imaing and joint deconvolution",
    author="amarchal",
    author_email="antoine.marchal@anu.edu.au",
    url="https://github.com/your_username/deconv",
    packages=find_packages(),
    install_requires=[
        "numpy", 
        "scipy",
        "torch"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
