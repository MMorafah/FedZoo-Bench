import os
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='FedZoo-Bench',
    version='1.0.0',
    author='Mahdi Morafah, Weijia Wang',
    author_email='wweijia@eng.ucsd.edu',
    description='PyTorch implementation of the state-of-the-art federated learning benchmarks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/MMorafah/FedZoo-Bench',
    license='MIT',
    packages=['FedZoo-Bench'],
    install_requires=['torch', 'torchvision', 'numpy'], # TODO: specify versions
)
