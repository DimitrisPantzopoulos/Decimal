from setuptools import setup, find_packages

setup(
    name="Decimal",
    version="0.1.0",
    description="AlphaZero implementation for Hex",
    author="Dimitris Pantzopoulos",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "tqdm",
    ],
)
