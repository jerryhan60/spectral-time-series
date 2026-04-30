from setuptools import setup, find_packages

setup(
    name="poly-precond",
    version="0.1.0",
    description="Polynomial Input Preconditioning for Zero-Shot Time Series Forecasting",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "einops",
        "numpy",
    ],
)
