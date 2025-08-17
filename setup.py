from setuptools import setup, find_packages

setup(
    name="dsopt",
    version="1.0.0",
    description="Discrete parallel Monte Carlo-based surrogate optimization for mixed design spaces",
    packages=find_packages(),
    install_requires=["xgboost", "numpy", "pandas"],
    python_requires=">=3.8",
)