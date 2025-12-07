from setuptools import setup, find_packages

setup(
    name="safety_benchmark",
    version="0.1.0",
    packages=find_packages(),
    description="Safe Manipulation Benchmark for Robot Learning",
    author="",  # Add author info if desired
    python_requires=">=3.7",
    install_requires=[
        "telemoma==0.3.0",
        "open3d==0.19.0",
    ]
) 