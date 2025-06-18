from setuptools import setup, find_packages

setup(
    name="sugmind",
    version="1.0.0",
    description="Python client for Segmind APIs",
    author="Sebastian",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "Pillow>=8.0.0",
    ],
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)