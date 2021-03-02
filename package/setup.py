from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="package_divyansh",
    version="0.0.1",
    author="divyansh",
    author_email="divyansh.sharma@tigeranalytics.com",
    description="A package to predict housing prices.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/divyansh3005/housing_package",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
