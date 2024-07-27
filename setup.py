from setuptools import setup, find_packages

VERSION = "0.1.0"
DESCRIPTION = "One Foundation Model Fits All: Single-stage Foundation Model Training with Zero-shot Deployment"

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="ofm_sam",
    version=VERSION,
    packages=find_packages(),
    install_requires=required,
    entry_points={
        "console_scripts": [
            # If you have any scripts you want to be executable from the command line
        ]
    },
    # Optional attributes
    author="",
    author_email="",
    description=DESCRIPTION,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",  # This is important for a README.md file
    url="https://github.com/wmabebe/OFM_SAM",  # URL to the homepage of your package
    # More metadata: https://packaging.python.org/guides/distributing-packages-using-setuptools/#setup-args
)
