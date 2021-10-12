from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="SimSEP",
    version="0.0.1",
    author="Lake Yin",
    author_email="yinl3@rpi.edu",
    description="SIMilarity based Self-supervised Embedding Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LakeYin/SimSEP",
    packages=find_packages(include=["simsep"]),
    install_requires=["tensorflow","numpy","pandas"],
    python_requires=">=3.6",
)