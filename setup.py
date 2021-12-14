from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="SAE-Pipe",
    version="0.0.1",
    author="Lake Yin",
    author_email="yinl3@rpi.edu",
    description="Self-supervised Arbitrary Embedding Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LakeYin/SAE-Pipe",
    packages=find_packages(),
    install_requires=["tensorflow","numpy"],
    python_requires=">=3.6",
)