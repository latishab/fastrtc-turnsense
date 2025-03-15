from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fastrtc-turnsense",
    version="0.1.0",
    author="Latisha Besariani HENDRA",
    author_email="latishabesariani@gmail.com",
    description="Turn-taking detection system for conversational AI built on FastRTC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fastrtc-turnsense",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "fastrtc>=0.1.0",
        "numpy>=1.19.0",
        "onnxruntime>=1.12.0",
        "transformers>=4.20.0",
        "huggingface-hub>=0.10.0",
        "librosa>=0.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=3.9",
        ],
    },
) 