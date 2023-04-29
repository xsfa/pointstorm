from setuptools import setup, find_packages

setup(
    name="pointstorm",
    version="1.1.0",
    description="Emebedding vectors for real-time streaming data",
    author="Tesfa Shenkute",
    author_email="tesfaaog@gmail.com",
    url="https://github.com/xsfa/pointstorm",
    packages=find_packages(),
    install_requires=[
        'bytewax==0.10.1',
        'requests>=2.28.0',
        'kafka-python==2.0.2',
        'confluent-kafka',
        'faker',
        'transformers',
        'torch'
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
