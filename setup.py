from setuptools import setup, find_packages

version = open('VERSION').read().strip()
license = open('LICENSE').read().strip()

setup(
    name="pointstorm",
    version=version,
    license=license,
    description="Embedding vectors for data on the move",
    author="xsfa",
    author_email="tesfaaog@gmail.com",
    url="https://github.com/xsfa/pointstorm",
    packages=find_packages(),
    install_requires=[
        'bytewax==0.16.0',
        'requests>=2.28.0',
        'kafka-python==2.0.2',
        'confluent-kafka',
        'faker',
        'transformers',
        'torch',
        'pydantic',
        'unstructured'
        'numpy',
        'unittest'
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
