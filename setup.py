import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dankypipe",
    version="0.0.3",
    author="Vivek Pagadala, Luke Waninger",
    author_email="luke.waninger@gmail.com",
    description="A dankly qualified validation pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lukeWaninger/GSCAP",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'boto3==1.9.84',
        'jsonschema==2.6.0',
        'lightgbm==2.2.2',
        'numpy==1.16.0',
        'pandas==0.24.0',
        'paramiko==2.4.2',
        'requests>=2.20.0',
        'scikit-learn==0.20.2',
        'scipy==1.2.0',
        'tqdm==4.29.1',
        'urllib3==1.26.5',
        'pyyaml>=4.2b1'
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={
        '': '*.txt'
    }
)
