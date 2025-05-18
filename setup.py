import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name='video-extractor',
    version='1.1.2',
    author="wudu",
    author_email="296525335@qq.com",
    description="support export ppt from a video",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['video2ppt', ],
    package_dir={'video2ppt': 'video2ppt'},
    py_modules=['evp'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    install_requires=[
        'click>=8.0.0',
        'fpdf2>=2.7.5',
        'matplotlib>=3.5.0',
        'opencv-python>=4.5.5.64',
        'numpy>=1.21.0',
        'psutil>=5.9.0',
        'pillow>=9.0.0',
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'accelerate>=0.20.0',
        'sentencepiece>=0.1.99',
        'protobuf>=3.20.0'
    ],
    entry_points='''
        [console_scripts]
        evp=video2ppt:video2ppt
        evp-window=video2ppt:window2ppt
        evp-smart-window=video2ppt:smart_window2ppt
    ''',
)