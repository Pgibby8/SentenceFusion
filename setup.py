import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Pgibby8", # Replace with your own username
    version="0.0.1",
    author="Paul Gibby",
    author_email="pgibby8@hotmail.com",
    description="A package for fusing the semantics of two sentences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pgibby8/SentenceFusion",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
