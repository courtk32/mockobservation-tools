import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mockobservation_tools",
    version=__version__,
    author="Courtney Klein",
    author_email="kleinca@uci.edu",
    description="Software tools for mock observations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/courtk32/mockobservation-tools",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'astropy',
        'firestudio',
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'h5py',
    ],
)
