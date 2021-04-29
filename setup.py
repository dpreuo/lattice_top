from setuptools import setup, Extension, find_packages
import numpy
import os


setup(
    name="lattice_top",
    version='0.1',
    description='Lattice topological systems and local markers',
    author="Peru d'Ornellas",
    author_email='ppd19@iac.ac.uk',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)


# command to build inplace is: python setup.py build_ext --inplace
# command to install is: pip install --editable .
