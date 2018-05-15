from distutils.cmd import Command
from setuptools import setup, find_packages
from os import listdir
from os.path import isfile, isdir, join, exists, dirname
import sys
import os

name = 'sgidspace'
description = 'Synthetic Genomics Public DSpace Libraries'
url='https://github.com/syntheticgenomics/sgidspace'

abs_path = os.path.dirname(os.path.abspath(__file__))
requirements_file = os.path.join(abs_path, 'requirements.txt')
with open(requirements_file) as f:
    reqs = f.read().splitlines()

# set the version
exec(open(os.path.join(abs_path, name, 'version.py')).read())

setup(name=name,
      version=__version__,
      description=description,
      url=url,
      packages=find_packages(),
      install_requires=reqs,
      entry_points={
        'console_scripts': [
            'train_dspace = sgidspace.train_dspace:main',
            'infer_dspace = sgidspace.inference:main',
        ]
      },
      package_data={'sgidspace': ['data/*']},
      zip_safe=False)
