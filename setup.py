from distutils.cmd import Command
from setuptools import setup, find_packages
from pip.req import parse_requirements
from pip.download import PipSession
from os import listdir
from os.path import isfile, isdir, join, exists, dirname
import sys
import os

name = 'sgidspace'
description = 'Synthetic Genomics Public DSpace Libraries'
url='https://github.com/syntheticgenomics/sgidspace'

abs_path = os.path.dirname(os.path.abspath(__file__))
requirements_file = os.path.join(abs_path, 'requirements.txt')
install_reqs = parse_requirements(requirements_file, session=PipSession())
reqs = [str(ir.req) for ir in install_reqs]

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
        ]
      },
      zip_safe=False)
