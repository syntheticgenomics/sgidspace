#------------------------------------------------------------------------------
#
#  This file is part of sgidspace.
#
#  sgidspace is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sgidspace is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with sgidspace.  If not, see <http://www.gnu.org/licenses/>.
#
#------------------------------------------------------------------------------
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
            'parse_split_uniprot_dspace = sgidspace.parse_split_uniprot:main',
            'shuffle_uniprot_dspace = sgidspace.shuffle_uniprot:main',
        ]
      },
      package_data={'sgidspace': ['data/*']},
      zip_safe=False)
