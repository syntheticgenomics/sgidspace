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
import re
import glob
import gzip
import json
import argparse

from signal import signal, SIG_DFL, SIGPIPE
signal(SIGPIPE, SIG_DFL)

filters = [
          '^LOC[0-9]',
          '^ACEY_',
          '^ACIB2EUKG',
          '^ABSGL',
          '^KNAG',
          '^MO[0-9][0-9][0-9][0-9][0-9]',
          'ORF[0-9]',
          '^PARPA',
          '^[0-9]',
          '^PISO0',
          '^PPCYP[0-9]',
          '^SLC[0-9]',
          '^SPOSA[0-9]',
          '^SSCI[0-9]',
          '^ST[0-9][0-9][0-9]'
          '^TBLA[0-9]',
          '^TDEL[0-9]',
          '^TPHA[0-9]',
          '^ZPAR[0-9]',
          '^AAEL[0-9][0-9]',
          '^ACYPI[0-9][0-9]',
          '^AGAP[0-9][0-9][0-9]',
          '^ALNC[0-9][0-9]',
          '^AT[0-9]G[0-9]',
          '^B[0-9][0-9]*[A-Z][0-9]',
          '^BBOV',
          '^BNA[A-Z][0-9][0-9]',
          '^BNA[A-Z]NN',
          'CONTIG',
          '^DKFZP',
          '^FG[0-9][0-9][0-9]',
          '^GA[0-9][0-9][0-9]',
          '^K7',
          '^KAFR',
          '^LOAG',
          '^MGC[0-9][0-9]',
          '^MO[0-9][0-9]',
          '^NCAS[0-9]',
          '^NDAI[0-9]',
          '^OJ[0-9][0-9]',
          '^OS[0-9][0-9]G',
          '^OS[A-Z][A-Z][A-Z].*[0-9]',
          '^P[0-9][0-9][0-9][0-9][A-Z]',
          '^PISO0',
          '^PM[A-Z]G.*[0-9][0-9]',
          '^POCGH01',
          '^POWCR01',
          '^RVY.*[0-9][0-9]',
          '^TBLA[0-9]',
          '^VAR.*[0-9][0-9][0-9]',
          '^VAR.*WDBLA',
          '^WBGENE[0-9][0-9]',
          '^A[0-9][A-Z][A-Z]',
          '^BLA[A-Z][A-Z][A-Z]',
          '^BM1.*[0-9][0-9]',
          '^BMA.*[A-Z][A-Z]',
          '^CNI.*[A-Z][A-Z][A-Z]',
          '^CRE.*[A-Z][A-Z][A-Z]',
          '^PCMP.*[A-Z]',
          '^PHI92.*GENE',
          '^POLX.*[0-9]',
          '^SFRICE',
          ]

def process_gene_name(name):

    # Convert to uppercase
    name = name.upper()

    # Entry filters
    for f in filters:
        if re.search(f, name, flags=re.MULTILINE):
            return 'None'

    # Perform string substitutions
    name = re.sub('_', '-', name, flags=re.IGNORECASE)

    return name


def main():

    parser = argparse.ArgumentParser(description='Extract gene names from data files.')
    parser.add_argument('data', help="A data directory containing processed data files")

    args = parser.parse_args()

    filenames = glob.glob(args.data + '/*.json.gz')

    for filename in filenames:
        data_handle = gzip.GzipFile(filename, 'rb')
        for data in data_handle:
            if isinstance(data, bytes):
                data = json.loads(data.decode('utf-8'))
            else:
                data = json.loads(data)
            if 'gene_name' in data:
                print(data['gene_name'] + '\t-->\t' + process_gene_name(data['gene_name']))


if __name__ == '__main__':
    main()
