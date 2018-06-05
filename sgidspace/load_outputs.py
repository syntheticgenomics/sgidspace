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
import os
import json
import gzip

TYPES = {'boolean', 'onehot', 'multihot', 'numeric', 'embedding_autoencoder'}


def _load_class_table(filename, datadir):
    path = os.path.join(datadir, filename)

    labels = []
    rel_freq = []
    freq = []

    f = open(path).__iter__()
    f.next()
    for line in f:
        if line.startswith("#"):
            continue
        if line.startswith("Label\t"):
            continue
        cols = line.rstrip().split('\t')

        # Should not happen, but there are keywords with tabs in them :(
        if len(cols) > 3:
            continue

        labels.append(cols[0])
        if len(cols) > 1:
            rel_freq.append(float(cols[1]))
            if len(cols) > 2:
                freq.append(float(cols[2]))

    if (len(rel_freq) == 0):
        return {'labels': labels}
    elif (len(freq) == 0):
        return {'labels': labels, 'rel_freq': rel_freq}
    else:
        return {'labels': labels, 'rel_freq': rel_freq, 'freq': freq}


def validate_output_type(output_type):
    if output_type not in TYPES:
        raise ValueError((
            'When loading outputs, found {found}, but expected one of '
            'the following: {expected}'
        ).format(
            found=output_type,
            expected=TYPES,
        ))


def load_outputs(outputs_file, datadir=None):
    if not datadir:
        datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    outputs = list()
    with open(os.path.join(datadir, outputs_file)) as f:
        next(f)
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip().split('\t')
            if len(cols) < 5:
                continue
            if cols[1] != 'Y' or line[0] == '#':
                continue

            output = {
                'field': cols[0],
                'scale': float(cols[2]),
                'type': cols[3],
                'datadir': cols[6],
                'minhot': int(cols[8]),
            }

            validate_output_type(output['type'])

            if cols[4] != 'NA':
                output['classcount'] = int(cols[5])
                ct = _load_class_table(cols[4], datadir)
                output['class_labels'] = ct['labels'][:output['classcount']]
                if 'freq' in ct:
                    output['class_freq'] = ct['freq'][:output['classcount']]

            if cols[7] != '.':
                f = cols[7]
                path = os.path.join(datadir, f)
                if os.path.exists(path):
                    output['datafun'] = json.load(gzip.open(path))
                    output['name'] = output['field'] + '_dict'
                else:
                    output['datafun'] = f
                    output['name'] = output['field'] + '_' + output['datafun'].split('.')[0]
            else:
                output['datafun'] = None
                output['name'] = output['field']

            if cols[3] == 'onehot':
                output['class_labels'].append('None')
                output['classcount'] += 1
                if 'class_freq' in output:
                    output['class_freq'].append(1 - sum(output['class_freq']))

            if cols[9] != '.':
                output['text_length'] = int(cols[9])
            else:
                output['text_length'] = None

            if output['type'] == 'boolean':
                output['classcount'] = 2
                output['class_labels'] = ['False', 'True']
            elif output['type'] in ['numeric']:
                output['classcount'] = 1
                output['class_labels'] = []
            elif output['type'] in ['embedding_autoencoder']:
                output['classcount'] = int(cols[5])
                output['class_labels'] = []

            outputs.append(output)

        return outputs
