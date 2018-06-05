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
import argparse
import glob
import gzip
import json
from random import Random


class ShuffleJsonFiles(object):
    # stores all records in memory - will require a lot of RAM

    def __init__(self):
        self._parse_args()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--input_prefix', required=True, help="Input prefix of json.gz files")
        parser.add_argument('--output_prefix', required=True, help="Output prefix of shuffled json.gz files")
        parser.add_argument('--random_seed', required=False, default=2018, help="Random seed to use")
        self.args = parser.parse_args()

    def _import_all_records(self):
        self.all_records = []
        for filename in sorted(glob.glob(self.args.input_prefix + '*')):
            with gzip.open(filename) as f:
                for line in f:
                    self.all_records.append(json.loads(line))

    def _shuffle_records(self):
        seeded_random = Random()
        seeded_random.seed(self.args.random_seed)
        seeded_random.shuffle(self.all_records)

    def _next_output_file(self, output_prefix, output_file_index):
        filename = '%s_%06d.json.gz' % (output_prefix, output_file_index)
        f = gzip.open(filename, 'w')
        return f

    def _write_output_files(self):
        output_file_index = 1
        file_handle = self._next_output_file(self.args.output_prefix, output_file_index)

        lines_in_current_file = 0
        for record in self.all_records:
            print >> file_handle, json.dumps(record)
            lines_in_current_file += 1

            if lines_in_current_file == 10000:
                file_handle.close()
                output_file_index += 1
                file_handle = self._next_output_file(self.args.output_prefix, output_file_index)
                lines_in_current_file = 0

        file_handle.close()

    def run(self):
        self._import_all_records()
        self._shuffle_records()
        self._write_output_files()


def main():
    sjf = ShuffleJsonFiles()
    sjf.run()


if __name__ == '__main__':
    main()
