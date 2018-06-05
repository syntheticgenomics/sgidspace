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
import gzip
import json
import sys
import re
from Bio import SwissProt
from random import Random


class UniprotDataDict(object):
    def __init__(self, inputSprot, inputTrembl, inputIdsToUse):
        self.inputSprot = inputSprot
        self.inputTrembl = inputTrembl
        self.inputIdsToUse = inputIdsToUse

        self._create_good_id_dict()

    def _create_good_id_dict(self):
        self.ids_to_use = {}

        if self.inputIdsToUse:
            with open(self.inputIdsToUse) as f:
                for line in f:
                    line = line.rstrip()
                    self.ids_to_use[line] = 1

    def _uniprot_file_handle(self):
        uniprot_files = []
        if self.inputSprot:
            uniprot_files.append(self.inputSprot)
        if self.inputTrembl:
            uniprot_files.append(self.inputTrembl)

        current_handle = None
        fileNumber = 0
        for file_name in uniprot_files:
            if current_handle:
                current_handle.close()
            current_handle = gzip.open(file_name)
            fileNumber += 1
            yield current_handle, fileNumber

    def _check_id_to_use(self, id):
        if self.inputIdsToUse:
            if id in self.ids_to_use:
                return True
            else:
                sys.stderr.write("Not using ID " + str(id) + "\n")
                return False
        else:
            return True

    def _file_number_to_source(self, file_number):
        if (file_number == 2) or (file_number == 1 and not self.inputSprot):
            return 'trembl'
        else:
            return 'sprot'

    def generate_uniprot_record(self):
        for file_handle, file_number in self._uniprot_file_handle():
            data_source = self._file_number_to_source(file_number)
            for record in SwissProt.parse(file_handle):
                if self._check_id_to_use(record.accessions[0]):
                    current_record_dict = self._parse_record(record, data_source)
                    yield current_record_dict

    def _parse_record(self, record, data_source):
        record_parsed = {}
        record_parsed['data_source'] = data_source
        record_parsed["translation_name"] = record.accessions[0]

        record_parsed["translation_length"] = record.sequence_length
        record_parsed["protein_sequence"] = record.sequence
        record_parsed["translation_tax_id"] = str(record.taxonomy_id[0])

        translation_description_search = re.search('Full\=([^;{]+)( \{[^;]+\})*\;', record.description)
        if translation_description_search:
            record_parsed['translation_description'] = translation_description_search.group(1)

        ec_number_search = re.search('EC\=([^;{]+)( \{[^;]+\})*\;', record.description)
        if ec_number_search:
            record_parsed['ec_number'] = ec_number_search.group(1)

        for comment in record.comments:
            function_search = re.search('FUNCTION:\s([^\.]+)', comment)
            if function_search:
                record_parsed['function'] = function_search.group(1)
                break

        # gene_name: Name=, ORFNames=, OrderedLocusNames=   BUT only Name= is a good id to keep
        if hasattr(record, 'gene_name'):
            name_match = re.search('Name=([a-zA-Z0-9_-]+)', record.gene_name)
            if name_match:
                record_parsed['gene_name'] = name_match.group(1)

        # cross references are numerous and can be mostly handled identically
        possible_cross_refs = ['Pfam', 'TIGRFAMs', 'KEGG', 'GO', 'InterPro', 'OrthoDB', 'PROSITE', 'SUPFAM',
                               'SMART', 'ProteinModelPortal', 'Reactome', 'Gene3D']
        for entry in possible_cross_refs:
            cr = self._cross_ref(record, entry)
            if cr:
                record_parsed[entry.lower()] = cr

        # metacyc needs a regex so is treated separately from other cross refs
        cr = self._cross_ref(record, 'BioCyc')
        if cr:
            metacyc_match = re.search('MetaCyc:([^;]+)', str(cr))
            if metacyc_match:
                record_parsed['metacyc'] = metacyc_match.group(1)

        record_parsed["tm_helix_count"] = self._tm_helix_count(record)

        return record_parsed

    def _cross_ref(self, record, feature):
        return map(lambda x: x[1], filter(lambda x: x[0] == feature, record.cross_references))

    def _tm_helix_count(self, record):
        tm_helix = 0
        if record.features:
            for feature in record.features:
                if len(feature) > 3:
                    if feature[0] == 'TRANSMEM' and re.search('Helical', feature[3]):
                        tm_helix += 1
        return tm_helix


class UniprotDataJsonLines(object):
    def __init__(self, output):
        self.output = output

    def _which_split(self, val_fraction, test_fraction, current_rand):
        train_fraction = 1 - val_fraction - test_fraction
        min_train_rand = 0
        max_train_rand = train_fraction
        min_val_rand = max_train_rand
        max_val_rand = min_val_rand + val_fraction
        min_test_rand = max_val_rand
        max_test_rand = 1

        if current_rand > min_test_rand:
            split = 'test'
        elif current_rand > min_val_rand:
            split = 'val'
        else:
            split = 'train'

        return split

    def split_and_output_json_lines(self, data_records, val_fraction, test_fraction, random_seed=92037):
        # SGI mailing address zip code is the default random seed

        zip_code_rand = Random()
        zip_code_rand.seed(random_seed)

        file_handle = {}
        output_file_number = {'train': 1, 'val': 1, 'test': 1}
        for split in ['train', 'val', 'test']:
            file_handle[split] = self._next_output_file(self.output + split, output_file_number[split])
        n_lines_in_current_file = {'train': 0, 'val': 0, 'test': 0}

        for record in data_records:
            current_rand = zip_code_rand.random()
            current_split = self._which_split(val_fraction, test_fraction, current_rand)

            print >> file_handle[current_split], json.dumps(record)
            n_lines_in_current_file[current_split] += 1

            if n_lines_in_current_file[current_split] == 10000:
                n_lines_in_current_file[current_split] = 0
                output_file_number[current_split] += 1
                file_handle[current_split].close()
                file_handle[current_split] = self._next_output_file(self.output + current_split, output_file_number[current_split])

    def _next_output_file(self, output_prefix, output_file_index):
        filename = '%s_%06d.json.gz' % (output_prefix, output_file_index)
        f = gzip.open(filename, 'w')
        return f



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputSprot', required=False, help="Input sprot dat.gz file")
    parser.add_argument('--inputTrembl', required=False, help="Input trembl dat.gz file")
    parser.add_argument('--inputIdsToUse', required=False, help="Input file of IDs to use (discard all others)")
    parser.add_argument('--output', required=True, help="Out prefix of json.gz files")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    udd = UniprotDataDict(args.inputSprot, args.inputTrembl, args.inputIdsToUse)
    data_records = udd.generate_uniprot_record()

    udjs = UniprotDataJsonLines(args.output)
    udjs.split_and_output_json_lines(data_records, 0.1, 0.1)        # hard-coded 10% val and 10% test

if __name__ == '__main__':
    main()
