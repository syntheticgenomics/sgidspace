import numpy as np

from sgidspace.sequence_generator import SGISequenceGenerator, IUPAC_CODES
from sgidspace.filters import filter_records, get_nested_key
from sgidspace.keywords import extract_keywords
from sgidspace.gene_names import process_gene_name
import keras.backend as K


def make_batch_processor(
        main_datadir,
        subdir,
        batch_size,
        outputs,
        inference=False,
        classflags=None,
        unbounded_iteration=True,
        from_embed=False
):
    seq_generator = SGISequenceGenerator(
        main_datadir + '/' + subdir + '/*.json*',
        unbounded_iteration=unbounded_iteration
    )
    return BatchProcessor(
            seq_generator,
            batch_size,
            outputs,
            inference=inference,
            classflags=classflags,
            from_embed=from_embed
        )


def flatten_ec(ecnum):
    if ecnum is None:
        return None
    elif ecnum is '':
        return ''

    out = []
    x = ecnum.split('.')
    for i in xrange(len(x)):
        if x[i] != '-':
            out.append('.'.join(x[:(i+1)]))

    return out


def transform_records(generator, data_dicts):
    """
    modify records from the form they take on disk to the format used by batch processor
    """
    for data in generator:
        if 'translation_description' in data:
            data['translation_description_keywords'] = extract_keywords(data['translation_description'])

        if 'gene_name' in data:
            data['gene_name'] = [process_gene_name(data['gene_name'])]

        if 'ec_number' in data:
            data['ec_number_ecflat'] = flatten_ec(data['ec_number'])

        if 'gene3d' in data:
            x = []
            for v in data['gene3d']:
                x += flatten_ec(v)
            data['gene3d_ecflat'] = list(np.unique(x))

        if 'diamond' in data:
            data['diamond_cluster_id'] = [data['diamond']['cluster_id']]

        for field in data_dicts:
            if field in data:
                dtransform = data_dicts[field]
                if type(data[field]) is list:
                    s = set()
                    for x in data[field]:
                        for v in dtransform.get(str(x), []):
                            s.add(v)
                            data[field + '_dict'] = list(s)
                else:
                    data[field + '_dict'] = dtransform.get(str(data[field]), [])

        yield data


class BatchProcessor():
    """
    Generates batches of data for training

    SGISequenceGenerator is used as a starting point which is subsequently
    filtered, transformed, and grouped into batches of size `batch_size`.
    """

    def __iter__(self):
        return self

    def next(self):
        if self.done:
            raise StopIteration()
        return self.fetch_batch()

    def __init__(
            self,
            seq_generator,
            batch_size,
            outputs,
            inference=False,
            classflags=None,
            from_embed=False
    ):
        self.batch_size = batch_size
        self.seq_generator = seq_generator
        self.outputs = outputs
        self.inference = inference
        self.done = False
        self.input_symbols = {label: i for i, label in enumerate(IUPAC_CODES)}
        self.from_embed = from_embed

        if from_embed:
            self.esize = 256

        self.class_index = {}
        data_dicts = {}
        for o in outputs:
            if 'class_labels' in o:
                self.class_index[o['name']] = {label: i for i, label in enumerate(o['class_labels'])}
            if type(o['datafun']) is dict:
                data_dicts[o['field']] = o['datafun']

        self.seq_generator = transform_records(self.seq_generator, data_dicts)

        if not from_embed:
            self.seq_generator = filter_records(self.seq_generator, outputs, classflags=classflags, inference=inference)

    def fetch_records(self):
        records = []

        for i in xrange(self.batch_size):
            r = next(self.seq_generator, None)
            if r is None:
                self.done = True
                break

            records.append(r)

        return records

    def fetch_batch(self):
        records = self.fetch_records()

        # initialize input
        X = {}
        if self.from_embed:
            X['embedding'] = np.zeros(
                [
                    len(records),
                    self.esize,
                ],
                dtype=K.floatx(),
            )
        else:
            X['sequence_input'] = np.zeros(
                [
                    len(records),
                    2000,
                    len(IUPAC_CODES),
                ],
                dtype=K.floatx(),
            )
        Y = {}

        # initialize output buffer
        for o in self.outputs:
            dtype = K.floatx()
            shape = [len(records), o['classcount']]
            Y[o['name']] = np.zeros(shape, dtype=dtype)

        # Copy record information
        for i in xrange(len(records)):
            record = records[i]

            if self.from_embed:
                X['embedding'][i,:] = np.array(record['embedding'], dtype=K.floatx())
            else:
                # input_sequence
                input_sequence = record['protein_sequence']
                for sequence_index in xrange(len(input_sequence)):
                    symbol_index = self.input_symbols.get(input_sequence[sequence_index])
                    if symbol_index is not None:
                        X['sequence_input'][i, sequence_index, symbol_index] = 1

                # add zero padding for the rest
                aai = self.input_symbols.get("*")
                X['sequence_input'][i, sequence_index + 1:, aai] = 1

                for o in self.outputs:
                    output_name = o['name']
                    r = get_nested_key(record, o['name'])

                    if r is not None:
                        if o['type'] == 'numeric':
                            Y[output_name][i] = r
                        elif o['type'] == 'boolean':
                            Y[output_name][i, int(r)] = 1
                        elif o['type'] in ['onehot', 'multihot']:
                            for label in r:
                                index = self.class_index[output_name].get(str(label))
                                if index is not None:
                                    Y[output_name][i, index] = 1
                                elif o['type'] == 'onehot':
                                    Y[output_name][i, o['classcount'] - 1] = 1
                    else:
                        if o['type'] == 'onehot':
                            Y[output_name][i, o['classcount'] - 1] = 1
        
        if self.inference:
            return X, records
        else:
            return X, Y
