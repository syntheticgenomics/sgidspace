#!/usr/bin/env python
import argparse
import json
import gzip
import math
import os
import numpy as np

from sgidspace.batch_processor import make_batch_processor, BatchProcessor
from sgidspace.sgikeras.models import patch_all, load_model
patch_all()
from sgidspace.sgikeras.metrics import precision, recall, fmeasure

from keras.models import Model
from sgidspace.architecture import build_network

from load_outputs import load_outputs

class InferenceEngine(object):
    def __init__(self, model, onehot_count, multihot_thresh, oprecision, eprecision, vector_form=False, skip_tasks=[], from_embed=False):
        if type(model) is str:
            self.model = load_model(
                model, custom_objects={'precision': precision,
                                       'recall': recall,
                                       'fmeasure': fmeasure}
            )
        else:
            self.model = model

        self.eauto_index = None

        self.model.outputs.append(self.model.get_layer('embedding').output)
        self.embedding_index = len(self.model.outputs) - 1

        self.model.outputs.append(self.model.get_layer('eauto3d').output)
        self.eauto_index = len(self.model.outputs) - 1

        self.onehot_count = onehot_count
        self.multihot_thresh = multihot_thresh
        self.oprecision = oprecision
        self.eprecision = eprecision
        self.vector_form = vector_form
        self.skip_tasks = skip_tasks
        self.from_embed = from_embed

        self.outputs = load_outputs('outputs.txt')

        if self.from_embed:
            inputs, outputs = build_network(self.outputs, from_embed=True)
            model2 = Model(inputs=inputs, outputs=outputs)
            for layer in model2.layers:
                try:
                    layer.set_weights(self.model.get_layer(layer.name).get_weights())
                except:
                    pass
            self.model = model2
            self.eauto_index = len(self.model.outputs) - 1

    def _signif(self, x, n=3):
        return np.around(x.astype(np.float64), n)

    def _filter_and_format_hot(self, output, probs, good_indices):
        # filter class labels based on good_indices and produce a list
        good_labels = [output['class_labels'][i] for i in good_indices]

        # filter probabilities based on good_indices and produce a list
        good_probs = self._signif(probs[good_indices], self.oprecision).tolist()

        return [{'id': i, 'prob': p} for i, p in zip(good_labels, good_probs)]

    def _format_probs(self, output, probs):
        if output['type'] == 'multihot':
            good_indices = np.where(probs >= self.multihot_thresh)[0]
            hits = self._filter_and_format_hot(output, probs, good_indices)
        elif output['type'] == 'onehot':
            ct = min(self.onehot_count, len(probs))
            good_indices = np.argpartition(-probs, range(ct))[:ct]
            hits = self._filter_and_format_hot(output, probs, good_indices)
        elif output['type'] in ('numeric'):
            hits = float(self._signif(probs[0], self.oprecision))
        elif output['type'] in ('boolean'):
            hits = float(self._signif(probs[1], self.oprecision))
        elif output['type'] == 'embedding_autoencoder':
            hits = self._signif(probs, self.oprecision).tolist()
        else:
            hits = self._signif(probs, self.oprecision).tolist()
            hits.append('other')

        return hits

    def generate(self, dataloader, vector_form=False):
        """
        Yield predictions!  Separating the dataloader from the inference engine
        allows us to reuse the inference engine on different data sets.
        """
        for X, records in dataloader:

            yhat = self.model.predict_on_batch(X)
            for recordi in xrange(len(records)):
                out = {}
                for yi in xrange(len(self.outputs)):
                    output = self.outputs[yi]
                    probs = yhat[yi][recordi, :]
                    if vector_form:
                        out[output['name']] = probs
                    else:
                        out[output['name']] = self._format_probs(output, probs)

                if not self.from_embed:
                    probs = self._signif(yhat[self.embedding_index][recordi, :], self.eprecision)
                    out['embedding'] = probs.tolist()

                if self.eauto_index != None:
                    probs = self._signif(yhat[self.eauto_index][recordi, :], self.eprecision)
                    out['embedding_auto'] = probs.tolist()

                # Wipe skipped outputs
                for o in self.skip_tasks:
                    if o in out:
                        del(out[o])


                records[recordi]['prediction'] = out
                for output in self.outputs:
                    if output['datafun'] is not None:
                        records[recordi].pop(output['name'], None)

                yield records[recordi]


def json_inference(blob, batch_size, inf):
    def query_gen(queries):
        for q in queries:
            yield q

    seq_gen = query_gen(blob)
    dataloader = BatchProcessor(seq_gen, batch_size, outputs=inf.outputs, inference=True, from_embed=from_embed)
    return inf.generate(dataloader)


def datadir_inference(datadir, batch_size, inf, from_embed=False):
    dataloader = make_batch_processor(
        datadir, '', batch_size, outputs=inf.outputs, inference=True, unbounded_iteration=False, from_embed=from_embed
    )
    return inf.generate(dataloader)


class ChunkedFileWriter(object):
    def __init__(self, prefix, chunksize, outdir):
        if chunksize <= 0:
            raise Exception('chunk size must be greater than or equal to 1!')
        self.chunksize = chunksize
        if not os.path.isdir(outdir):
            if os.path.exists(outdir):
                raise Exception('% already exists, but is not a directory!' % outdir)
            os.makedirs(outdir)
        self.prefix = os.path.join(outdir, prefix)

    def write_results(self, infgen):
        results = []
        for i, r in enumerate(infgen, 1):
            results.append(r)
            if i % self.chunksize == 0:
                self._write_chunk(results, int(i / self.chunksize))
                results = []

        # store the remainder
        if len(results) > 0:
            self._write_chunk(results, math.floor((i + self.chunksize) / self.chunksize))

    def _write_chunk(self, reslist, count):
        name = '%s-%06d.json.gz' % (self.prefix, count)
        with gzip.open(name, 'wb') as gzout:
            for r in reslist:
                # as jsonline
                gzout.write(json.dumps(r) + '\n')
        print 'wrote %s' % name


class StdoutWriter(object):
    def write_results(self, infgen):
        for r in infgen:
            print '%s\n' % json.dumps(r)


def parse_args():
    parser = argparse.ArgumentParser(description='Model inference.')

    parser.add_argument('model', help="Model file to use")
    parser.add_argument(
        'datadir', help="Data directory containing *.json.gz data to run inference on."
    )
    parser.add_argument('--skip', default=[], help='Tasks to skip', nargs='*')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size. (default=32)')
    parser.add_argument(
        '--onehot_count',
        type=int,
        default=5,
        help='Number of top hits to return for onehot outputs. (default=5)'
    )
    parser.add_argument(
        '--multihot_thresh',
        type=float,
        default=0.1,
        help='Threshold for multihot hits. (default=0.1)'
    )
    parser.add_argument(
        '--oprecision',
        type=int,
        default=3,
        help='Floating point precision in significant figures. (default=3)'
    )
    parser.add_argument(
        '--eprecision',
        type=int,
        default=6,
        help='Floating point precision for embedding in significant figures. (default=6)'
    )
    parser.add_argument('--prefix', default='inferred-output', help='Output file name prefix')
    parser.add_argument(
        '--chunksize',
        type=int,
        default=10000,
        help='Number of records to write to each output file'
    )
    parser.add_argument(
        '--outdir', default='.', help='The directory in which to store the output files'
    )
    parser.add_argument(
        '--stdout', action='store_true', help='Print records to stdout as jsonline'
    )
    parser.add_argument(
        '--from_embed', action='store_true', help='Run from "embedding" field rather than sequence'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    args.skip.append('embedding_autoencoder')

    inf = InferenceEngine(
        args.model, args.onehot_count, args.multihot_thresh, args.oprecision, args.eprecision, skip_tasks=args.skip, from_embed=args.from_embed
    )
    result_gen = datadir_inference(args.datadir, args.batch_size, inf, from_embed=args.from_embed)

    if args.stdout:
        writer = StdoutWriter()
    else:
        writer = ChunkedFileWriter(args.prefix, args.chunksize, args.outdir)

    writer.write_results(result_gen)


if __name__ == '__main__':
    main()
