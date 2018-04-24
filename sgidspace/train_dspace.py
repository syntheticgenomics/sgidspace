import argparse
import json
import os
import subprocess

from sgidspace.batch_processor import make_batch_processor
from sgidspace.architecture import build_network
import tensorflow as tf
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.utils import plot_model
from keras.optimizers import Nadam

# Monkey patching
from sgidspace.sgikeras.models import patch_all, load_model
patch_all()
from sgidspace.sgikeras.metrics import precision, recall, fmeasure

import keras.backend as K
import tensorflow as tf
import numpy as np

from datetime import datetime

from load_outputs import load_outputs

start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_callbacks(outdir):
    # Get callbacks

    tensorboard_callback = TensorBoard(
        log_dir=outdir + '/tfboard/run-' + start_time,
        batch_size=1024,
    )

    best_checkpoint_callback = ModelCheckpoint(
        '%s/model.run-%s.best.hdf5' % (outdir, start_time),
        monitor='val_loss',
        save_best_only=True,
        mode='auto',
        period=1,
        verbose=1
    )

    latest_checkpoint_callback = ModelCheckpoint(
        '%s/model.run-%s.latest.hdf5' % (outdir, start_time),
        monitor='val_loss',
        save_best_only=False,
        mode='auto',
        period=1
    )

    callbacks = [
        tensorboard_callback,
        best_checkpoint_callback,
        latest_checkpoint_callback,
    ]

    return callbacks


def git_hash():
    dspace_dir = os.path.abspath(os.path.dirname(__file__) + "../../")
    try:
        git_hash = subprocess.check_output([
            "git", "--git-dir", dspace_dir + "/.git", "--work-tree", dspace_dir, "rev-parse", "HEAD"
        ]).strip()
        print('git hash:', git_hash)
    except subprocess.CalledProcessError:
        git_hash = 'git not available'
    return git_hash


def build_model(outputs):
    """
    Main function that calls layers within the architecture module
    """
    # Get run info
    input_layers, output_layers = build_network(outputs)
    model = Model(inputs=input_layers, outputs=output_layers)

    param_count = model.count_params()
    seed = 123456
    np.random.seed(seed)

    model.metadata = {
        'git_hash': git_hash(),
        'start_time': start_time,
        'param_count': param_count,
        'seed': seed
    }

    print("Number of parameters: " + str(param_count))

    losses = {}
    for o in outputs:
        print(o['name'])
        if o['type'] in ['multihot']:
            losses[o['name']] = 'binary_crossentropy'
        elif o['type'] in ['onehot', 'boolean', 'positional']:
            losses[o['name']] = 'categorical_crossentropy'
        elif o['type'] in ['numeric', 'embedding_autoencoder']:
            losses[o['name']] = 'mean_squared_error'

    metrics = {}
    for o in outputs:
        if o['type'] == 'onehot':
            metrics[o['name']] = ['top_k_categorical_accuracy', 'categorical_accuracy']
        elif o['type'] == 'boolean':
            metrics[o['name']] = 'categorical_accuracy'
        elif o['type'] == 'multihot':
            metrics[o['name']] = [precision, recall, fmeasure]
        elif o['type'] in ['numeric', 'embedding_autoencoder']:
            metrics[o['name']] = 'mean_squared_error'

    optimizer = Nadam(lr=0.001)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=metrics,
        loss_weights={o['name']: o['scale'] for o in outputs}
    )

    return model


def change_gpu_visibility(n_gpus):
    # change GPU visibility
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        currently_visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    else:
        currently_visible = []
    not_currently_visible = list(set("01234567").difference(set(currently_visible)))
    gpu_ids = ",".join([
        currently_visible[i]
        if i < len(currently_visible) else not_currently_visible[i - len(currently_visible)]
        for i in range(n_gpus)
    ])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


def train_model(
        epochs,
        main_datadir,
        outdir,
        outputs,
        n_gpus=1,
        output_subdirectory=None
):
    """
    Main function for setting up model and running training
    """
    change_gpu_visibility(n_gpus)

    # add subdirectory to output
    subdirectory = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    if output_subdirectory is not None:
        subdirectory = subdirectory + '_' + output_subdirectory
    outdir = os.path.join(outdir, subdirectory)
    os.system('mkdir -p {}'.format(outdir))
    print('output directory:', outdir)

    # Get class weights
    class_weights = []
    for o in outputs:
        if o['type'] in ('onehot', 'multihot'):
            class_weights.append({
                i: min(1, max(0.01, 1 / (w * o['classcount'])))
                for i, w in enumerate(o['class_freq'])
            })
        else:
            class_weights.append(None)

    # Get the dataloaders
    dataloader_train = make_batch_processor(
        main_datadir, 'train', 512, outputs
    )
    dataloader_validation = make_batch_processor(
        main_datadir, 'val', 512, outputs
    )

    model = build_model(outputs)

    # Draw model graph
    plot_model(model, to_file=outdir + '/model.png', show_shapes=True)

    training_start_time = datetime.now()
    print("Started training at: " + training_start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # Draw model graph
    plot_model(model, to_file=outdir + '/model.png', show_shapes=True)

    # for mbatch in xrange(0, epochs):
    model.fit_generator(
        dataloader_train,
        steps_per_epoch=1000,
        workers=1,
        validation_data=dataloader_validation,
        validation_steps=100,
        use_multiprocessing=False,
        callbacks=get_callbacks(outdir),
        epochs=epochs
    )
    training_end_time = datetime.now()
    print("Ended training at: " + training_end_time.strftime("%Y-%m-%d %H:%M:%S"))


def main():

    # Construct argument parser
    parser = argparse.ArgumentParser(description='DSPACE Model training.')
    parser.add_argument(
        '-w', '--datadir', type=str, default="data", help="Main data directory for all outputs"
    )
    parser.add_argument(
        '-o', '--output', default='/output', help='output path to write output files to'
    )
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()

    # Construct outputs
    outputs = load_outputs(os.path.abspath(os.path.join(os.path.dirname(__file__), 'outputs.txt')))

    # Train model
    train_model(args.epochs, args.datadir, args.output, outputs)


if __name__ == '__main__':
    main()
