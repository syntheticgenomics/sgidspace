# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import sgidspace.sgikeras
import keras

import warnings
import json
import os
import numpy as np

from keras import backend as K
from keras import optimizers
from keras.utils.io_utils import ask_to_proceed_with_overwrite
from keras.engine import topology
from keras.legacy import models as legacy_models
from keras.models import model_from_config, Sequential

try:
    import h5py
except ImportError:
    h5py = None


def save_model(model, filepath, overwrite=True, include_optimizer=True):
    """Save a model to a HDF5 file.

    The saved model contains:
        - the model's configuration (topology)
        - the model's weights
        - the model's optimizer's state (if any)
        - the model's metadata

    Thus the saved model can be reinstantiated in
    the exact same state, without any of the code
    used for model definition or training.

    # Arguments
        model: Keras model instance to be saved.
        filepath: String, path where to save the model.
        overwrite: Whether we should overwrite any existing
            model at the target location, or instead
            ask the user with a manual prompt.
        include_optimizer: If True, save optimizer's state together.

    # Raises
        ImportError: if h5py is not available.
    """

    if h5py is None:
        raise ImportError('`save_model` requires h5py.')

    def get_json_type(obj):
        """Serialize any object to a JSON-serializable structure.

        # Arguments
            obj: the object to serialize

        # Returns
            JSON-serializable structure representing `obj`.

        # Raises
            TypeError: if `obj` cannot be serialized.
        """
        # if obj is a serializable Keras class instance
        # e.g. optimizer, layer
        if hasattr(obj, 'get_config'):
            return {
                'class_name': obj.__class__.__name__,
                'config': obj.get_config(),
            }

        # if obj is any numpy type
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return {
                    'type': type(obj),
                    'value': obj.tolist(),
                }
            else:
                return obj.item()

        # misc functions (e.g. loss function)
        if callable(obj):
            return obj.__name__

        # if obj is a python 'type'
        if type(obj).__name__ == type.__name__:
            return obj.__name__

        raise TypeError('Not JSON Serializable:', obj)

    from keras import __version__ as keras_version

    # If file exists and should not be overwritten.
    if not overwrite and os.path.isfile(filepath):
        proceed = ask_to_proceed_with_overwrite(filepath)
        if not proceed:
            return

    f = h5py.File(filepath, 'w')
    f.attrs['keras_version'] = str(keras_version).encode('utf8')
    f.attrs['backend'] = K.backend().encode('utf8')
    f.attrs['model_config'] = json.dumps({
        'class_name': model.__class__.__name__,
        'config': model.get_config()
    },
                                         default=get_json_type).encode('utf8')

    if hasattr(model, 'metadata'):
        f.attrs['metadata'] = json.dumps(model.metadata).encode('utf8')

    model_weights_group = f.create_group('model_weights')
    if legacy_models.needs_legacy_support(model):
        model_layers = legacy_models.legacy_sequential_layers(model)
    else:
        model_layers = model.layers
    topology.save_weights_to_hdf5_group(model_weights_group, model_layers)

    if include_optimizer and hasattr(model, 'optimizer'):
        if isinstance(model.optimizer, optimizers.TFOptimizer):
            warnings.warn(
                'TensorFlow optimizers do not '
                'make it possible to access '
                'optimizer attributes or optimizer state '
                'after instantiation. '
                'As a result, we cannot save the optimizer '
                'as part of the model save file.'
                'You will have to compile your model again after loading it. '
                'Prefer using a Keras optimizer instead '
                '(see keras.io/optimizers).'
            )
        else:
            f.attrs['training_config'] = json.dumps({
                'optimizer_config': {
                    'class_name': model.optimizer.__class__.__name__,
                    'config': model.optimizer.get_config()
                },
                'loss': model.loss,
                'metrics': model.metrics,
                'sample_weight_mode': model.sample_weight_mode,
                'loss_weights': model.loss_weights,
            },
                                                    default=get_json_type).encode('utf8')

            # Save optimizer weights.
            symbolic_weights = getattr(model.optimizer, 'weights')
            if symbolic_weights:
                optimizer_weights_group = f.create_group('optimizer_weights')
                weight_values = K.batch_get_value(symbolic_weights)
                weight_names = []
                for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
                    # Default values of symbolic_weights is /variable for theano and cntk
                    if K.backend() == 'theano' or K.backend() == 'cntk':
                        if hasattr(w, 'name') and w.name != "/variable":
                            name = str(w.name)
                        else:
                            name = 'param_' + str(i)
                    else:
                        if hasattr(w, 'name') and w.name:
                            name = str(w.name)
                        else:
                            name = 'param_' + str(i)
                    weight_names.append(name.encode('utf8'))
                optimizer_weights_group.attrs['weight_names'] = weight_names
                for name, val in zip(weight_names, weight_values):
                    param_dset = optimizer_weights_group.create_dataset(
                        name, val.shape, dtype=val.dtype
                    )
                    if not val.shape:
                        # scalar
                        param_dset[()] = val
                    else:
                        param_dset[:] = val
    f.flush()
    f.close()


def load_model(filepath, custom_objects=None, compile=True):
    """Loads a model saved via `save_model`.

    # Arguments
        filepath: String, path to the saved model.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model
            after loading.

    # Returns
        A Keras model instance. If an optimizer was found
        as part of the saved model, the model is already
        compiled. Otherwise, the model is uncompiled and
        a warning will be displayed. When `compile` is set
        to False, the compilation is omitted without any
        warning.

    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')

    if not custom_objects:
        custom_objects = {}

    def convert_custom_objects(obj):
        """Handles custom object lookup.

        # Arguments
            obj: object, dict, or list.

        # Returns
            The same structure, where occurrences
                of a custom object name have been replaced
                with the custom object.
        """
        if isinstance(obj, list):
            deserialized = []
            for value in obj:
                deserialized.append(convert_custom_objects(value))
            return deserialized
        if isinstance(obj, dict):
            deserialized = {}
            for key, value in obj.items():
                deserialized[key] = convert_custom_objects(value)
            return deserialized
        if obj in custom_objects:
            return custom_objects[obj]
        return obj

    with h5py.File(filepath, mode='r') as f:
        # instantiate model
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model found in config file.')
        model_config = json.loads(model_config.decode('utf-8'))
        model = model_from_config(model_config, custom_objects=custom_objects)

        # set weights
        topology.load_weights_from_hdf5_group(f['model_weights'], model.layers)

        # Early return if compilation is not required.
        if not compile:
            return model

        # instantiate optimizer
        training_config = f.attrs.get('training_config')
        if training_config is None:
            warnings.warn(
                'No training configuration found in save file: '
                'the model was *not* compiled. Compile it manually.'
            )
            return model
        training_config = json.loads(training_config.decode('utf-8'))
        optimizer_config = training_config['optimizer_config']
        optimizer = optimizers.deserialize(optimizer_config, custom_objects=custom_objects)

        # Recover loss functions and metrics.
        loss = convert_custom_objects(training_config['loss'])
        metrics = convert_custom_objects(training_config['metrics'])
        sample_weight_mode = training_config['sample_weight_mode']
        loss_weights = training_config['loss_weights']

        # Compile model.
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights,
            sample_weight_mode=sample_weight_mode
        )

        # Set optimizer weights.
        if 'optimizer_weights' in f:
            # Build train function (to get weight updates).
            if isinstance(model, Sequential):
                model.model._make_train_function()
            else:
                model._make_train_function()
            optimizer_weights_group = f['optimizer_weights']
            optimizer_weight_names = [
                n.decode('utf8') for n in optimizer_weights_group.attrs['weight_names']
            ]
            optimizer_weight_values = [optimizer_weights_group[n] for n in optimizer_weight_names]
            try:
                model.optimizer.set_weights(optimizer_weight_values)
            except ValueError:
                warnings.warn(
                    'Error in loading the saved optimizer '
                    'state. As a result, your model is '
                    'starting with a freshly initialized '
                    'optimizer.'
                )

        metadata = f.attrs.get('metadata')
        if metadata is None:
            warnings.warn('No metadata found for model.')
        else:
            model.metadata = json.loads(metadata.decode('utf-8'))

    return model


def patch_all():
    keras.models.save_model = sgidspace.sgikeras.models.save_model
