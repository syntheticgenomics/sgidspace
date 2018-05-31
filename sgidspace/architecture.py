from keras.layers import Input, Conv1D, Dense, MaxPooling1D, Flatten, Activation, Reshape, Subtract
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras import backend as K
import numpy as np
from sgidspace.sequence_generator import IUPAC_CODES


def build_network(outputs, from_embed=False):
    
    # Inputs
    if from_embed:
        with tf.variable_scope('embedding_input') as scope:
            embedding = Input(shape=[256], dtype='float32', name='embedding')
    else:
        with tf.variable_scope('sequence_input') as scope:
            seq_input = Input(shape=(2000, len(IUPAC_CODES)), dtype='float32', name='sequence_input')

    # Encoder
    with tf.variable_scope('encoder') as scope:
        if not from_embed:
            x = Conv1D(16, 3, input_shape=(None, 2000, len(IUPAC_CODES)), activation='elu', padding='same', name='encoder_conv0', kernel_initializer='he_uniform')(seq_input)
            x = BatchNormalization(name='encoder_bn0')(x)
            x = Conv1D(24, 3, activation='elu', padding='same', name='encoder_conv1', kernel_initializer='he_uniform')(x)
            x = BatchNormalization(name='encoder_bn1')(x)
            x = MaxPooling1D()(x)
        
            x = Conv1D(32, 5, activation='elu', padding='same', name='encoder_conv2', kernel_initializer='he_uniform')(x)
            x = BatchNormalization(name='encoder_bn2')(x)
            x = Conv1D(48, 5, activation='elu', padding='same', name='encoder_conv3', kernel_initializer='he_uniform')(x)
            x = BatchNormalization(name='encoder_bn3')(x)
            x = MaxPooling1D()(x)
        
            x = Conv1D(64, 7, activation='elu', padding='same', name='encoder_conv4', kernel_initializer='he_uniform')(x)
            x = BatchNormalization(name='encoder_bn4')(x)
            x = Conv1D(96, 7, activation='elu', padding='same', name='encoder_conv5', kernel_initializer='he_uniform')(x)
            x = BatchNormalization(name='encoder_bn5')(x)
            x = MaxPooling1D()(x)

            x = Flatten()(x)
            x = Dense(2048, activation='elu', kernel_initializer='he_uniform', name='encoder_aff0')(x)
            x = BatchNormalization(name='encoder_bn6')(x)
            x = Dense(1024, activation='elu', kernel_initializer='he_uniform', name='encoder_aff1')(x)
            x = BatchNormalization(name='encoder_bn7')(x)
            x = Dense(512, activation='elu', kernel_initializer='he_uniform', name='encoder_aff2')(x)
            x = BatchNormalization(name='encoder_bn8')(x)
            x = Dense(256, activation='elu', kernel_initializer='he_uniform', name='encoder_aff3')(x)
            embedding = BatchNormalization(name='embedding')(x)

        x = Dense(128, activation='elu', kernel_initializer='he_uniform', name='embed_auto0')(embedding)
        x = BatchNormalization(name='embed_auto_bn0')(x)
        x = Dense(64, activation='elu', kernel_initializer='he_uniform', name='embed_auto1')(x)
        x = BatchNormalization(name='embed_auto_bn1')(x)
        x = Dense(32, activation='elu', kernel_initializer='he_uniform', name='embed_auto2')(x)
        x = BatchNormalization(name='embed_auto_bn2')(x)
        x = Dense(16, activation='elu', kernel_initializer='he_uniform', name='embed_auto3')(x)
        x = BatchNormalization(name='embed_auto_bn3')(x)
        x = Dense(8, activation='elu', kernel_initializer='he_uniform', name='embed_auto4')(x)
        x = BatchNormalization(name='embed_auto_bn4')(x)
        x = Dense(3, activation='elu', kernel_initializer='he_uniform', name='embed_auto5')(x)
        x = BatchNormalization(name='eauto3d')(x)
        x = Dense(8, activation='elu', kernel_initializer='he_uniform', name='embed_auto6')(x)
        x = BatchNormalization(name='embed_auto_bn6')(x)
        x = Dense(16, activation='elu', kernel_initializer='he_uniform', name='embed_auto7')(x)
        x = BatchNormalization(name='embed_auto_bn7')(x)
        x = Dense(32, activation='elu', kernel_initializer='he_uniform', name='embed_auto8')(x)
        x = BatchNormalization(name='embed_auto_bn8')(x)
        x = Dense(64, activation='elu', kernel_initializer='he_uniform', name='embed_auto9')(x)
        x = BatchNormalization(name='embed_auto_bn9')(x)
        x = Dense(128, activation='elu', kernel_initializer='he_uniform', name='embed_auto10')(x)
        x = BatchNormalization(name='embed_auto_bn10')(x)
        x = Dense(256, activation=None, kernel_initializer='he_uniform', name='embedding_autoencoder_out')(x)

        eauto_diff = Subtract(name='embedding_autoencoder')([x, embedding])

    # Decoders
    output_layers = []
    with tf.variable_scope('decoder') as scope:
        for o in outputs:
            if o['name'] == 'embedding_autoencoder':
                output_layers.append(eauto_diff)
            else:
                output_layers.append(generic_decoder(embedding, o))

    if from_embed:
        return [embedding], output_layers
    else:
        return [seq_input], output_layers


def generic_decoder(embedding, output):

    if output['name'] in ['diamond_cluster_id', 'translation_tax_id_dict', 'gene_name']:
        embedding = Dense(
            36, activation='elu', name='post_embedding_affine_' + output['name'], kernel_initializer='he_uniform'
        )(embedding)
        embedding = BatchNormalization(name='post_embedding_affine_' + output['name'] + '_bn')(embedding)

    if output['type'] in ['onehot']:
        x = Dense(
            output['classcount'], activation='softmax', name=output['name']
        )(embedding)
    elif output['type'] in ['multihot']:
        x = Dense(
            output['classcount'], activation='sigmoid', name=output['name']
        )(embedding)
    elif output['type'] == 'boolean':
        x = Dense(2, activation='softmax', name=output['name'])(embedding)
    elif output['type'] == 'numeric':
        x = Dense(1, activation=None, name=output['name'])(embedding)
    else:
        raise ValueError('Output type "' + str(output['type']) + '" is not supported.')

    return x

