""" Utility functions. """
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
from collections import defaultdict
FLAGS = flags.FLAGS


# Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images


# Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope)
    if FLAGS.max_pool:
        normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
    return normed


def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp


# Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))


def xent(pred, label, update_batch_size):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / update_batch_size


# Data helpers
def get_data(dataset, num_encoding_dims, test_set):
    """
    Assumes each split of a dataset is in a .npz file with keys 'X', 'Y', and 'Z' corresponding to uint8 images, labels, and float32 encodings, respectively.
    """
    splits = ['train', 'val', 'test']
    print('Encoder: {}'.format(FLAGS.encoder))
    if FLAGS.encoder == 'bigan':
        data_folder = './data/bigan_encodings'
        filenames = {split: os.path.join(data_folder, '{}.u-{}_{}.npz'.format(dataset, num_encoding_dims, split))
                     for split in splits}
    elif FLAGS.encoder == 'infogan':
        data_folder = './data/infogan_encodings'
        filenames = {split: os.path.join(data_folder, '{}.{}_{}.npz'.format(dataset, num_encoding_dims, split))
                     for split in splits}
    elif FLAGS.encoder == 'acai':
        data_folder = './data/acai_encodings'
        filenames = {split: os.path.join(data_folder, '{}_{}_{}.npz'.format(dataset, num_encoding_dims, split))
                     for split in splits}
    elif FLAGS.encoder == 'deepcluster':
        print('Deep cluster embeddings are whitened and normalized already!')
        data_folder = './data/deepcluster_encodings'
        filenames = {split: os.path.join(data_folder, '{}_{}_{}.npz'.format(dataset, num_encoding_dims, split))
                     for split in splits}
    else:
        raise NotImplementedError

    def get_XYZ(filename):
        data = np.load(filename)
        if FLAGS.encoder == 'infogan':
            X, Y, Z = data['X'], data['Y'], data['Z_raw']
        else:
            X, Y, Z = data['X'], data['Y'], data['Z']
        return X, Y, Z
    X_train, Y_train, Z_train = get_XYZ(filenames['train'])
    X_val, Y_val, Z_val = get_XYZ(filenames['val'])
    X_test, Y_test, Z_test = get_XYZ(filenames['test'])

    if dataset == 'celeba':
        # Assumes Y contains celeba filenames.
        if type(Y_train[0]) == np.bytes_:
            Y_train = np.array([int(y.decode('utf-8')[:y.decode('utf-8').find('.jpg')]) for y in Y_train])
            Y_val = np.array([int(y.decode('utf-8')[:y.decode('utf-8').find('.jpg')]) for y in Y_val])
            Y_test = np.array([int(y.decode('utf-8')[:y.decode('utf-8').find('.jpg')]) for y in Y_test])

        n_train_attributes = 20
        n_val_attributes = 10
        n_test_attributes = 10
        name_to_attributes = defaultdict(int)
        with open('./data/celeba/cropped/Anno/list_attr_celeba.txt') as f:
            # first two lines of dataset files are expect to
            # be #examples and names of the attributes
            n_datapoints = int(f.readline().strip())
            attribute_names = f.readline().strip().split()
            for i, lines in enumerate(f):
                example_name, *example_attributes = lines.strip().split()
                example_name = int(example_name[:example_name.find('.jpg')])
                name_to_attributes[example_name] = np.array(list(map(lambda a: 0 if int(a) < 0 else 1, example_attributes)))
            print([(i, name) for (i, name) in enumerate(attribute_names)])

        def split_attributes(names):
            attributes = []
            for name in names:
                attributes.append(name_to_attributes[name])
            attributes = np.stack(attributes, axis=0)
            return attributes

        [attributes_train, attributes_val, attributes_test] = map(split_attributes, [Y_train, Y_val, Y_test])
        attributes_train = attributes_train[:, 0:n_train_attributes]
        attributes_val = attributes_val[:, n_train_attributes:n_train_attributes+n_val_attributes]
        attributes_test = attributes_test[:, -n_test_attributes:]

        i = 500
        assert np.all(name_to_attributes[Y_test[i]][-n_test_attributes:] == attributes_test[i])
        assert np.all(name_to_attributes[Y_val[i]][n_train_attributes:n_train_attributes+n_val_attributes] == attributes_val[i])
        Y_train, Y_val, Y_test = attributes_train, attributes_val, attributes_test

    if not test_set:   # use val as test
        X_test, Y_test, Z_test = X_val, Y_val, Z_val
    
    return X_train, Y_train, Z_train, X_test, Y_test, Z_test
