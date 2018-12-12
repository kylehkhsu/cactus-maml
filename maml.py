""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
try:
    import special_grads
except KeyError as e:
    print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e,
          file=sys.stderr)

from collections import OrderedDict
from tensorflow.python.platform import flags
from utils import xent, conv_block, normalize, bn_relu_conv_block

FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output_train=1, dim_output_val=1, test_num_updates=5):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output_train = dim_output_train
        self.dim_output_val = dim_output_val
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_num_updates = test_num_updates
        self.loss_func = xent
        self.classification = True
        if FLAGS.on_encodings:
            print('Meta-learning on encodings')
            self.dim_hidden = [FLAGS.num_filters] * FLAGS.num_hidden_layers
            print('hidden layers: {}'.format(self.dim_hidden))
            self.forward = self.forward_fc
            self.construct_weights = self.construct_fc_weights
        else:
            if FLAGS.conv:
                self.dim_hidden = FLAGS.num_filters
                if FLAGS.resnet:
                    if FLAGS.input_type == 'images_84x84':
                        self.forward = self.forward_resnet84
                        self.construct_weights = self.construct_resnet_weights84
                        self.num_parts_per_res_block = FLAGS.num_parts_per_res_block
                        blocks = ['input']
                        for i in range(FLAGS.num_res_blocks):
                            blocks.append('res{}'.format(i))
                            if i != FLAGS.num_res_blocks - 1:
                                blocks.append('maxpool')
                        blocks.append('output')
                        self.blocks = blocks
                        print('blocks', self.blocks)
                    elif FLAGS.input_type == 'images_224x224':
                        self.forward = self.forward_resnet224
                        self.construct_weights = self.construct_resnet_weights224
                        assert FLAGS.num_parts_per_res_block == 2
                        assert FLAGS.num_res_blocks == 4
                        self.num_parts_per_res_block = FLAGS.num_parts_per_res_block
                        self.blocks = ['input', 'maxpool', 'res0', 'maxpool', 'res1', 'maxpool', 'res2', 'maxpool', 'res3', 'output']
                    else:
                        raise ValueError
                else:
                    self.forward = self.forward_conv
                    self.construct_weights = self.construct_conv_weights
            else:
                self.dim_hidden = [1024, 512, 256, 128]
                print('hidden layers: {}'.format(self.dim_hidden))
                self.forward = self.forward_fc
                self.construct_weights = self.construct_fc_weights
        if FLAGS.dataset == 'mnist' or FLAGS.dataset == 'omniglot':
            self.channels = 1
        else:
            self.channels = 3
        self.img_size = int(np.sqrt(self.dim_input/self.channels))
        if FLAGS.dataset not in ['mnist', 'omniglot', 'miniimagenet', 'celeba', 'imagenet']:
            raise ValueError('Unrecognized data source.')

        # resnet things


    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # a: training data for inner gradient, b: test data for meta gradient
        if prefix == 'metatrain_':
            inner_update_batch_size = FLAGS.inner_update_batch_size_train
        else:
            inner_update_batch_size = FLAGS.inner_update_batch_size_val
        outer_update_batch_size = FLAGS.outer_update_batch_size
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']
            if prefix == 'metaval_':
                self.mv_inputa = self.inputa
                self.mv_inputb = self.inputb
                self.mv_labela = self.labela
                self.mv_labelb = self.labelb

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                training_scope.reuse_variables()
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()
                print(weights.keys())

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas, lossesb, outputbs = [], [], [], []
            accuraciesa, accuraciesb = [], []
            num_updates = max(self.test_num_updates, FLAGS.num_updates)
            outputbs = [[]]*num_updates
            lossesb = [[]]*num_updates
            accuraciesb = [[]]*num_updates
            if FLAGS.from_scratch:
                train_accuracies = [[]]*num_updates

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch. """
                inputa, inputb, labela, labelb = inp
                task_outputbs, task_lossesb = [], []

                if FLAGS.from_scratch:
                    task_outputas = []

                if self.classification:
                    task_accuraciesb = []
                    if FLAGS.from_scratch:
                        task_accuraciesa = []


                task_outputa = self.forward(inputa, weights, prefix, reuse=reuse)  # only reuse on the first iter
                if FLAGS.from_scratch:
                    task_outputas.append(task_outputa)
                task_lossa = self.loss_func(task_outputa, labela, inner_update_batch_size)

                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))
                output = self.forward(inputb, fast_weights, prefix, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb, outer_update_batch_size))

                for j in range(num_updates - 1):
                    outputa = self.forward(inputa, fast_weights, prefix, reuse=True)
                    loss = self.loss_func(outputa, labela, inner_update_batch_size)
                    if FLAGS.from_scratch:
                        task_outputas.append(outputa)

                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.update_lr*gradients[key] for key in fast_weights.keys()]))
                    output = self.forward(inputb, fast_weights, prefix, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb, outer_update_batch_size))

                task_output = [task_outputa, task_outputbs, task_lossa, task_lossesb]

                if self.classification:
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1))
                    for j in range(num_updates):
                        task_accuraciesb.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputbs[j]), 1), tf.argmax(labelb, 1)))
                        if FLAGS.from_scratch:
                            task_accuraciesa.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputas[j]), 1), tf.argmax(labela, 1)))
                    task_output.extend([task_accuracya, task_accuraciesb])
                    if FLAGS.from_scratch:
                        task_output.extend([task_accuraciesa])

                return task_output

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            out_dtype = [tf.float32, [tf.float32]*num_updates, tf.float32, [tf.float32]*num_updates]
            if self.classification:
                out_dtype.extend([tf.float32, [tf.float32]*num_updates])
                if FLAGS.from_scratch:
                    out_dtype.extend([[tf.float32] * num_updates])
            result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)
            if self.classification:
                if FLAGS.from_scratch:
                    outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb, train_accuracies = result
                else:
                    outputas, outputbs, lossesa, lossesb, accuraciesa, accuraciesb = result
            else:
                outputas, outputbs, lossesa, lossesb  = result

        ## Performance & Optimization
        if 'train' in prefix:
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            # after the map_fn
            self.outputas, self.outputbs = outputas, outputbs
            if self.classification:
                self.total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.total_accuracies2 = total_accuracies2 = [tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                self.gvs = gvs = optimizer.compute_gradients(self.total_losses2[FLAGS.num_updates-1])
                if FLAGS.dataset == 'miniimagenet' or FLAGS.dataset == 'celeba' or FLAGS.dataset == 'imagenet':
                    gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
                self.metatrain_op = optimizer.apply_gradients(gvs)
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.to_float(FLAGS.meta_batch_size)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
            if self.classification:
                self.metaval_total_accuracy1 = total_accuracy1 = tf.reduce_sum(accuraciesa) / tf.to_float(FLAGS.meta_batch_size)
                self.metaval_total_accuracies2 = total_accuracies2 =[tf.reduce_sum(accuraciesb[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
                self.mv_outputbs = outputbs
                if FLAGS.from_scratch:
                    self.metaval_train_accuracies = [tf.reduce_sum(train_accuracies[j]) / tf.to_float(FLAGS.meta_batch_size) for j in range(num_updates)]
        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        if self.classification:
            tf.summary.scalar(prefix+'Pre-update accuracy', total_accuracy1)

        for j in range(num_updates):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            if self.classification:
                tf.summary.scalar(prefix+'Post-update accuracy, step ' + str(j+1), total_accuracies2[j])

    ### Network construction functions (fc networks and conv networks)
    def construct_fc_weights(self):
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output_train], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output_train]))
        return weights

    def forward_fc(self, inp, weights, prefix, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1,len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        logits = tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]
        if 'val' in prefix:
            logits = tf.gather(logits, tf.range(self.dim_output_val), axis=1)
        return logits

    def construct_conv_weights(self):
        weights = {}

        dtype = tf.float32
        conv_initializer =  tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        fc_initializer =  tf.contrib.layers.xavier_initializer(dtype=dtype)
        k = 3

        channels = self.channels
        weights['conv1'] = tf.get_variable('conv1', [k, k, channels, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv2'] = tf.get_variable('conv2', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b2'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv3'] = tf.get_variable('conv3', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b3'] = tf.Variable(tf.zeros([self.dim_hidden]))
        weights['conv4'] = tf.get_variable('conv4', [k, k, self.dim_hidden, self.dim_hidden], initializer=conv_initializer, dtype=dtype)
        weights['b4'] = tf.Variable(tf.zeros([self.dim_hidden]))
        if FLAGS.dataset == 'miniimagenet' or FLAGS.dataset == 'celeba' or FLAGS.dataset == 'imagenet':
            # assumes max pooling
            weights['w5'] = tf.get_variable('w5', [self.dim_hidden*5*5, self.dim_output_train], initializer=fc_initializer)
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output_train]), name='b5')
        else:
            weights['w5'] = tf.Variable(tf.random_normal([self.dim_hidden, self.dim_output_train]), name='w5')
            weights['b5'] = tf.Variable(tf.zeros([self.dim_output_train]), name='b5')

        return weights

    def forward_conv(self, inp, weights, prefix, reuse=False, scope=''):
        # reuse is for the normalization parameters.
        channels = self.channels
        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, channels])

        hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope+'0')
        hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope+'1')
        hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope+'2')
        hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope+'3')
        if FLAGS.dataset == 'miniimagenet' or FLAGS.dataset == 'celeba' or FLAGS.dataset == 'imagenet':
            # last hidden layer is 6x6x64-ish, reshape to a vector
            hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])
        else:
            hidden4 = tf.reduce_mean(hidden4, [1, 2])

        logits = tf.matmul(hidden4, weights['w5']) + weights['b5']

        if 'val' in prefix:
            logits = tf.gather(logits, tf.range(self.dim_output_val), axis=1)
        return logits

    def construct_resnet_weights224(self):
        weights = OrderedDict()
        dtype = tf.float32

        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        bias_initializer = tf.zeros_initializer(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        def make_conv_layer_weights(weights, scope, k, filters_in, filters_out, bias=True):
            weights['{}/conv'.format(scope)] = tf.get_variable('{}/conv'.format(scope), [k, k, filters_in, filters_out], initializer=conv_initializer, dtype=dtype)
            if bias:
                weights['{}/bias'.format(scope)] = tf.get_variable('{}/bias'.format(scope), [filters_out], initializer=bias_initializer, dtype=dtype)
        def make_fc_layer_weights(weights, scope, dims_in, dims_out):
            weights['{}/fc'.format(scope)] = tf.get_variable('{}/fc'.format(scope), [dims_in, dims_out], initializer=fc_initializer, dtype=dtype)
            weights['{}/bias'.format(scope)] = tf.get_variable('{}/bias'.format(scope), [dims_out], initializer=bias_initializer, dtype=dtype)
        for block_name in self.blocks:
            if block_name == 'input':
                make_conv_layer_weights(weights, block_name, k=7, filters_in=self.channels, filters_out=64)
            elif 'res' in block_name:
                j = int(block_name[-1])
                last_block_filter = 64 if j == 0 else 64 * 2 ** (j-1)
                this_block_filter = 64 if j == 0 else last_block_filter * 2
                print(block_name, last_block_filter, this_block_filter)
                make_conv_layer_weights(weights, '{}/shortcut'.format(block_name), k=1, filters_in=last_block_filter,
                                        filters_out=this_block_filter, bias=False)
                for i in range(self.num_parts_per_res_block):
                    make_conv_layer_weights(weights, '{}/part{}'.format(block_name, i), k=3,
                                            filters_in=last_block_filter if i == 0 else this_block_filter,
                                            filters_out=this_block_filter)
            elif block_name == 'output':
                make_fc_layer_weights(weights, block_name, dims_in=512, dims_out=self.dim_output_train)
        return weights

    def construct_resnet_weights84(self):
        weights = OrderedDict()
        dtype = tf.float32

        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
        bias_initializer = tf.zeros_initializer(dtype=dtype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
        def make_conv_layer_weights(weights, scope, k, filters_in, filters_out):
            weights['{}/conv'.format(scope)] = tf.get_variable('{}/conv'.format(scope), [k, k, filters_in, filters_out], initializer=conv_initializer, dtype=dtype)
            weights['{}/bias'.format(scope)] = tf.get_variable('{}/bias'.format(scope), [filters_out], initializer=bias_initializer, dtype=dtype)
        def make_fc_layer_weights(weights, scope, dims_in, dims_out):
            weights['{}/fc'.format(scope)] = tf.get_variable('{}/fc'.format(scope), [dims_in, dims_out], initializer=fc_initializer, dtype=dtype)
            weights['{}/bias'.format(scope)] = tf.get_variable('{}/bias'.format(scope), [dims_out], initializer=bias_initializer, dtype=dtype)
        for block_name in self.blocks:
            if block_name == 'input':
                make_conv_layer_weights(weights, block_name, k=3, filters_in=self.channels, filters_out=64)
            elif 'res' in block_name:
                j = int(block_name[-1])
                last_block_filter = 64 if j == 0 else 64 * 2 ** (j-1)
                this_block_filter = 64 if j == 0 else last_block_filter * 2
                print(block_name, last_block_filter, this_block_filter)
                for i in range(self.num_parts_per_res_block):
                    make_conv_layer_weights(weights, '{}/part{}'.format(block_name, i), k=3, filters_in=64, filters_out=64)
            elif block_name == 'output':
                make_fc_layer_weights(weights, block_name, dims_in=512, dims_out=self.dim_output_train)
        return weights

    def forward_resnet224(self, inp, weights, prefix, reuse=False):

        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])

        for block_name in self.blocks:
            if block_name == 'input':
                conv = weights['{}/conv'.format(block_name)]
                bias = weights['{}/bias'.format(block_name)]
                inp = tf.nn.conv2d(inp, filter=conv, strides=[1, 2, 2, 1], padding="SAME") + bias
            elif 'res' in block_name:
                shortcut = inp
                conv = weights['{}/shortcut/conv'.format(block_name)]
                shortcut = tf.nn.conv2d(input=shortcut, filter=conv, strides=[1, 1, 1, 1], padding="SAME")
                for part in range(self.num_parts_per_res_block):
                    part_name = 'part{}'.format(part)
                    scope = '{}/{}'.format(block_name, part_name)
                    conv = weights['{}/{}/conv'.format(block_name, part_name)]
                    bias = weights['{}/{}/bias'.format(block_name, part_name)]
                    inp = bn_relu_conv_block(inp=inp, conv=conv, bias=bias, reuse=reuse, scope=scope)
                inp = shortcut + inp
            elif 'maxpool' in block_name:
                inp = tf.nn.max_pool(inp, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
            elif 'output' in block_name:
                inp = tf.reduce_mean(inp, [1, 2])
                fc = weights['{}/fc'.format(block_name)]
                bias = weights['{}/bias'.format(block_name)]
                inp = tf.matmul(inp, fc) + bias
                if 'val' in prefix:
                    inp = tf.gather(inp, tf.range(self.dim_output_val), axis=1)
        return inp

    def forward_resnet84(self, inp, weights, prefix, reuse=False):

        inp = tf.reshape(inp, [-1, self.img_size, self.img_size, self.channels])

        for block_name in self.blocks:
            if block_name == 'input':
                conv = weights['{}/conv'.format(block_name)]
                bias = weights['{}/bias'.format(block_name)]
                inp = tf.nn.conv2d(inp, filter=conv, strides=[1, 1, 1, 1], padding="SAME") + bias

            elif 'res' in block_name:
                shortcut = inp
                for part in range(self.num_parts_per_res_block):
                    part_name = 'part{}'.format(part)
                    scope = '{}/{}'.format(block_name, part_name)
                    conv = weights['{}/{}/conv'.format(block_name, part_name)]
                    bias = weights['{}/{}/bias'.format(block_name, part_name)]
                    inp = bn_relu_conv_block(inp=inp, conv=conv, bias=bias, reuse=reuse, scope=scope)
                inp = shortcut + inp
            elif 'maxpool' in block_name:
                inp = tf.nn.max_pool(inp, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")
            elif 'output' in block_name:
                inp = tf.reduce_mean(inp, [1, 2])
                fc = weights['{}/fc'.format(block_name)]
                bias = weights['{}/bias'.format(block_name)]
                inp = tf.matmul(inp, fc) + bias
                if 'val' in prefix:
                    inp = tf.gather(inp, tf.range(self.dim_output_val), axis=1)
        return inp

    def wrap(self, inp, weights, prefix, reuse=False, scope=''):
        unused = self.forward_resnet(inp, weights, prefix, reuse=False)
        return self.forward_resnet(inp, weights, prefix, reuse=True)


if __name__ == '__main__':
    import ipdb

    FLAGS = flags.FLAGS

    ## Dataset/method options
    flags.DEFINE_string('dataset', 'omniglot', 'omniglot or mnist or miniimagenet or celeba')
    flags.DEFINE_integer('num_encoding_dims', -1, 'of unsupervised representation learning method')
    flags.DEFINE_string('encoder', 'acai', 'acai or bigan or deepcluster or infogan')

    ## Training options
    flags.DEFINE_integer('metatrain_iterations', 30000, 'number of metatraining iterations.')
    flags.DEFINE_integer('meta_batch_size', 8, 'number of tasks sampled per meta-update')
    flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
    flags.DEFINE_float('update_lr', 0.05, 'step size alpha for inner gradient update.')
    flags.DEFINE_integer('inner_update_batch_size_train', 1,
                         'number of examples used for inner gradient update (K for K-shot learning).')
    flags.DEFINE_integer('inner_update_batch_size_val', 5, 'above but for meta-val')
    flags.DEFINE_integer('outer_update_batch_size', 5, 'number of examples used for outer gradient update')
    flags.DEFINE_integer('num_updates', 5, 'number of inner gradient updates during training.')
    flags.DEFINE_string('mt_mode', 'gtgt', 'meta-training mode (for sampling, labeling): gtgt or encenc')
    flags.DEFINE_string('mv_mode', 'gtgt', 'meta-validation mode (for sampling, labeling): gtgt or encenc')
    flags.DEFINE_integer('num_classes_train', 5, 'number of classes used in classification for meta-training')
    flags.DEFINE_integer('num_classes_val', 5, 'number of classes used in classification for meta-validation.')
    flags.DEFINE_float('margin', 0.0, 'margin for generating partitions using random hyperplanes')
    flags.DEFINE_integer('num_partitions', 1, 'number of partitions, -1 for same as number of meta-training tasks')
    flags.DEFINE_string('partition_algorithm', 'kmeans', 'hyperplanes or kmeans')
    flags.DEFINE_integer('num_clusters', -1, 'number of clusters for kmeans')
    flags.DEFINE_boolean('scaled_encodings', True, 'if True, use randomly scaled encodings for kmeans')
    flags.DEFINE_boolean('on_encodings', False, 'if True, train MAML on top of encodings')
    flags.DEFINE_integer('num_hidden_layers', 2, 'number of mlp hidden layers')
    flags.DEFINE_integer('num_parallel_calls', 8, 'for loading data')
    flags.DEFINE_integer('gpu', 7, 'CUDA_VISIBLE_DEVICES=')

    ## Model options
    flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
    flags.DEFINE_integer('num_filters', 32, 'number of filters for each conv layer')
    flags.DEFINE_bool('conv', True, 'whether or not to use a convolutional network')
    flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
    flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

    ## Logging, saving, and testing options
    flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
    flags.DEFINE_string('logdir', './log', 'directory for summaries and checkpoints.')
    flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
    flags.DEFINE_bool('train', True, 'True to train, False to test.')
    flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
    flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
    flags.DEFINE_integer('log_inner_update_batch_size_val', -1,
                         'specify log directory iubsv. (use to test with different iubsv)')
    flags.DEFINE_float('train_update_lr', -1,
                       'value of inner gradient step step during training. (use if you want to test with a different value)')
    flags.DEFINE_bool('save_checkpoints', False, 'if True, save model weights as checkpoints')
    flags.DEFINE_bool('debug', False, 'if True, use tf debugger')
    flags.DEFINE_string('suffix', '', 'suffix for an exp_string')
    flags.DEFINE_bool('from_scratch', False, 'fast-adapt from scratch')
    flags.DEFINE_integer('num_eval_tasks', 1000, 'number of tasks to meta-test on')

    # Imagenet
    flags.DEFINE_string('input_type', 'images_84x84',
                        'features or features_processed or images_fullsize or images_84x84')
    flags.DEFINE_string('data_dir', '/data3/kylehsu/data', 'location of data')
    flags.DEFINE_bool('resnet', False, 'use resnet architecture')
    flags.DEFINE_integer('num_res_blocks', 5, 'number of resnet blocks')
    flags.DEFINE_integer('num_parts_per_res_block', 2, 'number of bn-relu-conv parts in a res block')

    FLAGS.resnet = True

    maml = MAML(dim_input=3*84*84, dim_output_train=10, dim_output_val=5, test_num_updates=5)
    maml.channels = 3
    maml.img_size = 84
    weights = maml.construct_resnet_weights()
    input_ph = tf.placeholder(tf.float32)
    unused = maml.forward_resnet(input_ph, weights, 'hi', reuse=False)




    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    input = np.ones((1, 84 * 84 * 3), dtype=np.float32)

    y = sess.run(maml.forward_resnet(input_ph, weights, 'val', reuse=True), {input_ph: input})


    ipdb.set_trace()
    x=1





