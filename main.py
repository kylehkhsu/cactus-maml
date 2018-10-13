"""
    See scripts/ for illustrative examples of usage.
"""
import csv
import numpy as np
import random
import tensorflow as tf

from data_generator import DataGenerator
from maml import MAML
from tensorflow.python.platform import flags
from tensorflow.python import debug as tf_debug
from tqdm import tqdm
import os

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'omniglot', 'omniglot or mnist or miniimagenet or celeba')
flags.DEFINE_integer('num_encoding_dims', -1, 'of unsupervised representation learning method')
flags.DEFINE_string('encoder', 'acai', 'acai or bigan or deepcluster or infogan')

## Training options
flags.DEFINE_integer('metatrain_iterations', 30000, 'number of metatraining iterations.')
flags.DEFINE_integer('meta_batch_size', 8, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_float('update_lr', 0.05, 'step size alpha for inner gradient update.')
flags.DEFINE_integer('inner_update_batch_size_train', 1, 'number of examples used for inner gradient update (K for K-shot learning).')
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
flags.DEFINE_integer('log_inner_update_batch_size_val', -1, 'specify log directory iubsv. (use to test with different iubsv)')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step step during training. (use if you want to test with a different value)')
flags.DEFINE_bool('save_checkpoints', False, 'if True, save model weights as checkpoints')
flags.DEFINE_bool('debug', False, 'if True, use tf debugger')
flags.DEFINE_string('suffix', '', 'suffix for an exp_string')
flags.DEFINE_bool('from_scratch', False, 'fast-adapt from scratch')
flags.DEFINE_integer('num_eval_tasks', 1000, 'number of tasks to meta-test on')

logdir = FLAGS.logdir

def train(model, saver, sess, exp_string, data_generator, resume_itr=0):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 10000
    PRINT_INTERVAL = 100
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5

    if FLAGS.log:
        train_writer = tf.summary.FileWriter(logdir + '/' + exp_string, graph=None)     # omitting the graph
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    for itr in tqdm(range(resume_itr, FLAGS.metatrain_iterations), 'meta-training'):
        feed_dict = {}
        input_tensors = [model.metatrain_op]

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates-1]])

        result = sess.run(input_tensors, feed_dict)

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
                train_writer.flush()
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            print_str = 'Iteration ' + str(itr)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            tqdm.write(print_str)
            prelosses, postlosses = [], []

        if FLAGS.save_checkpoints and (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, logdir + '/' + exp_string + '/model' + str(itr))

        if (itr!=0) and itr % TEST_PRINT_INTERVAL == 0:
            feed_dict = {}
            if model.classification:
                input_tensors = [model.metaval_total_accuracy1, model.metaval_total_accuracies2[FLAGS.num_updates-1], model.summ_op]
            else:
                input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates-1], model.summ_op]

            result = sess.run(input_tensors, feed_dict)
            tqdm.write('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

    saver.save(sess, logdir + '/' + exp_string +  '/model' + str(itr+1))


NUM_TEST_POINTS = FLAGS.num_eval_tasks

def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes_train  # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []
    ensemble_accuracies, no_ensemble_accuracies = [], []

    for _ in tqdm(range(NUM_TEST_POINTS), 'meta-testing'):
        feed_dict = {model.meta_lr : 0.0}
        if FLAGS.from_scratch:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2 + model.metaval_train_accuracies, feed_dict)
        else:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)

    if FLAGS.from_scratch:
        train_accuracies = metaval_accuracies[:, -test_num_updates:]
        metaval_accuracies = metaval_accuracies[:, :-test_num_updates]

    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    if FLAGS.from_scratch:
        print('Mean training accuracy')
        print(np.mean(train_accuracies, 0))

    out_name = ''
    if (FLAGS.partition_algorithm == 'kmeans' or FLAGS.partition_algorithm == 'kmodes') and FLAGS.mv_mode == 'encenc':
        out_name += '_k' + str(FLAGS.num_clusters_test)
    out_name += '_mode' + str(FLAGS.mv_mode) + '_ncv' + str(FLAGS.num_classes_val) + '_test_iubsv' + str(FLAGS.inner_update_batch_size_val) + '_test_q' + str(FLAGS.outer_update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '_iter' + str(FLAGS.metatrain_iterations)
    out_name = logdir + '/' + exp_string + '/' + out_name[1:]
    out_filename = out_name + '.csv'

    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)
        if FLAGS.from_scratch:
            writer.writerow(np.mean(train_accuracies, 0))


def main():
    if FLAGS.train:
        test_num_updates = 20
    elif FLAGS.from_scratch:
        test_num_updates = 200
    else:
        test_num_updates = 50

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        FLAGS.meta_batch_size = 1

    sess = tf.InteractiveSession()

    data_generator = DataGenerator(FLAGS.inner_update_batch_size_train + FLAGS.outer_update_batch_size,
                                   FLAGS.inner_update_batch_size_val + FLAGS.outer_update_batch_size,
                                   FLAGS.meta_batch_size)

    dim_output_train = data_generator.dim_output_train
    dim_output_val = data_generator.dim_output_val
    dim_input = data_generator.dim_input


    tf_data_load = True
    num_classes_train = data_generator.num_classes_train
    num_classes_val = data_generator.num_classes_val

    if FLAGS.train: # only construct training model if needed
        random.seed(5)
        image_tensor, label_tensor = data_generator.make_data_tensor()
        inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes_train*FLAGS.inner_update_batch_size_train, -1])
        inputb = tf.slice(image_tensor, [0,num_classes_train*FLAGS.inner_update_batch_size_train, 0], [-1,-1,-1])
        labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes_train*FLAGS.inner_update_batch_size_train, -1])
        labelb = tf.slice(label_tensor, [0,num_classes_train*FLAGS.inner_update_batch_size_train, 0], [-1,-1,-1])
        input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    random.seed(6)
    image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
    inputa = tf.slice(image_tensor, [0,0,0], [-1,num_classes_val*FLAGS.inner_update_batch_size_val, -1])
    inputb = tf.slice(image_tensor, [0,num_classes_val*FLAGS.inner_update_batch_size_val, 0], [-1,-1,-1])
    labela = tf.slice(label_tensor, [0,0,0], [-1,num_classes_val*FLAGS.inner_update_batch_size_val, -1])
    labelb = tf.slice(label_tensor, [0,num_classes_val*FLAGS.inner_update_batch_size_val, 0], [-1,-1,-1])
    metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

    model = MAML(dim_input, dim_output_train, dim_output_val, test_num_updates=test_num_updates)
    if FLAGS.train or not tf_data_load:
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    if FLAGS.debug:
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.log_inner_update_batch_size_val == -1:
        FLAGS.log_inner_update_batch_size_val = FLAGS.inner_update_batch_size_val
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    exp_string = ''
    exp_string += '.nu_' + str(FLAGS.num_updates) + '.ilr_' + str(FLAGS.train_update_lr)
    if FLAGS.mt_mode != 'gtgt':
        if FLAGS.partition_algorithm == 'hyperplanes':
            exp_string += '.m_' + str(FLAGS.margin)
        if FLAGS.partition_algorithm == 'kmeans' or FLAGS.partition_algorithm == 'kmodes':
            exp_string += '.k_' + str(FLAGS.num_clusters)
            exp_string += '.p_' + str(FLAGS.num_partitions)
            if FLAGS.scaled_encodings and FLAGS.num_partitions != 1:
                exp_string += '.scaled'
        if FLAGS.mt_mode == 'encenc':
            exp_string += '.ned_' + str(FLAGS.num_encoding_dims)
    exp_string += '.mt_' + FLAGS.mt_mode
    exp_string += '.mbs_' + str(FLAGS.meta_batch_size) + \
                  '.nct_' + str(FLAGS.num_classes_train) + \
                  '.iubst_' + str(FLAGS.inner_update_batch_size_train) + \
                    '.iubsv_' + str(FLAGS.log_inner_update_batch_size_val) + \
                    '.oubs' + str(FLAGS.outer_update_batch_size)
    exp_string = exp_string[1:]     # get rid of leading period

    if FLAGS.on_encodings:
        exp_string += '.onenc'
        exp_string += '.nhl_' + str(FLAGS.num_hidden_layers)
    if FLAGS.num_filters != 64:
        exp_string += '.hidden' + str(FLAGS.num_filters)
    if FLAGS.max_pool:
        exp_string += '.maxpool'
    if FLAGS.stop_grad:
        exp_string += '.stopgrad'
    if FLAGS.norm == 'batch_norm':
        exp_string += '.batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += '.layernorm'
    elif FLAGS.norm == 'None':
        exp_string += '.nonorm'
    else:
        print('Norm setting not recognized.')
    if FLAGS.suffix != '':
        exp_string += '.' + FLAGS.suffix

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()

    print(exp_string)

    if FLAGS.resume or not FLAGS.train:
        model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)
        else:
            print("No checkpoint found")

    if FLAGS.from_scratch:
        exp_string = ''

    if FLAGS.from_scratch and not os.path.isdir(logdir):
        os.makedirs(logdir)

    if FLAGS.train:
        train(model, saver, sess, exp_string, data_generator, resume_itr)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)


if __name__ == "__main__":
    main()
