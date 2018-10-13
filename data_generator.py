""" Code for loading data. """
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import get_data
from task_generator import TaskGenerator
from tqdm import tqdm

FLAGS = flags.FLAGS


class DataGenerator(object):
    def __init__(self, num_samples_per_class_train, num_samples_per_class_val, batch_size):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class_train = num_samples_per_class_train
        self.num_samples_per_class_val = num_samples_per_class_val
        self.num_classes_train = 1  # by default 1 (only relevant for classification problems)

        if FLAGS.datasource == 'miniimagenet':
            self.num_classes_train = FLAGS.num_classes_train
            self.num_classes_val = FLAGS.num_classes_val
            self.img_size = (84, 84)
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output_train = self.num_classes_train
            self.dim_output_val = self.num_classes_val
            self.X_train, self.Y_train, self.Z_train, self.X_val, self.Y_val, self.Z_val = \
                get_data(FLAGS.datasource, FLAGS.num_encoding_dims, FLAGS.test_set)
        elif FLAGS.datasource == 'omniglot':
            self.num_classes_train = FLAGS.num_classes_train
            self.num_classes_val = FLAGS.num_classes_val
            self.img_size = (28, 28)
            self.dim_input = np.prod(self.img_size)
            self.dim_output_train = self.num_classes_train
            self.dim_output_val = self.num_classes_val
            self.X_train, self.Y_train, self.Z_train, self.X_val, self.Y_val, self.Z_val = \
                get_data(FLAGS.datasource, FLAGS.num_encoding_dims, FLAGS.test_set)
        elif FLAGS.datasource == 'mnist':
            self.num_classes_train = FLAGS.num_classes_train
            self.num_classes_val = FLAGS.num_classes_val
            self.img_size = (28, 28)
            self.dim_input = np.prod(self.img_size)
            self.dim_output_train = self.num_classes_train
            self.dim_output_val = self.num_classes_val
            self.X_train, self.Y_train, self.Z_train, self.X_val, self.Y_val, self.Z_val = \
                get_data(FLAGS.datasource, FLAGS.num_encoding_dims, FLAGS.test_set)
        elif FLAGS.datasource == 'celeba':
            self.num_classes_train = FLAGS.num_classes_train
            self.num_classes_val = FLAGS.num_classes_val
            self.img_size = (84, 84)
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output_train = self.num_classes_train
            self.dim_output_val = self.num_classes_val
            self.X_train, self.attributes_train, self.Z_train, self.X_val, self.attributes_val, self.Z_val = \
                get_data(FLAGS.datasource, FLAGS.num_encoding_dims, FLAGS.test_set)
        else:
            raise ValueError('Unrecognized data source')
        if FLAGS.on_encodings:
            self.dim_input = self.Z_train.shape[1]


    def make_data_tensor(self, train=True):
        if train:
            mode = FLAGS.mt_mode
            num_classes = self.num_classes_train
            num_tasks = FLAGS.metatrain_iterations * self.batch_size
            num_splits = 1000
            if FLAGS.num_partitions == -1:
                num_partitions = num_tasks
            else:
                num_partitions = FLAGS.num_partitions
            if FLAGS.datasource == 'celeba':
                assert num_classes == 2, "CelebA must have two classes"
                X, attributes, Z = self.X_train, self.attributes_train, self.Z_train
            else:
                X, Y, Z = self.X_train, self.Y_train, self.Z_train
            num_samples_per_class = self.num_samples_per_class_train
            num_train_samples_per_class = FLAGS.inner_update_batch_size_train
            print('Setting up tasks for meta-training')
        else:
            mode = FLAGS.mv_mode
            if mode == 'encenc':
                raise NotImplementedError
            num_tasks = FLAGS.num_eval_tasks
            num_splits = 100
            num_partitions = num_tasks
            if FLAGS.datasource == 'celeba':
                X, attributes, Z = self.X_val, self.attributes_val, self.Z_val
            else:
                X, Y, Z = self.X_val, self.Y_val, self.Z_val
            num_classes = self.num_classes_val
            num_samples_per_class = self.num_samples_per_class_val
            num_train_samples_per_class = FLAGS.inner_update_batch_size_val
            print('Setting up tasks for meta-val')

        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_train_samples_per_class, num_samples_per_class=num_samples_per_class)
        partition_algorithm = FLAGS.partition_algorithm
        margin = FLAGS.margin

        print('Generating indices for {} tasks'.format(num_tasks))
        if mode == 'gtgt':
            if FLAGS.datasource == 'celeba':
                partitions = task_generator.get_celeba_task_pool(attributes=attributes)
            else:
                print('Using ground truth partition to create classes')
                partition = task_generator.get_partition_from_labels(labels=Y)
                partitions = [partition]
        elif mode == 'encenc':
            if partition_algorithm == 'hyperplanes':
                print('Using {} hyperplanes-based partition(s) of encoding space to create classes, margin={}'.format(num_partitions, margin))
                partitions = task_generator.get_partitions_hyperplanes(encodings=Z, num_splits=num_splits,
                                                                       margin=margin, num_partitions=num_partitions)
            elif partition_algorithm == 'kmeans':
                print('Using {} k-means based partition(s) of encoding space to create classes'.format(num_partitions))
                partitions = task_generator.get_partitions_kmeans(encodings=Z, train=train)
            else:
                raise ValueError('Unrecognized partition-generating algorithm: either hyperplanes or kmeans')
        elif mode == 'randrand':
            print('Randomly sampled and labeled tasks')
            partitions = []
            for p in tqdm(range(num_partitions)):
                labels = np.random.choice(FLAGS.num_clusters, size=Y.shape, replace=True)
                partition = task_generator.get_partition_from_labels(labels=labels)
                partitions.append(partition)
        else:
            raise ValueError('Unrecognized task generation scheme')
        print('Average number of classes per partition: {}'.format(np.mean([len(list(partition.keys()))for partition in partitions])))
        if FLAGS.on_encodings:
            features = features_ph = tf.placeholder(Z.dtype, Z.shape)
        else:
            assert X.dtype == 'uint8'
            features_ph = tf.placeholder(X.dtype, X.shape)
            features = tf.reshape(features_ph, [-1, self.dim_input])

        def gather_preprocess(task):
            for split in ['train', 'test']:
                task['{}_labels'.format(split)] = tf.one_hot(task['{}_labels'.format(split)], num_classes)
                if not FLAGS.on_encodings:
                    task['{}_features'.format(split)] = tf.cast(tf.gather(features, task['{}_indices'.format(split)]), tf.float32) / 255.0
                else:
                    task['{}_features'.format(split)] = tf.gather(features, task['{}_indices'.format(split)])
            return task

        def stack(task):
            features = tf.concat((task['train_features'], task['test_features']), axis=0)
            labels = tf.concat((task['train_labels'], task['test_labels']), axis=0)
            return features, labels

        tasks = task_generator.get_tasks(num_tasks=num_tasks, partitions=partitions)
        train_ind, train_labels, test_ind, test_labels = [task[0] for task in tasks], [task[1] for task in tasks], [task[2] for task in tasks], [task[3] for task in tasks]

        dataset = tf.data.Dataset.from_tensor_slices(
            {"train_indices": train_ind, "train_labels": train_labels, "test_indices": test_ind, "test_labels": test_labels})
        dataset = dataset.map(map_func=gather_preprocess, num_parallel_calls=FLAGS.num_parallel_calls)
        dataset = dataset.map(map_func=stack, num_parallel_calls=FLAGS.num_parallel_calls)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(4)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        features_batch, labels_batch = iterator.get_next()

        if FLAGS.on_encodings:
            iterator.initializer.run(feed_dict={features_ph: Z})
        else:
            iterator.initializer.run(feed_dict={features_ph: X})

        return features_batch, labels_batch
