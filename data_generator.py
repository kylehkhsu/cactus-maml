""" Code for loading data. """
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import get_data
from task_generator import TaskGenerator
from tqdm import tqdm
import os
from collections import defaultdict
import json
import ipdb

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

        if FLAGS.dataset == 'miniimagenet':
            self.num_classes_train = FLAGS.num_classes_train
            self.num_classes_val = FLAGS.num_classes_val
            self.img_size = (84, 84)
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output_train = self.num_classes_train
            self.dim_output_val = self.num_classes_val
            self.X_train, self.Y_train, self.Z_train, self.X_val, self.Y_val, self.Z_val = \
                get_data(FLAGS.dataset, FLAGS.num_encoding_dims, FLAGS.test_set)
        elif FLAGS.dataset == 'omniglot':
            self.num_classes_train = FLAGS.num_classes_train
            self.num_classes_val = FLAGS.num_classes_val
            self.img_size = (28, 28)
            self.dim_input = np.prod(self.img_size)
            self.dim_output_train = self.num_classes_train
            self.dim_output_val = self.num_classes_val
            self.X_train, self.Y_train, self.Z_train, self.X_val, self.Y_val, self.Z_val = \
                get_data(FLAGS.dataset, FLAGS.num_encoding_dims, FLAGS.test_set)
        elif FLAGS.dataset == 'mnist':
            self.num_classes_train = FLAGS.num_classes_train
            self.num_classes_val = FLAGS.num_classes_val
            self.img_size = (28, 28)
            self.dim_input = np.prod(self.img_size)
            self.dim_output_train = self.num_classes_train
            self.dim_output_val = self.num_classes_val
            self.X_train, self.Y_train, self.Z_train, self.X_val, self.Y_val, self.Z_val = \
                get_data(FLAGS.dataset, FLAGS.num_encoding_dims, FLAGS.test_set)
        elif FLAGS.dataset == 'celeba':
            self.num_classes_train = FLAGS.num_classes_train
            self.num_classes_val = FLAGS.num_classes_val
            self.img_size = (84, 84)
            self.dim_input = np.prod(self.img_size) * 3
            self.dim_output_train = self.num_classes_train
            self.dim_output_val = self.num_classes_val
            self.X_train, self.attributes_train, self.Z_train, self.X_val, self.attributes_val, self.Z_val = \
                get_data(FLAGS.dataset, FLAGS.num_encoding_dims, FLAGS.test_set)
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
            if FLAGS.dataset == 'celeba':
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
            if FLAGS.dataset == 'celeba':
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
            if FLAGS.dataset == 'celeba':
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
                if FLAGS.on_pixels:
                    Z = np.copy(X)
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


class DataGeneratorImageNet(object):
    def __init__(self, num_samples_per_class_train, num_samples_per_class_val, batch_size):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.batch_size = batch_size
        self.num_samples_per_class_train = num_samples_per_class_train
        self.num_samples_per_class_val = num_samples_per_class_val

        self.num_classes_train = FLAGS.num_classes_train
        self.num_classes_val = FLAGS.num_classes_val

        if FLAGS.input_type == 'images_84x84':
            self.dim_input = 84 * 84 * 3
        elif FLAGS.input_type == 'images_224x224':
            self.dim_input = 224 * 224 * 3
        elif FLAGS.input_type == 'features':
            self.dim_input = 4096
        elif FLAGS.input_type == 'features_processed':
            self.dim_input = 256
        else:
            raise ValueError
        self.dim_output_train = self.num_classes_train
        self.dim_output_val = self.num_classes_val

        self.split_to_path_to_info_dict = self.get_stuff(FLAGS.input_type)

    def get_stuff(self, sub_dir):
        extensions = ['.jpeg', '.npy']
        data_path = os.path.join(FLAGS.data_dir, 'imagenet', sub_dir)

        # get example names and classes
        print("Getting data from", data_path)
        class_to_file_paths = defaultdict(list)
        class_list = os.listdir(data_path)
        assert len(class_list) == 1000, 'There should be 1000 ImageNet training classes'

        for cls, class_path in tqdm([(cls, os.path.join(data_path, cls)) for cls in class_list]):
            for file_name, file_path in [(file_name, os.path.join(class_path, file_name))
                                         for file_name in os.listdir(class_path)]:
                if any(map(lambda extension: extension in file_name.lower(), extensions)):
                    class_to_file_paths[cls].append(file_path)

        # split imagenet training into miniimagenet splits
        miniimagenet_splits_dir = os.path.join(FLAGS.data_dir, 'imagenet', 'miniimagenet_splits')
        split_to_classes = dict()
        miniimagenet_split_to_classes = dict()
        for split in ['train', 'val', 'test']:
            with open(os.path.join(miniimagenet_splits_dir, split + '.csv')) as f:
                classes = set()
                next(f)
                for line in f:
                    cls = line[line.find(',') + 1: line.find('\n')]
                    classes.add(cls)
            miniimagenet_split_to_classes[split] = classes

        for split in ['val', 'test']:
            split_to_classes[split] = miniimagenet_split_to_classes[split]

        split_to_classes['miniimagenet_train'] = miniimagenet_split_to_classes['train']
        # globals().update(locals())  # hack to get around comprehension scoping issue
        train_classes = set(class_list) - (split_to_classes['val'] | split_to_classes['test'])
        split_to_classes['train'] = train_classes

        if FLAGS.num_clusters == -1:
            cluster_files = ['train_k500.json', 'train_k1000.json', 'train_k10000.json']
        else:
            cluster_files = ['train_k{}.json'.format(FLAGS.num_clusters)] + \
                            ['train_k{}_{}.json'.format(FLAGS.num_clusters, i) for i in range(1, FLAGS.num_partitions)]
        print('cluster_files: ', cluster_files)
        name_to_cluster_train_list = []
        for cluster_file in cluster_files:
            with open(file=os.path.join(FLAGS.data_dir, 'imagenet', 'clusters', cluster_file), mode='r') as f:
                name_to_cluster_train_list.append(json.load(fp=f))

        split_to_path_to_info_dict = defaultdict(dict)
        for split, classes in split_to_classes.items():
            for class_ind, cls in enumerate(tqdm(classes)):
                for file_path in class_to_file_paths[cls]:
                    name = file_path[file_path.rfind('/') + 1 : file_path.rfind('.')]
                    info = {'path': file_path, 'class_ind': class_ind, 'class': cls}
                    if split == 'train':
                        for i, name_to_cluster_train in enumerate(name_to_cluster_train_list):
                            info['cluster_ind{}'.format(i)] = name_to_cluster_train[name]
                    split_to_path_to_info_dict[split][file_path] = info
        return split_to_path_to_info_dict

    def make_data_tensor(self, train=True):
        if train:
            mode = FLAGS.mt_mode
            num_classes = self.num_classes_train
            num_samples_per_class = self.num_samples_per_class_train
            num_train_samples_per_class = FLAGS.inner_update_batch_size_train
            path_to_info_dict = self.split_to_path_to_info_dict['train']
            miniimagenet_path_to_info_dict = self.split_to_path_to_info_dict['miniimagenet_train']
            print('Setting up tasks for meta-training')
        else:
            mode = FLAGS.mv_mode
            if mode == 'encenc':
                raise NotImplementedError
            num_tasks = FLAGS.num_eval_tasks
            if FLAGS.test_set:
                path_to_info_dict = self.split_to_path_to_info_dict['test']
            else:
                path_to_info_dict = self.split_to_path_to_info_dict['val']
            num_classes = self.num_classes_val
            num_samples_per_class = self.num_samples_per_class_val
            num_train_samples_per_class = FLAGS.inner_update_batch_size_val
            print('Setting up tasks for meta-val')

        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_train_samples_per_class,
                                       num_samples_per_class=num_samples_per_class)
        partition_algorithm = FLAGS.partition_algorithm
        margin = FLAGS.margin

        file_paths = list(path_to_info_dict.keys())
        file_path_to_ind = {file_path: ind for ind, file_path in enumerate(file_paths)}

        # create partitions
        partitions = []
        if not train or not FLAGS.miniimagenet_only or mode == 'semi':
            num_partitions = len([key for key in list(path_to_info_dict[file_paths[0]].keys()) if 'cluster_ind' in key]) if mode == 'encenc' else 1
            for i in tqdm(range(num_partitions)):
                partition = defaultdict(list)
                class_ind_key = {'encenc': 'cluster_ind{}'.format(i),
                                 'semi': 'cluster_ind{}'.format(i),
                                 'gtgt': 'class_ind'}[mode]
                for file_path, info in tqdm(path_to_info_dict.items()):
                    partition[info[class_ind_key]].append(file_path_to_ind[file_path])
                partition = task_generator.clean_partition(partition)
                partitions.append(partition)
        if train and (FLAGS.miniimagenet_only or mode == 'semi'):
            partition = defaultdict(list)
            class_ind_key = {'semi': 'class_ind',
                             'gtgt': 'class_ind'}[mode]
            for file_path, info in tqdm(miniimagenet_path_to_info_dict.items()):
                partition[info[class_ind_key]].append(file_path_to_ind[file_path])
            for cls, indices in partition.items():
                partition[cls] = indices[:600]
            partitions.append(partition)
        print('Number of partitions: {}'.format(len(partitions)))
        print('Average number of clusters/classes: {}'.format(np.mean([len(partition.keys()) for partition in partitions])))

        def sample_task():
            if mode == 'semi':
                assert len(partitions) == 2
                assert 0 <= FLAGS.p_gtgt <= 1
                p = [1 - FLAGS.p_gtgt, FLAGS.p_gtgt]
            else:
                p = None
            while True:
                i = np.random.choice(len(partitions), replace=False, p=p)
                train_ind, train_labels, test_ind, test_labels = task_generator.get_task(partition=partitions[i])
                train_ind, train_labels, test_ind, test_labels = np.array(train_ind), np.array(train_labels), \
                                                                 np.array(test_ind), np.array(test_labels)
                yield train_ind, train_labels, test_ind, test_labels

        def make_dict(train_ind, train_labels, test_ind, test_labels):
            return {"train_indices": train_ind, "train_labels": train_labels, "test_indices": test_ind, "test_labels": test_labels}

        def preprocess_image(file_path):
            image_string = tf.read_file(file_path)
            image = tf.image.decode_jpeg(image_string, channels=3)
            image_processed = tf.cast(tf.reshape(image, [self.dim_input]), tf.float32) / 255.0
            return image_processed

        def preprocess_feature(file_path):
            return tf.py_func(lambda file_path: np.load(file_path.decode('utf-8')), [file_path], tf.float32)

        preprocess_func = {'images_84x84': preprocess_image,
                           'images_224x224': preprocess_image,
                           'features': preprocess_feature}[FLAGS.input_type]
        ind_to_file_path_ph = tf.placeholder_with_default(file_paths, shape=len(file_paths))

        def gather_preprocess(task):
            for split in ['train', 'test']:
                task['{}_labels'.format(split)] = tf.one_hot(task['{}_labels'.format(split)], num_classes)
                task['{}_inputs'.format(split)] = tf.map_fn(fn=preprocess_func, dtype=tf.float32, elems=tf.gather(ind_to_file_path_ph, task['{}_indices'.format(split)]))
            return task

        def stack(task):
            inputs = tf.concat((task['train_inputs'], task['test_inputs']), axis=0)
            labels = tf.concat((task['train_labels'], task['test_labels']), axis=0)
            return inputs, labels
        #
        # tasks = task_generator.get_tasks(num_tasks=num_tasks, partitions=partitions)
        # train_ind, train_labels, test_ind, test_labels = zip(*tasks)
        #
        # train_ind, train_labels, test_ind, test_labels = np.array(train_ind), np.array(train_labels), \
        #                                                  np.array(test_ind), np.array(test_labels)
        # train_ind_ph = tf.placeholder(dtype=tf.int64, shape=train_ind.shape)
        # train_labels_ph = tf.placeholder(dtype=tf.int64, shape=train_labels.shape)
        # test_ind_ph = tf.placeholder(dtype=tf.int64, shape=test_ind.shape)
        # test_labels_ph = tf.placeholder(dtype=tf.int64, shape=test_labels.shape)
        # dataset = tf.data.Dataset.from_tensor_slices(
        #     {"train_indices": train_ind_ph, "train_labels": train_labels_ph,
        #      "test_indices": test_ind_ph, "test_labels": test_labels_ph})
        # dataset = dataset.map(map_func=gather_preprocess, num_parallel_calls=FLAGS.num_parallel_calls)
        # dataset = dataset.map(map_func=stack, num_parallel_calls=FLAGS.num_parallel_calls)
        # dataset = dataset.batch(batch_size=self.batch_size)
        # dataset = dataset.prefetch(4)
        # dataset = dataset.repeat()
        # iterator = dataset.make_initializable_iterator()
        # inputs_batch, labels_batch = iterator.get_next()
        #
        # # sess = tf.InteractiveSession()
        # iterator.initializer.run(feed_dict={train_ind_ph: train_ind,
        #                                     train_labels_ph: train_labels,
        #                                     test_ind_ph: test_ind,
        #                                     test_labels_ph: test_labels})

        dataset = tf.data.Dataset.from_generator(sample_task, output_types=(tf.int64, tf.int64, tf.int64, tf.int64))
        dataset = dataset.map(map_func=make_dict, num_parallel_calls=1)
        dataset = dataset.map(map_func=gather_preprocess, num_parallel_calls=FLAGS.num_parallel_calls)
        dataset = dataset.map(map_func=stack, num_parallel_calls=FLAGS.num_parallel_calls)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(4)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        inputs_batch, labels_batch = iterator.get_next()

        return inputs_batch, labels_batch
