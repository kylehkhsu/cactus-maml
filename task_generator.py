import numpy as np
import random
from tensorflow.python.platform import flags
from functools import reduce
from tqdm import tqdm
FLAGS = flags.FLAGS
from collections import defaultdict
from itertools import combinations, product
import os
from sklearn.cluster import KMeans

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'  # default runs out of space for parallel processing


class TaskGenerator(object):
    def __init__(self, num_classes, num_train_samples_per_class, num_samples_per_class):
        self.num_classes = num_classes
        self.num_train_samples_per_class = num_train_samples_per_class
        self.num_samples_per_class = num_samples_per_class

    def get_task(self, partition):
        """
        A task consists of training and testing examples and labels. Instead of examples, we pass indices.
        """
        train_indices, train_labels, test_indices, test_labels = [], [], [], []
        classes = list(partition.keys())
        sampled_classes = random.sample(classes, self.num_classes)
        random.shuffle(sampled_classes)     # the same classes given a different label ordering is a new task
        for label, cls in zip(range(self.num_classes), sampled_classes):
            class_indices = random.sample(partition[cls], self.num_samples_per_class)
            train_indices.extend(class_indices[:self.num_train_samples_per_class])
            test_indices.extend(class_indices[self.num_train_samples_per_class:])
            train_labels.extend([label for i in range(self.num_train_samples_per_class)])
            test_labels.extend([label for i in range(self.num_samples_per_class - self.num_train_samples_per_class)])
        return train_indices, train_labels, test_indices, test_labels

    def get_tasks(self, num_tasks, partitions):
        tasks = []
        for i_task in tqdm(range(num_tasks), desc='get_tasks'):
            if num_tasks == len(partitions):
                i_partition = i_task
            else:
                i_partition = random.sample(range(len(partitions)), 1)[0]
            task = self.get_task(partition=partitions[i_partition])
            tasks.append(task)
        return tasks

    def get_splits_hyperplanes(self, encodings, num_splits, margin):
        """
        A split is a tuple of two zones, each of which is an array of indices whose encodings are more than margin away
        from a random hyperplane.
        """
        assert margin >= 0
        n, d = encodings.shape
        splits = []
        good_splits, bad_splits = 0, 0
        min_samples_per_zone = self.num_samples_per_class * 10
        for i_split in tqdm(range(num_splits), desc='get_splits_hyperplanes'):
            while True:
                normal_vector = np.random.uniform(low=-1.0, high=1.0, size=(d,))
                unit_normal_vector = normal_vector / np.linalg.norm(normal_vector)
                if FLAGS.encoder == 'deepcluster':  # whitened and normalized
                    point_on_plane = np.random.uniform(low=0.0, high=0.0, size=(d,))
                else:
                    point_on_plane = np.random.uniform(low=-0.8, high=0.8, size=(d,))
                relative_vector = encodings - point_on_plane # broadcasted
                signed_distance = np.dot(relative_vector, unit_normal_vector)
                below = np.where(signed_distance <= -margin)[0]
                above = np.where(signed_distance >= margin)[0]
                if len(below) < (min_samples_per_zone) or len(above) < (min_samples_per_zone):
                    bad_splits += 1
                else:
                    splits.append((below, above))
                    good_splits += 1
                    break
        print("Generated {} random splits, with {} failed splits.".format(num_splits, bad_splits))
        return splits

    def get_partitions_hyperplanes(self, encodings, num_splits, margin, num_partitions):
        """Create partitions where each element must be a certain margin away from all split-defining hyperplanes."""
        splits = self.get_splits_hyperplanes(encodings=encodings, num_splits=num_splits, margin=margin)
        bad_partitions = 0
        partitions = []
        for i in tqdm(range(num_partitions), desc='get_partitions_hyperplanes'):
            partition, num_failed = self.get_partition_from_splits(splits)
            partitions.append(partition)
            bad_partitions += num_failed
            if (i+1) % (num_partitions // 10) == 0:
                tqdm.write('\t good partitions: {}, bad partitions: {}'.format(i + 1, bad_partitions))
        print("Generated {} partitions respecting margin {}, with {} failed partitions.".format(num_partitions, margin, bad_partitions))
        return partitions

    def get_partition_from_splits(self, splits):
        num_splits = len(splits)
        splits_per_partition = np.int(np.ceil(np.log2(self.num_classes)))

        num_failed = 0
        while True:
            which_splits = np.random.choice(num_splits, splits_per_partition, replace=False)
            splits_for_this_partition = [splits[i] for i in which_splits]
            partition = defaultdict(list)
            num_big_enough_classes = 0
            for i_class, above_or_belows in enumerate(product([0, 1], repeat=splits_per_partition)):
                zones = [splits_for_this_partition[i][above_or_belows[i]] for i in range(splits_per_partition)]
                indices = reduce(np.intersect1d, zones)
                if len(indices) >= self.num_samples_per_class:
                    num_big_enough_classes += 1
                    partition[i_class].extend(indices.tolist())
            if num_big_enough_classes >= self.num_classes:
                break
            else:
                num_failed += 1
        return partition, num_failed

    def get_partitions_kmeans(self, encodings, train):
        encodings_list = [encodings]
        if train:
            if FLAGS.scaled_encodings:
                n_clusters_list = [FLAGS.num_clusters]
                for i in range(FLAGS.num_partitions - 1):
                    weight_vector = np.random.uniform(low=0.0, high=1.0, size=encodings.shape[1])
                    encodings_list.append(np.multiply(encodings, weight_vector))
            else:
                n_clusters_list = [FLAGS.num_clusters] * FLAGS.num_partitions
        else:
            n_clusters_list = [FLAGS.num_clusters_test]
        assert len(encodings_list) * len(n_clusters_list) == FLAGS.num_partitions
        if FLAGS.dataset == 'celeba' or FLAGS.num_partitions != 1:
            n_init = 1  # so it doesn't take forever
        else:
            n_init = 10
        init = 'k-means++'

        print('Number of encodings: {}, number of n_clusters: {}, number of inits: '.format(len(encodings_list), len(n_clusters_list)), n_init)

        kmeans_list = []
        for n_clusters in tqdm(n_clusters_list, desc='get_partitions_kmeans_n_clusters'):
            for encodings in tqdm(encodings_list, desc='get_partitions_kmeans_encodings'):
                while True:
                    kmeans = KMeans(n_clusters=n_clusters, init=init, precompute_distances=True, n_jobs=n_init,
                                    n_init=n_init, max_iter=3000).fit(encodings)
                    uniques, counts = np.unique(kmeans.labels_, return_counts=True)
                    num_big_enough_clusters = np.sum(counts > self.num_samples_per_class)
                    if num_big_enough_clusters > 0.75 * n_clusters:
                        break
                    else:
                        tqdm.write("Too few classes ({}) with greater than {} examples.".format(num_big_enough_clusters,
                                                                                           self.num_samples_per_class))
                        tqdm.write('Frequency: {}'.format(counts))
                kmeans_list.append(kmeans)
        partitions = []
        for kmeans in kmeans_list:
            partition = self.get_partition_from_labels(kmeans.labels_)
            partitions.append(partition)
        return partitions

    def get_partition_from_labels(self, labels):
        """
        Constructs a partition of the set of indices in labels, grouping indices according to their label.
        :param labels: np.array of labels, whose i-th element is the label for the i-th datapoint
        :return: a dictionary mapping class label to a list of indices that have that label
        """
        partition = defaultdict(list)
        for ind, label in enumerate(labels):
            partition[label].append(ind)
        self.clean_partition(partition)
        return partition

    def clean_partition(self, partition):
        """
        Removes subsets that are too small from a partition.
        """
        for cls in list(partition.keys()):
            if len(partition[cls]) < self.num_samples_per_class:
                del(partition[cls])
        return partition

    def get_celeba_task_pool(self, attributes, order=3, print_partition=None):
        """
        Produces partitions: a list of dictionaries (key: 0 or 1, value: list of data indices), which is
        compatible with the other methods of this class.
        """
        num_pools = 0
        partitions = []
        from scipy.special import comb
        for attr_comb in tqdm(combinations(range(attributes.shape[1]), order), desc='get_task_pool', total=comb(attributes.shape[1], order)):
            for booleans in product(range(2), repeat=order-1):
                booleans = (0,) + booleans  # only the half of the cartesian products that start with 0
                positive = np.where(np.all([attributes[:, attr] == i_booleans for (attr, i_booleans) in zip(attr_comb, booleans)], axis=0))[0]
                if len(positive) < self.num_samples_per_class:
                    continue
                negative = np.where(np.all([attributes[:, attr] == 1 - i_booleans for (attr, i_booleans) in zip(attr_comb, booleans)], axis=0))[0]
                if len(negative) < self.num_samples_per_class:
                    continue
                # inner_pool[booleans] = {0: list(negative), 1: list(positive)}
                partitions.append({0: list(negative), 1: list(positive)})
                num_pools += 1
                if num_pools == print_partition:
                    print(attr_comb, booleans)
        print('Generated {} task pools by using {} attributes from {} per pool'.format(num_pools, order, attributes.shape[1]))
        return partitions