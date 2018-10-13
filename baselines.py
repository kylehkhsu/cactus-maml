import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics import accuracy_score
from collections import defaultdict
import os
import time
from tensorflow.python.platform import flags
from utils import get_data
from task_generator import TaskGenerator
from tqdm import tqdm
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'mnist', 'mnist or omniglot or miniimagenet or celeba')
flags.DEFINE_integer('way', -1, 'classes for few-shot learning')
flags.DEFINE_integer('shot', -1, 'examples per class for few-shot learning')
flags.DEFINE_boolean('test_set', False, 'use validation set (default) or test set')
flags.DEFINE_string('encoder', 'bigan', 'bigan or aae')
flags.DEFINE_string('algorithm', 'kmeans', 'baseline algorithm to run')
flags.DEFINE_integer('num_tasks', 1000, 'number of few shot tasks to evaluate on')
flags.DEFINE_integer('num_clusters', 10, 'number of clusters for kmeans')
flags.DEFINE_integer('num_encoding_dims', 10, 'num_encoding_dims')
flags.DEFINE_integer('units', -1, 'number of units in hidden dense layer')
flags.DEFINE_float('dropout', -1.0, 'dropout rate')
flags.DEFINE_integer('n_neighbours', -1, 'k_nn for nearest neighbours')
flags.DEFINE_float('inverse_reg', -1, 'inverse regularization strength for logistic regression')

os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp' # default parallel processing directory runs out of space
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def embedding_cluster_matching(num_classes=FLAGS.way, num_shots=FLAGS.shot, num_tasks=FLAGS.num_tasks,
           num_clusters=FLAGS.num_clusters, num_encoding_dims=FLAGS.num_encoding_dims,
           dataset=FLAGS.dataset, test_set=FLAGS.test_set):
    if dataset != 'celeba':
        _, _, Z_train, X_test, Y_test, Z_test = get_data(dataset, num_encoding_dims, test_set)
    else:
        _, _, Z_train, X_test, attributes_test, Z_test = get_data(dataset, num_encoding_dims, test_set)

    start = time.time()
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0, precompute_distances=True, n_jobs=10, n_init=10, max_iter=3000).fit(Z_train)
    print("Ran KMeans with n_clusters={} in {:.5} seconds, objective {}.".format(num_clusters, time.time() - start, kmeans.score(Z_train)))

    if dataset != 'celeba':
        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_shots, num_samples_per_class=num_shots+5)
        partition = task_generator.get_partition_from_labels(Y_test)
        partitions = [partition]
    else:
        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_shots, num_samples_per_class=num_shots+5)
        partitions = task_generator.get_celeba_task_pool(attributes_test)
    tasks = task_generator.get_tasks(num_tasks=num_tasks, partitions=partitions)

    for num_shots in [FLAGS.shot]:
        accuracies = []
        start = time.time()
        num_degenerate_tasks = 0

        for i_task, task in enumerate(tasks):
            if (i_task + 1) % (num_tasks // 10) == 0:
                print('test {}, accuracy {:.5}'.format(i_task + 1, np.mean(accuracies)))

            ind_train_few, Y_train_few, ind_test_few, Y_test_few = task
            Z_train_few, Z_test_few = Z_test[ind_train_few], Z_test[ind_test_few]

            clusters_to_labels_few = defaultdict(list)
            examples_to_clusters_few = kmeans.predict(Z_train_few)
            for i in range(len(Y_train_few)):
                clusters_to_labels_few[examples_to_clusters_few[i]].append(Y_train_few[i])
            for (cluster, labels) in list(clusters_to_labels_few.items()):
                uniques, counts = np.unique(labels, return_counts=True)
                clusters_to_labels_few[cluster] = [uniques[np.argmax(counts)]]
                # if len(np.unique(labels)) > 1:      # delete degenerate clusters
                #     del clusters_to_labels_few[cluster]
            if len(clusters_to_labels_few) == 0:
                num_degenerate_tasks += 1
                continue
            centroid_ind_to_cluster = np.array(list(clusters_to_labels_few.keys()))
            centroids = kmeans.cluster_centers_[centroid_ind_to_cluster]
            distances = distance.cdist(Z_test_few, centroids)
            predicted_clusters = centroid_ind_to_cluster[np.argmin(distances, axis=1)]
            predictions = []
            for cluster in predicted_clusters:
                predictions.append(clusters_to_labels_few[cluster][0])

            accuracies.append(accuracy_score(Y_test_few, predictions))
        print('dataset={}, encoder={}, num_encoding_dims={}, num_clusters={}'.format(dataset, FLAGS.encoder, num_clusters, num_encoding_dims))
        print('{}-way {}-shot nearest-cluster after clustering embeddings: {:.5} with 95% CI {:.5} over {} tests'.format(num_classes, num_shots, np.mean(accuracies), 1.96*np.std(accuracies)/np.sqrt(num_tasks), num_tasks))
        print('{} few-shot classification tasks: {:.5} seconds with {} degenerate tasks.'.format(num_tasks, time.time() - start, num_degenerate_tasks))


def embedding_mlp(num_classes=FLAGS.way, num_shots=FLAGS.shot, num_tasks=FLAGS.num_tasks,
                  num_encoding_dims=FLAGS.num_encoding_dims, test_set=FLAGS.test_set, dataset=FLAGS.dataset,
                  units=FLAGS.units, dropout=FLAGS.dropout):
    import keras
    from keras.layers import Dense, Dropout
    from keras.losses import categorical_crossentropy
    from keras.callbacks import EarlyStopping
    from keras import backend as K

    if dataset != 'celeba':
        _, _, _, X_test, Y_test, Z_test = get_data(dataset, num_encoding_dims, test_set)
        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_shots, num_samples_per_class=num_shots+5)
        partition = task_generator.get_partition_from_labels(Y_test)
        partitions = [partition]
    else:
        _, _, _, X_test, attributes_test, Z_test = get_data(dataset, num_encoding_dims, test_set)
        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_shots, num_samples_per_class=num_shots+5)
        partitions = task_generator.get_celeba_task_pool(attributes_test)
    tasks = task_generator.get_tasks(num_tasks=num_tasks, partitions=partitions)

    train_accuracies, test_accuracies = [], []

    start = time.time()
    for i_task, task in enumerate(tqdm(tasks)):
        if (i_task + 1) % (num_tasks // 10) == 0:
            tqdm.write('test {}, accuracy {:.5}'.format(i_task + 1, np.mean(test_accuracies)))
        ind_train_few, Y_train_few, ind_test_few, Y_test_few = task
        Z_train_few, Z_test_few = Z_test[ind_train_few], Z_test[ind_test_few]
        Y_train_few, Y_test_few = keras.utils.to_categorical(Y_train_few, num_classes=num_classes), keras.utils.to_categorical(Y_test_few, num_classes=num_classes)

        model = keras.Sequential()
        model.add(Dense(units=units, activation='relu', input_dim=Z_train_few.shape[1]))
        model.add(Dropout(rate=dropout))
        model.add(Dense(units=num_classes, activation='softmax'))
        model.compile(loss=categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=2)
        model.fit(Z_train_few, Y_train_few, batch_size=Z_train_few.shape[0], epochs=500, verbose=0, validation_data=(Z_test_few, Y_test_few), callbacks=[early_stopping])
        train_score = model.evaluate(Z_train_few, Y_train_few, verbose=0)
        train_accuracies.append(train_score[1])
        test_score = model.evaluate(Z_test_few, Y_test_few, verbose=0)
        test_accuracies.append(test_score[1])
        K.clear_session()

    print('units={}, dropout={}'.format(units, dropout))
    print('{}-way {}-shot embedding mlp: {:.5} with 95% CI {:.5} over {} tests'.format(num_classes, num_shots, np.mean(test_accuracies), 1.96*np.std(test_accuracies)/np.sqrt(num_tasks), num_tasks))
    print('Mean training accuracy: {:.5}; standard deviation: {:.5}'.format(np.mean(train_accuracies), np.std(train_accuracies)))
    print('{} few-shot classification tasks: {:.5} seconds.'.format(num_tasks, time.time() - start))


def embedding_nearest_neighbour(n_neighbors=FLAGS.n_neighbours, num_classes=FLAGS.way, num_shots=FLAGS.shot, num_tasks=FLAGS.num_tasks,
                                num_encoding_dims=FLAGS.num_encoding_dims, test_set=FLAGS.test_set,
                                dataset=FLAGS.dataset):
    print('{}-way {}-shot embedding nearest neighbour'.format(num_classes, num_shots))
    if dataset != 'celeba':
        _, _, _, X_test, Y_test, Z_test = get_data(dataset, num_encoding_dims, test_set)
        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_shots, num_samples_per_class=num_shots+5)
        partition = task_generator.get_partition_from_labels(Y_test)
        partitions = [partition]
    else:
        _, _, _, X_test, attributes_test, Z_test = get_data(dataset, num_encoding_dims, test_set)
        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_shots, num_samples_per_class=num_shots+5)
        partitions = task_generator.get_celeba_task_pool(attributes_test)
    tasks = task_generator.get_tasks(num_tasks=num_tasks, partitions=partitions)

    accuracies = []

    for i_task, task in enumerate(tasks):
        if (i_task + 1) % (num_tasks // 10) == 0:
            print('test {}, accuracy {:.5}'.format(i_task + 1, np.mean(accuracies)))
        ind_train_few, Y_train_few, ind_test_few, Y_test_few = task
        Z_train_few, Z_test_few = Z_test[ind_train_few], Z_test[ind_test_few]

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        knn.fit(Z_train_few, Y_train_few)
        accuracy = knn.score(Z_test_few, Y_test_few)

        accuracies.append(accuracy)

    print('{}-way {}-shot embedding nearest neighbour: {:.5} with 95% CI {:.5} over {} tests'.format(num_classes, num_shots, np.mean(accuracies), 1.96*np.std(accuracies)/np.sqrt(num_tasks), num_tasks))


def embedding_logistic_regression(C=FLAGS.inverse_reg, penalty='l2', multi_class='multinomial', num_classes=FLAGS.way, num_shots=FLAGS.shot, num_tasks=FLAGS.num_tasks,
                                num_encoding_dims=FLAGS.num_encoding_dims, test_set=FLAGS.test_set,
                                dataset=FLAGS.dataset):
    print('{}-way {}-shot logistic regression'.format(num_classes, num_shots))
    if dataset != 'celeba':
        _, _, _, X_test, Y_test, Z_test = get_data(dataset, num_encoding_dims, test_set)
        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_shots, num_samples_per_class=num_shots+5)
        partition = task_generator.get_partition_from_labels(Y_test)
        partitions = [partition]
    else:
        _, _, _, X_test, attributes_test, Z_test = get_data(dataset, num_encoding_dims, test_set)
        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_shots, num_samples_per_class=num_shots+5)
        partitions = task_generator.get_celeba_task_pool(attributes_test)
    tasks = task_generator.get_tasks(num_tasks=num_tasks, partitions=partitions)

    train_accuracies, test_accuracies = [], []

    start = time.time()
    for i_task, task in enumerate(tasks):
        if (i_task + 1) % (num_tasks // 10) == 0:
            print('test {}, train accuracy {:.5}, test accuracy {:.5}'.format(i_task + 1, np.mean(train_accuracies), np.mean(test_accuracies)))
        ind_train_few, Y_train_few, ind_test_few, Y_test_few = task
        Z_train_few, Z_test_few = Z_test[ind_train_few], Z_test[ind_test_few]

        logistic_regression = LogisticRegression(n_jobs=-1, penalty=penalty, C=C, multi_class=multi_class, solver='saga', max_iter=1000)
        logistic_regression.fit(Z_train_few, Y_train_few)
        test_accuracies.append(logistic_regression.score(Z_test_few, Y_test_few))
        train_accuracies.append(logistic_regression.score(Z_train_few, Y_train_few))
    print('penalty={}, C={}, multi_class={}'.format(penalty, C, multi_class))
    print('{}-way {}-shot logistic regression: {:.5} with 95% CI {:.5} over {} tests'.format(num_classes, num_shots, np.mean(test_accuracies), 1.96*np.std(test_accuracies)/np.sqrt(num_tasks), num_tasks))
    print('Mean training accuracy: {:.5}; standard deviation: {:.5}'.format(np.mean(train_accuracies), np.std(train_accuracies)))
    print('{} few-shot classification tasks: {:.5} seconds.'.format(num_tasks, time.time() - start))


def cluster_color_logistic_regression(C=FLAGS.inverse_reg, penalty='l2', multi_class='multinomial', n_clusters=FLAGS.num_clusters, num_classes=FLAGS.way, num_shots=FLAGS.shot, num_tasks=FLAGS.num_tasks,
                                num_encoding_dims=FLAGS.num_encoding_dims, test_set=FLAGS.test_set,
                                dataset=FLAGS.dataset):
    if dataset != 'celeba':
        _, _, Z_train, X_test, Y_test, Z_test = get_data(dataset, num_encoding_dims, test_set)
    else:
        _, _, Z_train, X_test, attributes_test, Z_test = get_data(dataset, num_encoding_dims, test_set)

    start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, precompute_distances=True, n_jobs=-1, n_init=100).fit(Z_train)
    print("Ran KMeans with n_clusters={} in {:.5} seconds.".format(n_clusters, time.time() - start))
    uniques, counts = np.unique(kmeans.labels_, return_counts=True)

    if dataset != 'celeba':
        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_shots, num_samples_per_class=num_shots+5)
        partition = task_generator.get_partition_from_labels(Y_test)
        partitions = [partition]
    else:
        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_shots, num_samples_per_class=num_shots+5)
        partitions = task_generator.get_celeba_task_pool(attributes_test)
    tasks = task_generator.get_tasks(num_tasks=num_tasks, partitions=partitions)

    train_accuracies, test_accuracies = [], []
    start = time.time()
    clusters_to_indices = task_generator.get_partition_from_labels(kmeans.labels_)
    for i_task, task in enumerate(tasks):
        if (i_task + 1) % (num_tasks // 10) == 0:
            print('test {}, train accuracy {:.5}, test accuracy {:.5}'.format(i_task + 1, np.mean(train_accuracies), np.mean(test_accuracies)))

        ind_train_few, Y_train_few, ind_test_few, Y_test_few = task
        Z_train_few, Z_test_few = Z_test[ind_train_few], Z_test[ind_test_few]
        clusters_to_labels_few = defaultdict(list)
        indices_to_clusters_few = kmeans.predict(Z_train_few)
        for i in range(Z_train_few.shape[0]):
            clusters_to_labels_few[indices_to_clusters_few[i]].append(Y_train_few[i])
        Z_train_fit, Y_train_fit = [], []
        for cluster in list(clusters_to_labels_few.keys()):
            labels = clusters_to_labels_few[cluster]
            if len(np.unique(labels)) == 1:     # skip degenerate clusters
                Z_train_fit.extend(Z_train[clusters_to_indices[cluster]])   # propagate labels to unlabeled datapoints
                Y_train_fit.extend([labels[0] for i in range(len(clusters_to_indices[cluster]))])
        Z_train_fit, Y_train_fit = np.stack(Z_train_fit, axis=0), np.stack(Y_train_fit, axis=0)
        Z_train_fit = np.concatenate((Z_train_fit, Z_train_few), axis=0)
        Y_train_fit = np.concatenate((Y_train_fit, Y_train_few), axis=0)

        logistic_regression = LogisticRegression(n_jobs=-1, penalty=penalty, C=C, multi_class=multi_class, solver='saga', max_iter=500)
        logistic_regression.fit(Z_train_fit, Y_train_fit)
        test_accuracies.append(logistic_regression.score(Z_test_few, Y_test_few))
        train_accuracies.append(logistic_regression.score(Z_train_fit, Y_train_fit))
    print('n_clusters={}, penalty={}, C={}, multi_class={}'.format(n_clusters, penalty, C, multi_class))
    print('{}-way {}-shot logistic regression after clustering: {:.5} with 95% CI {:.5} over {} tests'.format(num_classes, num_shots, np.mean(test_accuracies), 1.96*np.std(test_accuracies)/np.sqrt(num_tasks), num_tasks))
    print('Mean training accuracy: {:.5}; standard deviation: {:.5}'.format(np.mean(train_accuracies), np.std(train_accuracies)))
    print('{} few-shot classification tasks: {:.5} seconds.'.format(num_tasks, time.time() - start))


def cluster_fit_color(num_classes=FLAGS.way, num_tasks=FLAGS.num_tasks,
               num_clusters=FLAGS.num_clusters, num_encoding_dims = FLAGS.num_encoding_dims,
               test_set=FLAGS.test_set, dataset=FLAGS.dataset):
    assert dataset == 'mnist'
    import keras
    from keras.layers import Conv2D, Flatten, Dense
    from keras.losses import categorical_crossentropy
    from keras.optimizers import Adam
    from sklearn.cluster import KMeans

    X_train, Y_train, Z_train, X_test, Y_test, Z_test = get_data(dataset, num_encoding_dims, test_set)
    # Z_train, Z_test = whitening(Z_train, Z_test)
    start = time.time()
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0, precompute_distances=True, n_jobs=-1,
                    n_init=1000, max_iter=100000).fit(Z_train)
    print("Ran KMeans with n_clusters={} in {:.5} seconds, objective {}.".format(num_clusters, time.time() - start,
                                                                                 kmeans.score(Z_train)))

    X_train, X_test = X_train / 255.0, X_test / 255.0
    X_train, X_test = X_train.reshape((-1, 28, 28, 1)), X_test.reshape((-1, 28, 28, 1))

    cluster_labels_train = keras.utils.to_categorical(kmeans.labels_, num_clusters)
    cluster_labels_test = keras.utils.to_categorical(kmeans.predict(Z_test), num_clusters)

    model = keras.Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(units=num_clusters, activation='softmax'))
    model.summary()
    model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    model.fit(X_train, cluster_labels_train, batch_size=500, epochs=25, verbose=1, validation_data=(X_test, cluster_labels_test))
    score = model.evaluate(X_test, cluster_labels_test, verbose=1)
    print('Test loss: {}\tTest accuracy: {}'.format(score[0], score[1]))
    model.compile(loss=categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])

    for num_shots in [1, 5, 10]:
        accuracies, finetuned_accuracies = [], []
        num_degenerate_tasks = 0
        start = time.time()
        task_generator = TaskGenerator(num_classes=num_classes, num_train_samples_per_class=num_shots, num_samples_per_class=num_shots+5)
        partition = task_generator.get_partition_from_labels(Y_test)
        for i_test in range(num_tasks):
            if (i_test + 1) % (num_tasks // 10) == 0:
                print('test {}, accuracy {:.5}, finetuned accuracy {:.5}'.format(i_test + 1, np.mean(accuracies), np.mean(finetuned_accuracies)))

            task = task_generator.get_task(partition=partition)
            ind_train_few, Y_train_few, ind_test_few, Y_test_few = task
            X_train_few, X_test_few = X_test[ind_train_few], X_test[ind_test_few]

            cluster_to_labels_few = defaultdict(list)
            Z_train_few = np.argmax(model.predict(X_train_few), axis=1)
            Z_test_few = np.argmax(model.predict(X_test_few), axis=1)
            for i in range(len(Y_train_few)):
                cluster_to_labels_few[Z_train_few[i]].append(Y_train_few[i])
            cluster_to_label_few = defaultdict(int)
            for (cluster, labels) in list(cluster_to_labels_few.items()):
                uniques, counts = np.unique(labels, return_counts=True)
                cluster_to_label_few[cluster] = uniques[np.argmax(counts)]
            if len(cluster_to_label_few) == 0:
                num_degenerate_tasks += 1
                continue
            predictions = []
            for z in Z_test_few:
                predictions.append(cluster_to_label_few[z])
            accuracies.append(accuracy_score(Y_test_few, predictions))

        print('num_clusters={}, num_encoding_dims={}'.format(num_clusters, num_encoding_dims))
        print('{}-way {}-shot fit_kmeans: {:.5} with 95% CI {:.5} over {} tests'.format(num_classes, num_shots, np.mean(accuracies), 1.96*np.std(accuracies)/np.sqrt(num_tasks), num_tasks))
        print('{}-way {}-shot fit_kmeans finetuned: {:.5} with 95% CI {:.5} over {} tests'.format(num_classes, num_shots, np.mean(finetuned_accuracies), 1.96*np.std(finetuned_accuracies)/np.sqrt(num_tasks), num_tasks))
        print('{} few-shot classification tasks: {:.5} seconds with {} degenerate tasks.'.format(num_tasks, time.time() - start, num_degenerate_tasks))


if __name__ == '__main__':
    if FLAGS.algorithm == 'embedding_nearest_neighbour':
        embedding_nearest_neighbour()
    elif FLAGS.algorithm == 'embedding_logistic_regression':
        embedding_logistic_regression()
    elif FLAGS.algorithm == 'embedding_cluster_matching':
        embedding_cluster_matching()
    elif FLAGS.algorithm == 'embedding_mlp':
        embedding_mlp()
    else:
        raise ValueError()