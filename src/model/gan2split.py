'''
Generate synthetic multi-label samples using GAN while
performing stratified data splitting
'''

import collections
import logging
import math
import os
import pickle as pkl
import random
import sys
import time
import warnings

import numpy as np
import tensorflow as tf
import tqdm
from joblib import Parallel, delayed
from scipy import sparse
from scipy.sparse import lil_matrix
from tensorflow.keras import layers

from src.utility.file_path import DATASET_PATH, LOG_PATH, RESULT_PATH
from src.utility.utils import check_type, softmax, LabelBinarizer

np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)


class GANStratification(object):
    def __init__(self, num_subsamples: int = 10000, num_clusters: int = 5, num_examples2gen=20,
                 dimension_size: int = 50, sigma: float = 2, lambda_dis=1e-5, lambda_gen=1e-5, shuffle: bool = True,
                 split_size: float = 0.75, update_ratio=1, batch_size: int = 100, num_epochs: int = 5, max_iter_gen=30,
                 max_iter_dis=30, window_size=2, lr: float = 1e-3, num_jobs: int = 2, display_interval=30):

        """Clustering based stratified based multi-label data splitting.

        Parameters
        ----------
        num_clusters : int, default=5
            The number of communities to form. It should be greater than 1.

        shuffle : bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None.

        split_size : float, default=0.75
            It should be between 0.0 and 1.0 and represents the proportion
            of data to include in the test split.

        batch_size : int, default=100
            It should be a positive integer and represents the size of
            batch during splitting process.

        num_epochs : int, default=50
            The number of iterations of the k-means algorithm to run.

        lr : float, default=0.0001
            Learning rate.

        num_jobs : int, default=2
            The number of parallel jobs to run for splitting.
            ``-1`` means using all processors.

        lambda_gen = 1e-5  # l2 loss regulation weight for the generator
        lambda_dis = 1e-5  # l2 loss regulation weight for the discriminator
        num_examples2gen = 20  # number of samples for the generator
        max_iter_gen = 30  # number of inner loops for the generator
        max_iter_dis = 30  # number of inner loops for the discriminator
        display_interval = 30  # sample new nodes for the discriminator for every dis_interval iterations
        update_ratio = 1  # updating ratio when choose the trees
        window_size = 2
        """

        self.num_examples2gen = 20,
        self.lambda_dis = 1e-5
        self.lambda_gen = 1e-5
        self.update_ratio = 1
        self.max_iter_gen = 30,
        self.max_iter_dis = 30
        self.window_size = 2
        self.display_interval = 30

        if dimension_size < 2:
            dimension_size = 50
        self.dimension_size = dimension_size

        if num_subsamples < 100:
            num_subsamples = 100
        self.num_subsamples = num_subsamples

        if num_clusters < 1:
            num_clusters = 5
        self.num_clusters = num_clusters

        if sigma < 0.0:
            sigma = 2
        self.sigma = sigma

        self.shuffle = shuffle

        if split_size >= 1.0 or split_size <= 0.0:
            split_size = 0.8
        self.split_size = split_size

        if batch_size <= 0:
            batch_size = 100
        self.batch_size = batch_size

        if num_epochs <= 0:
            num_epochs = 100
        self.num_epochs = num_epochs

        if num_epochs <= 0:
            num_epochs = 5
        self.num_epochs = num_epochs

        if lr <= 0.0:
            lr = 0.0001
        self.lr = lr

        if num_jobs <= 0:
            num_jobs = 2
        self.num_jobs = num_jobs
        self.is_fit = False

        warnings.filterwarnings("ignore", category=Warning)

        self.__print_arguments()
        time.sleep(2)

    def __print_arguments(self, **kwargs):
        desc = "## Split multi-label data using clustering based approach..."
        print(desc)

        argdict = dict()
        argdict.update({'num_clusters': 'Number of clusters to form: {0}'.format(self.num_clusters)})
        argdict.update({'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update({'split_size': 'Split size: {0}'.format(self.split_size)})
        argdict.update({'batch_size': 'Number of examples to use in '
                                      'each iteration: {0}'.format(self.batch_size)})
        argdict.update({'num_epochs': 'Number of loops over training set: {0}'.format(self.num_epochs)})
        argdict.update({'lr': 'Learning rate: {0}'.format(self.lr)})
        argdict.update({'num_jobs': 'Number of parallel workers: {0}'.format(self.num_jobs)})

        for key, value in kwargs.items():
            argdict.update({key: value})
        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print('\t>> The following arguments are applied:\n\t\t{0}'.format(args), file=sys.stderr)

    def __normalize_laplacian(self, A, return_adj=False, norm_adj=False):
        """Normalize graph Laplacian matrix.

        Parameters
        ----------
        A : {array-like, sparse matrix} of shape (n_labels, n_labels)
            Matrix `A`.

        return_adj : bool, default=False
            Whether or not to return adjacency matrix or normalized Laplacian matrix.

        norm_adj : bool, default=False
            Whether or not to normalize adjacency matrix.

        Returns
        -------
        clusters labels : a list of clusters defining a cluster to a label association
        """

        def __scale_diagonal(D):
            D = D.sqrt()
            D = D.power(-1)
            return D

        A.setdiag(values=0)
        D = lil_matrix(A.sum(axis=1))
        D = D.multiply(sparse.eye(D.shape[0]))
        if return_adj:
            if norm_adj:
                D_inv = D.power(-0.5)
                A = D_inv.dot(A).dot(D_inv)
            return A
        else:
            L = D - A
            D = __scale_diagonal(D=D) / self.sigma
            return D.dot(L.dot(D))

    def __parallel_split(self, examples, check_list):
        """Online or batch based strategy to splitting multi-label dataset
        into train and test subsets.

        Parameters
        ----------
        examples : a list containing indices of label dataset.

        check_list : a list holding long term data indices that were included
                     in the ongoing data split process (either train or test data).

        Returns
        -------
        data partition : two lists of the resulted split data
        """

        temp_dict = dict({0: [], 1: []})
        examples_batch = [i for i in examples if i not in check_list]
        if self.shuffle:
            random.shuffle(examples_batch)
        sample_size = len(examples_batch)
        if sample_size == 0:
            return sorted(temp_dict[0]), sorted(temp_dict[1])
        else:
            check_list.extend(examples_batch)
            if sample_size > 0:
                j = round(sample_size * self.split_size)
                for k in range(2):
                    temp = temp_dict[k] + examples_batch[:j]
                    temp = list(set(temp))
                    temp_dict.update({k: temp})
                    examples_batch = examples_batch[j:]
                    del temp
            else:
                temp = temp_dict[0] + examples_batch
                temp = list(set(temp))
                temp_dict.update({0: temp})
        return sorted(temp_dict[0]), sorted(temp_dict[1])

    def __parallel_build_trees(self, trees, root, graph):
        trees[root] = {}
        trees[root][root] = [root]
        used_nodes = set()
        queue = collections.deque([root])
        while len(queue) > 0:
            cur_node = queue.popleft()
            used_nodes.add(cur_node)
            for sub_node in graph[cur_node]:
                if sub_node not in used_nodes:
                    trees[root][cur_node].append(sub_node)
                    trees[root][sub_node] = [cur_node]
                    queue.append(sub_node)
                    used_nodes.add(sub_node)

    def __build_trees(self, nodes, graph=None):
        """use BFS algorithm to construct the BFS-trees

        Args:
            graph:
            nodes: the list of nodes in the graph
        Returns:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
        """

        trees = {}
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads", verbose=0)
        parallel(delayed(self.__parallel_build_trees)(trees, root, graph)
                 for root in tqdm.tqdm(nodes))
        return trees

    def __build_layers(self, num_nodes, name="generator"):
        """initializing the generator

        Args:
            embeddings:
        """
        bound = 1.0 / math.sqrt(self.dimension_size)
        embeddings_initializer = tf.keras.initializers.truncated_normal(mean=0, stddev=bound,
                                                                        seed=12345)
        input_layer = layers.Input(shape=[num_nodes])
        embedding_layer = layers.Embedding(input_dim=num_nodes, output_dim=self.dimension_size,
                                           embeddings_initializer=embeddings_initializer, name="embedding")(input_layer)
        bias_layer = layers.Embedding(input_dim=num_nodes, output_dim=1,
                                      embeddings_initializer='zeros', name="bias")(input_layer)
        cat = layers.Concatenate()([embedding_layer, bias_layer])
        norm = layers.BatchNormalization()(cat)
        relu = layers.LeakyReLU()(norm)
        model = tf.keras.Model(inputs=[input_layer], outputs=[relu], name=name)
        return model

    def __discriminator_loss(self, model, node_id, node_neighbor_id, label=None, calc_reward=False):
        node_embedding = tf.nn.embedding_lookup(model.layers[1].embeddings, node_id)
        node_neighbor_embedding = tf.nn.embedding_lookup(model.layers[1].embeddings, node_neighbor_id)

        bias = tf.nn.embedding_lookup(model.layers[2].embeddings, node_neighbor_id)
        bias = tf.reshape(bias, bias.shape[0])
        score = tf.reduce_sum(tf.multiply(node_embedding, node_neighbor_embedding), axis=1) + bias
        if calc_reward:
            score = tf.clip_by_value(score, clip_value_min=-10, clip_value_max=10)
            reward = tf.math.log(1 + tf.exp(score))
            return reward
        else:
            label = tf.Variable(label, dtype=tf.float32)
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(label, score)) + self.lambda_dis * (
                    tf.nn.l2_loss(node_neighbor_embedding) + tf.nn.l2_loss(node_embedding) + tf.nn.l2_loss(bias))
            return loss

    def __generator_loss(self, model, node_id, node_neighbor_id, reward):
        reward = tf.Variable(reward, dtype=tf.float32)
        node_embedding = tf.nn.embedding_lookup(model.layers[1].embeddings, node_id)
        node_neighbor_embedding = tf.nn.embedding_lookup(model.layers[1].embeddings, node_neighbor_id)

        bias = tf.nn.embedding_lookup(model.layers[2].embeddings, node_neighbor_id)
        bias = tf.reshape(bias, bias.shape[0])
        score = tf.reduce_sum(tf.multiply(node_embedding, node_neighbor_embedding), axis=1) + bias
        prob = tf.clip_by_value(tf.nn.sigmoid(score), 1e-5, 1)
        loss = -tf.reduce_mean(tf.math.log(prob) * reward) + self.lambda_gen * (
                tf.nn.l2_loss(node_neighbor_embedding) + tf.nn.l2_loss(node_embedding))
        return loss

    def __optimizer(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        return optimizer

    def __generator_embed_score(self, model):
        temp = tf.matmul(model.layers[1].embeddings, model.layers[1].embeddings, transpose_b=True) + model.layers[
            2].embeddings
        return temp

    def __sample(self, weight_score, root, tree, sample_num, for_d):
        """ sample nodes from BFS-tree

        Args:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            for_d: bool, whether the samples are used for the generator or the discriminator
        Returns:
            samples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """

        samples = []
        paths = []
        n = 0

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                if for_d:  # skip 1-hop nodes (positive samples)
                    if node_neighbor == [root]:
                        # in current version, None is returned for simplicity
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                relevance_probability = weight_score[current_node].numpy()[node_neighbor].tolist()
                relevance_probability = softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths

    def __get_node_pairs_from_path(self, path):
        """
        given a path from root to a sampled node, generate all the node pairs within the given windows size
        e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
        node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
        :param path: a path from root to the sampled node
        :return pairs: a list of node pairs
        """

        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - self.window_size, 0), min(i + self.window_size + 1, len(path))):
                if i == j:
                    continue
                node = path[j]
                pairs.append([center_node, node])
        return pairs

    def __sample_generator(self, model, weight_score, root_nodes, trees):
        """sample nodes for the generator"""

        paths = []
        for i in root_nodes:
            if np.random.rand() < self.update_ratio:
                sample, paths_from_i = self.__sample(weight_score, i, trees[i], self.num_examples2gen, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        node_pairs = list(map(self.__get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])
        reward = self.__discriminator_loss(model=model, node_id=np.array(node_1), node_neighbor_id=np.array(node_2),
                                           calc_reward=True)
        return node_1, node_2, reward

    def __sample_discriminator(self, weight_score, root_nodes, graph, trees):
        """generate positive and negative samples for the discriminator, and record them in the txt file

        Args:
            model:
        """
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for i in root_nodes:
            if np.random.rand() < self.update_ratio:
                pos = graph[i]
                neg, _ = self.__sample(weight_score, i, trees[i], len(pos), for_d=True)
                if len(pos) != 0 and neg is not None:
                    # positive samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))

                    # negative samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
        return center_nodes, neighbor_nodes, labels

    def __train_gan(self, generator, discriminator, root_nodes, graph, trees, checkpoint, checkpoint_prefix):
        num_epochs = self.num_epochs + 1
        total_epochs = self.num_epochs * self.max_iter_dis * self.max_iter_gen
        for epoch in range(1, num_epochs):
            # D-steps
            center_nodes = []
            neighbor_nodes = []
            labels = []
            for idx in range(self.max_iter_dis):
                # generate new nodes for the discriminator for every dis_interval iterations
                if idx % self.display_interval == 0:
                    weight_score = self.__generator_embed_score(model=generator)
                    center_nodes, neighbor_nodes, labels = self.__sample_discriminator(weight_score=weight_score,
                                                                                       root_nodes=root_nodes,
                                                                                       graph=graph, trees=trees)
                    checkpoint.save(file_prefix=checkpoint_prefix)

                # training
                list_batches = list(range(0, len(center_nodes), self.batch_size))
                random.shuffle(list_batches)
                for start in list_batches:
                    with tf.GradientTape() as tape:
                        end = start + self.batch_size
                        loss = self.__discriminator_loss(model=discriminator, node_id=np.array(center_nodes[start:end]),
                                                         node_neighbor_id=np.array(neighbor_nodes[start:end]),
                                                         label=np.array(labels[start:end]))
                        gradients = tape.gradient(loss, discriminator.trainable_variables)
                        optimizer = self.__optimizer()
                        optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

            # G-steps
            first_node = []
            second_node = []
            reward = []
            for idx in range(self.max_iter_gen):
                if idx % self.display_interval == 0:
                    weight_score = self.__generator_embed_score(model=generator)
                    first_node, second_node, reward = self.__sample_generator(model=discriminator,
                                                                              weight_score=weight_score,
                                                                              root_nodes=root_nodes,
                                                                              trees=trees)
                    checkpoint.save(file_prefix=checkpoint_prefix)

                # training
                list_batches = list(range(0, len(first_node), self.batch_size))
                random.shuffle(list_batches)
                for start in list_batches:
                    with tf.GradientTape() as tape:
                        end = start + self.batch_size
                        loss = self.__generator_loss(model=generator,
                                                     node_id=np.array(first_node[start:end]),
                                                     node_neighbor_id=np.array(second_node[start:end]),
                                                     reward=np.array(reward[start:end]))
                        gradients = tape.gradient(loss, generator.trainable_variables)
                        optimizer = self.__optimizer()
                        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
            current_epoch = epoch * self.max_iter_dis * self.max_iter_gen
            desc = '\t\t--> Learning progress: {0:.2f}%...'.format((current_epoch / total_epochs) * 100)
            if epoch + 1 == total_epochs:
                print(desc)
            else:
                print(desc, end="\r")

    def fit(self, X, y):
        """Split multi-label y dataset into train and test subsets.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Matrix `X`.

        y : {array-like, sparse matrix} of shape (n_samples, n_labels)
            Matrix `y`.

        Returns
        -------
        data partition : two lists of the resulted data split
        """

        check, X = check_type(X)
        if not check:
            tmp = "The method only supports scipy.sparse and numpy.ndarray type of data"
            raise Exception(tmp)

        num_examples, num_features, num_labels = X.shape[0], X.shape[1], y.shape[1]

        # check whether data is singly labeled
        if num_labels == 1:
            # transform it to multi-label data
            classes = list(set([i[0] if i else 0 for i in X.data]))
            mlb = LabelBinarizer(labels=classes)
            X = mlb.transform(X)
            num_labels = len(classes)

        if not self.is_fit:
            desc = '\t>> Building Graph...'
            print(desc)
            # Construct graph
            idx = np.random.choice(a=list(range(num_examples)), size=self.num_subsamples, replace=True)
            A = y[idx].T.dot(y[idx])
            A = self.__normalize_laplacian(A=A, return_adj=True, norm_adj=True)
            A[A <= 0.05] = 0.0
            A = lil_matrix(A)
            graph = {i: list(A[i].nonzero()[1]) for i in range(A.shape[0])}
            # construct or read BFS-trees
            root_nodes = [i for i in range(num_labels)]
            trees = self.__build_trees(nodes=root_nodes, graph=graph)

        desc = '\t>> Building GAN model...'
        print(desc)
        generator = self.__build_layers(num_nodes=num_labels, name="generator")
        discriminator = self.__build_layers(num_nodes=num_labels, name="discriminator")

        checkpoint_prefix = os.path.join(LOG_PATH, "gan2split_ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.__optimizer(),
                                         discriminator_optimizer=self.__optimizer(),
                                         generator=generator,
                                         discriminator=discriminator)

        desc = '\t\t--> Training GAN model...'
        print(desc)
        self.__train_gan(generator=generator, discriminator=discriminator, root_nodes=root_nodes, graph=graph,
                         trees=trees,
                         checkpoint=checkpoint, checkpoint_prefix=checkpoint_prefix)
        self.__save_embeddings(generator=generator, discriminator=discriminator)

        desc = '\t>> Stratified Split...'
        print(desc)
        check_list = list()
        train_list = list()
        test_list = list()
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads", verbose=0)
        # Find the label with the fewest (but at least one) remaining
        # examples, breaking ties randomly
        for label_idx in range(num_labels):
            examples = list(X[:, label_idx].nonzero()[0])
            if len(examples) == 0:
                continue
            list_batches = list(range(0, len(examples), self.batch_size))
            results = parallel(delayed(self.__parallel_split)(examples[batch_idx:batch_idx + self.batch_size],
                                                              check_list)
                               for idx, batch_idx in enumerate(list_batches))
            desc = '\t\t--> Splitting progress: {0:.2f}%...'.format(((label_idx + 1) / num_labels) * 100)
            if label_idx + 1 == num_labels:
                print(desc)
            else:
                print(desc, end="\r")
            results = list(zip(*results))
            train_list.extend([i for item in results[0] for i in item])
            test_list.extend([i for item in results[1] for i in item])
            del results
        del check_list
        self.is_fit = True
        return sorted(train_list), sorted(test_list)

    def __save_embeddings(self, generator, discriminator):
        """write embeddings of the generator and the discriminator to files"""

        modes = [generator, discriminator]
        emb_filenames = [os.path.join(RESULT_PATH, "generator.txt"), os.path.join(RESULT_PATH, "discriminator.txt")]
        for i in range(2):
            embedding_matrix = modes[i].layers[1].embeddings
            embedding_list = embedding_matrix.numpy().tolist()
            embedding_str = [str(idx) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for idx, emb in enumerate(embedding_list)]
            with open(emb_filenames[i], "w+") as f:
                lines = [str(len(embedding_list)) + "\t" + str(self.dimension_size) + "\n"] + embedding_str
                f.writelines(lines)


if __name__ == "__main__":
    X_name = "Xbirds_train.pkl"
    y_name = "Ybirds_train.pkl"

    file_path = os.path.join(DATASET_PATH, X_name)
    with open(file_path, mode="rb") as f_in:
        X = pkl.load(f_in)

    file_path = os.path.join(DATASET_PATH, y_name)
    with open(file_path, mode="rb") as f_in:
        y = pkl.load(f_in)

    st = GANStratification(num_clusters=5, shuffle=True, split_size=0.8, batch_size=100, num_epochs=1, lr=0.0001,
                           num_jobs=2)
    training_set, test_set = st.fit(X=X, y=y)
    training_set, dev_set = st.fit(X=X[training_set], y=y[training_set])

    print("training set size: {0}".format(len(training_set)))
    print("validation set size: {0}".format(len(dev_set)))
    print("test set size: {0}".format(len(test_set)))
