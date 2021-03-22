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
import textwrap
import time
import warnings

import numpy as np
import tensorflow as tf
import tqdm
from joblib import Parallel, delayed
from tensorflow.keras import layers

from src.model.utils import DATASET_PATH, LOG_PATH, RESULT_PATH
from src.model.utils import check_type, softmax

np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)


class GAN2Embed(object):
    def __init__(self, dimension_size: int = 50, num_examples2gen=20, update_ratio=1, window_size=2,
                 shuffle: bool = True, batch_size: int = 100, num_epochs: int = 5, max_iter_gen=30,
                 max_iter_dis=30, lambda_gen=1e-5, lambda_dis=1e-5, lr: float = 1e-3,
                 display_interval: int = 5, num_jobs: int = 2, verbose: bool = True):

        """Clustering based stratified based multi-label data splitting.

        Parameters
        ----------
        dimension_size : int, default=50
            The dimension size of embeddings.

        num_examples2gen : int, default=20
            The number of samples for the generator.

        update_ratio : int, default=1
            Updating ratio when choose the trees.

        window_size : int, default=2
            Window size to skip.

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

        max_iter_gen : int, default=30
            The number of inner loops for the generator.

        max_iter_dis : int, default=30
            The number of inner loops for the discriminator.

        lambda_gen : float, default=1e-5
            The l2 loss regulation weight for the generator.

        lambda_dis : float, default=1e-5
            The l2 loss regulation weight for the discriminator.

        lr : float, default=0.0001
            Learning rate.

        display_interval : int, default=5
            Sample new nodes for the discriminator for every display
            interval iterations.

        num_jobs : int, default=2
            The number of parallel jobs to run for splitting.
            ``-1`` means using all processors.

        verbose : bool, default=True
            Display arguments.
        """

        if dimension_size < 2:
            dimension_size = 50
        self.dimension_size = dimension_size

        if num_examples2gen < 2:
            num_examples2gen = 20
        self.num_examples2gen = num_examples2gen

        if update_ratio <= 0:
            update_ratio = 1
        self.update_ratio = update_ratio

        if window_size <= 0:
            window_size = 2
        self.window_size = window_size

        self.shuffle = shuffle

        if batch_size <= 0:
            batch_size = 100
        self.batch_size = batch_size

        if num_epochs <= 0:
            num_epochs = 100
        self.num_epochs = num_epochs

        if max_iter_dis <= 0:
            max_iter_dis = 30
        self.max_iter_dis = max_iter_dis

        if max_iter_gen <= 0:
            max_iter_gen = 30
        self.max_iter_gen = max_iter_gen

        if lambda_gen <= 0.0:
            lambda_gen = 1e-5
        self.lambda_gen = lambda_gen

        if lambda_dis <= 0.0:
            lambda_dis = 1e-5
        self.lambda_dis = lambda_dis

        if lr <= 0.0:
            lr = 0.0001
        self.lr = lr

        if display_interval <= 0:
            display_interval = 5
        self.display_interval = display_interval

        if num_jobs <= 0:
            num_jobs = 2
        self.num_jobs = num_jobs
        self.is_fit = False

        warnings.filterwarnings("ignore", category=Warning)

        if verbose:
            self.__print_arguments()
            time.sleep(2)

    def __print_arguments(self, **kwargs):
        desc = "## Configuration parameters for GAN:"
        argdict = dict()
        argdict.update({'dimension_size': 'The dimension size of embeddings: {0}'.format(self.dimension_size)})
        argdict.update(
            {'num_examples2gen': 'The number of samples for the generator.: {0}'.format(self.num_examples2gen)})
        argdict.update({'update_ratio': 'Updating ratio when choose the trees: {0}'.format(self.update_ratio)})
        argdict.update({'window_size': 'Window size to skip.: {0}'.format(self.window_size)})
        argdict.update({'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update({'batch_size': 'Number of examples to use in '
                                      'each iteration: {0}'.format(self.batch_size)})
        argdict.update({'num_epochs': 'Number of loops over training set: {0}'.format(self.num_epochs)})
        argdict.update({'max_iter_gen': 'The number of inner loops for the generator: {0}'.format(self.max_iter_gen)})
        argdict.update(
            {'max_iter_dis': 'The number of inner loops for the discriminator: {0}'.format(self.max_iter_dis)})
        argdict.update({'lambda_gen': 'The l2 loss regulation weight for the generator: {0}'.format(self.lambda_gen)})
        argdict.update(
            {'lambda_dis': 'The l2 loss regulation weight for the discriminator: {0}'.format(self.lambda_dis)})
        argdict.update({'lr': 'Learning rate: {0}'.format(self.lr)})
        argdict.update({
            'display_interval': 'Sample new nodes for the discriminator for every display_interval iterations: {0}'.format(
                self.display_interval)})
        argdict.update({'num_jobs': 'Number of parallel workers: {0}'.format(self.num_jobs)})

        for key, value in kwargs.items():
            argdict.update({key: value})
        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print(textwrap.TextWrapper(width=75, subsequent_indent='   ').fill(desc), file=sys.stderr)
        print('\t\t{0}'.format(args), file=sys.stderr)

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

    def __build_trees(self, trees, root, graph):
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

    def __parallel_build_trees(self, nodes, graph=None):
        """use BFS algorithm to construct the BFS-trees

        Args:
            graph:
            nodes: the list of nodes in the graph
        Returns:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
        """

        trees = {}
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads", verbose=0)
        parallel(delayed(self.__build_trees)(trees, root, graph)
                 for root in tqdm.tqdm(nodes))
        return trees

    def __build_layers(self, num_nodes, name="generator"):
        """initializing the generator
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

    def __discriminator_loss(self, model, node_id, node_neighbor_id, label=None, calc_score=False):
        node_embedding = tf.nn.embedding_lookup(model.layers[1].embeddings, node_id)
        node_neighbor_embedding = tf.nn.embedding_lookup(model.layers[1].embeddings, node_neighbor_id)

        bias = tf.nn.embedding_lookup(model.layers[2].embeddings, node_neighbor_id)
        bias = tf.reshape(bias, bias.shape[0])
        score = tf.reduce_sum(tf.multiply(node_embedding, node_neighbor_embedding), axis=1) + bias
        if calc_score:
            score = tf.clip_by_value(score, clip_value_min=-10, clip_value_max=10)
            score = tf.math.log(1 + tf.exp(score))
            return score
        else:
            label = tf.Variable(label, dtype=tf.float32)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(label, score)) + self.lambda_dis * (
                    tf.nn.l2_loss(node_neighbor_embedding) + tf.nn.l2_loss(node_embedding) + tf.nn.l2_loss(bias))
            return loss

    def __generator_loss(self, model, node_id, node_neighbor_id, disc_score):
        disc_score = tf.Variable(disc_score, dtype=tf.float32)
        node_embedding = tf.nn.embedding_lookup(model.layers[1].embeddings, node_id)
        node_neighbor_embedding = tf.nn.embedding_lookup(model.layers[1].embeddings, node_neighbor_id)

        bias = tf.nn.embedding_lookup(model.layers[2].embeddings, node_neighbor_id)
        bias = tf.reshape(bias, bias.shape[0])
        score = tf.reduce_sum(tf.multiply(node_embedding, node_neighbor_embedding), axis=1) + bias
        prob = tf.clip_by_value(tf.nn.sigmoid(score), 1e-5, 1)
        loss = -tf.reduce_mean(tf.math.log(prob) * disc_score) + self.lambda_gen * (
                tf.nn.l2_loss(node_neighbor_embedding) + tf.nn.l2_loss(node_embedding) + tf.nn.l2_loss(bias))
        return loss

    def __optimizer(self):
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        return optimizer

    def __generator_embed_score(self, model):
        temp = tf.matmul(model.layers[1].embeddings, model.layers[1].embeddings, transpose_b=True) + model.layers[
            2].embeddings
        return temp

    def __sample(self, weight_score, node_idx, tree, num_examples, for_d):
        """ Sample nodes from BFS-tree

        Args:
            node_idx: int, root node
            tree: dict, BFS-tree
            num_examples: the number of required examples
            for_d: bool, whether the examples are used for the generator or the discriminator
        Returns:
            examples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """

        examples = list()
        paths = list()
        n = 0
        while len(examples) < num_examples:
            current_node = node_idx
            previous_node = -1
            paths.append(list())
            is_root = True
            paths[n].append(current_node)
            while True:
                neighbor2current_node = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(neighbor2current_node) == 0:  # the tree only has a root
                    return None, None
                if for_d:  # skip 1-hop nodes (positive examples)
                    if neighbor2current_node == [node_idx]:
                        # in current version, None is returned for simplicity
                        return None, None
                    if node_idx in neighbor2current_node:
                        neighbor2current_node.remove(node_idx)
                relevance_probability = weight_score[current_node].numpy()[neighbor2current_node].tolist()
                relevance_probability = softmax(relevance_probability)
                next_node = np.random.choice(neighbor2current_node, size=1, p=relevance_probability)[
                    0]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:
                    # TODO: this has negative consequences where nodes with
                    #  not satisfying this condition may be eliminated to
                    #  contructing samples
                    examples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return examples, paths

    def __get_node_pairs_from_path(self, path):
        """
        given a path from root to a sampled node, generate all the node pairs within the given windows size
        e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
        node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
        :param path: a path from root to the sampled node
        :return pairs: a list of node pairs
        """

        path = path[:-1]
        pairs = list()
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

        paths = list()
        for node_idx in root_nodes:
            if np.random.rand() < self.update_ratio:
                temp, paths_from_node_idx = self.__sample(weight_score=weight_score, node_idx=node_idx,
                                                          tree=trees[node_idx], num_examples=self.num_examples2gen,
                                                          for_d=False)
                del temp
                if paths_from_node_idx is not None:
                    paths.extend(paths_from_node_idx)
        node_pairs = list(map(self.__get_node_pairs_from_path, paths))
        node_1 = list()
        node_2 = list()
        for node_idx in range(len(node_pairs)):
            for pair in node_pairs[node_idx]:
                node_1.append(pair[0])
                node_2.append(pair[1])
        score = self.__discriminator_loss(model=model, node_id=np.array(node_1), node_neighbor_id=np.array(node_2),
                                          calc_score=True)
        return node_1, node_2, score

    def __sample_discriminator(self, weight_score, root_nodes, graph, trees):
        """generate positive and negative samples for the discriminator, and record them in the txt file
        """
        center_nodes = list()
        neighbor_nodes = list()
        labels = list()
        for node_idx in root_nodes:
            if np.random.rand() < self.update_ratio:
                positive = graph[node_idx]
                negative, temp = self.__sample(weight_score=weight_score, node_idx=node_idx, tree=trees[node_idx],
                                               num_examples=len(positive), for_d=True)
                del temp
                if len(positive) != 0 and negative is not None:
                    # positive samples
                    center_nodes.extend([node_idx] * len(positive))
                    neighbor_nodes.extend(positive)
                    labels.extend([0.9] * len(positive))

                    # negative samples
                    center_nodes.extend([node_idx] * len(positive))
                    neighbor_nodes.extend(negative)
                    labels.extend([0.1] * len(negative))
        return center_nodes, neighbor_nodes, labels

    def __train_gan(self, generator, discriminator, root_nodes, graph, trees, checkpoint, checkpoint_prefix):
        num_epochs = self.num_epochs + 1
        total_epochs = self.num_epochs * self.max_iter_dis * self.max_iter_gen
        for epoch in range(1, num_epochs):
            # D-steps
            center_nodes = list()
            neighbor_nodes = list()
            labels = list()
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
                for start_idx in list_batches:
                    with tf.GradientTape() as tape:
                        final_idx = start_idx + self.batch_size
                        loss = self.__discriminator_loss(model=discriminator,
                                                         node_id=np.array(center_nodes[start_idx:final_idx]),
                                                         node_neighbor_id=np.array(neighbor_nodes[start_idx:final_idx]),
                                                         label=np.array(labels[start_idx:final_idx]))
                        gradients = tape.gradient(loss, discriminator.trainable_variables)
                        optimizer = self.__optimizer()
                        optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))

            # G-steps
            first_node = list()
            second_node = list()
            disc_score = list()
            for idx in range(self.max_iter_gen):
                if idx % self.display_interval == 0:
                    weight_score = self.__generator_embed_score(model=generator)
                    first_node, second_node, disc_score = self.__sample_generator(model=discriminator,
                                                                                  weight_score=weight_score,
                                                                                  root_nodes=root_nodes,
                                                                                  trees=trees)
                    checkpoint.save(file_prefix=checkpoint_prefix)

                # training
                list_batches = list(range(0, len(first_node), self.batch_size))
                random.shuffle(list_batches)
                for start_idx in list_batches:
                    with tf.GradientTape() as tape:
                        final_idx = start_idx + self.batch_size
                        loss = self.__generator_loss(model=generator,
                                                     node_id=np.array(first_node[start_idx:final_idx]),
                                                     node_neighbor_id=np.array(second_node[start_idx:final_idx]),
                                                     disc_score=np.array(disc_score[start_idx:final_idx]))
                        gradients = tape.gradient(loss, generator.trainable_variables)
                        optimizer = self.__optimizer()
                        optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
            current_epoch = epoch * self.max_iter_dis * self.max_iter_gen
            desc = '\t\t--> Learning progress: {0:.2f}%...'.format((current_epoch / total_epochs) * 100)
            if epoch + 1 == total_epochs:
                print(desc)
            else:
                print(desc, end="\r")

    def fit(self, adjacency):
        """Extract embeddings using GAN.

        Parameters
        ----------
        adjacency : an adjacency {sparse matrix} of shape (num_nodes, num_nodes).

        Returns
        -------
        models : generator and discriminator models
        """

        check, adjacency = check_type(adjacency, False)
        if not check:
            tmp = "The method only supports scipy.sparse and numpy.ndarray type of data"
            raise Exception(tmp)

        num_nodes = adjacency.shape[0]

        desc = '\t>> Building BFS-trees...'
        print(desc)
        # construct BFS-trees
        graph = {i: list(adjacency[i].nonzero()[1]) for i in range(num_nodes)}
        root_nodes = [i for i in range(num_nodes)]
        trees = self.__parallel_build_trees(nodes=root_nodes, graph=graph)

        desc = '\t>> Building GAN model...'
        print(desc)
        generator = self.__build_layers(num_nodes=num_nodes, name="generator")
        discriminator = self.__build_layers(num_nodes=num_nodes, name="discriminator")

        checkpoint_prefix = os.path.join(LOG_PATH, "gan2split_ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.__optimizer(),
                                         discriminator_optimizer=self.__optimizer(),
                                         generator=generator,
                                         discriminator=discriminator)

        desc = '\t>> Training GAN model...'
        print(desc)
        self.__train_gan(generator=generator, discriminator=discriminator, root_nodes=root_nodes, graph=graph,
                         trees=trees, checkpoint=checkpoint, checkpoint_prefix=checkpoint_prefix)
        self.__save_embeddings(generator=generator, discriminator=discriminator)
        return generator, discriminator


if __name__ == "__main__":
    X_name = "Xbirds_train.pkl"
    y_name = "Ybirds_train.pkl"
    use_extreme = True

    file_path = os.path.join(DATASET_PATH, y_name)
    with open(file_path, mode="rb") as f_in:
        y = pkl.load(f_in)
        idx = list(set(y.nonzero()[0]))
        y = y[idx]

    file_path = os.path.join(DATASET_PATH, X_name)
    with open(file_path, mode="rb") as f_in:
        X = pkl.load(f_in)
        X = X[idx]

    st = GAN2Embed(shuffle=True, batch_size=100, num_epochs=1, lr=0.0001, num_jobs=2)
    st.fit(adjacency=None)
