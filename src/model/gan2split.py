'''
Generate synthetic multi-label samples using GAN while
performing stratified data splitting
'''

import logging
import os
import pickle as pkl
import sys
import textwrap
import time
import warnings

import numpy as np
import tensorflow as tf
from scipy import sparse
from scipy.cluster.vq import kmeans2
from scipy.sparse import lil_matrix

from extreme2split import ExtremeStratification
from gan2embed import GAN2Embed
from iterative2split import IterativeStratification
from naive2split import NaiveStratification
from utils import check_type, data_properties, LabelBinarizer

np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)


class GANStratification(object):
    def __init__(self, dimension_size: int = 50, num_examples2gen: int = 20, update_ratio: int = 1,
                 window_size: int = 2, num_subsamples: int = 10000, num_clusters: int = 5, sigma: float = 2,
                 swap_probability: float = 0.1, threshold_proportion: float = 0.1, decay: float = 0.1,
                 shuffle: bool = True, split_size: float = 0.75, batch_size: int = 100, max_iter_gen: int = 30,
                 max_iter_dis: int = 30, num_epochs: int = 5, lambda_gen: float = 1e-5, lambda_dis: float = 1e-5,
                 lr: float = 1e-3, display_interval=30, num_jobs: int = 2):

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

        num_subsamples : int, default=10000
            The number of subsamples to use for detecting communities.
            It should be greater than 100.

        num_clusters : int, default=5
            The number of communities to form. It should be greater than 1.

        sigma : float, default=2.0
            Scaling component to the graph degree matrix.
            It should be greater than 0.

        swap_probability : float, default=0.1
            A hyper-parameter for stratification.

        threshold_proportion : float, default=0.1
            A hyper-parameter for stratification.

        decay : float, default=0.1
            A hyper-parameter for stratification.

        shuffle : bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None.

        split_size : float, default=0.75
            It should be between 0.0 and 1.0 and represents the proportion
            of data to include in the test split.

        batch_size : int, default=100
            It should be a positive integer and represents the size of
            batch during splitting process.

        max_iter_gen : int, default=30
            The number of inner loops for the generator.

        max_iter_dis : int, default=30
            The number of inner loops for the discriminator.

        num_epochs : int, default=50
            The number of iterations of the k-means algorithm to run.

        lambda_gen : float, default=1e-5
            The l2 loss regulation weight for the generator.

        lambda_dis : float, default=1e-5
            The l2 loss regulation weight for the discriminator.

        lr : float, default=0.0001
            Learning rate.

        display_interval : int, default=30
            Sample new nodes for the discriminator for every display interval iterations.

        num_jobs : int, default=2
            The number of parallel jobs to run for splitting.
            ``-1`` means using all processors.
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

        if num_subsamples < 100:
            num_subsamples = 10000
        self.num_subsamples = num_subsamples

        if num_clusters < 1:
            num_clusters = 5
        self.num_clusters = num_clusters

        if sigma < 0.0:
            sigma = 2
        self.sigma = sigma

        if swap_probability < 0.0:
            swap_probability = 0.1
        self.swap_probability = swap_probability

        if threshold_proportion < 0.0:
            threshold_proportion = 0.1
        self.threshold_proportion = threshold_proportion

        if decay < 0.0:
            decay = 0.1
        self.decay = decay

        self.shuffle = shuffle

        if split_size >= 1.0 or split_size <= 0.0:
            split_size = 0.8
        self.split_size = split_size

        if batch_size <= 0:
            batch_size = 100
        self.batch_size = batch_size

        if max_iter_dis <= 0:
            max_iter_dis = 30
        self.max_iter_dis = max_iter_dis

        if max_iter_gen <= 0:
            max_iter_gen = 30
        self.max_iter_gen = max_iter_gen

        if num_epochs <= 0:
            num_epochs = 100
        self.num_epochs = num_epochs

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
            display_interval = 30
        self.display_interval = display_interval

        if num_jobs <= 0:
            num_jobs = 2
        self.num_jobs = num_jobs
        self.is_fit = False

        warnings.filterwarnings("ignore", category=Warning)

        self.__print_arguments()
        time.sleep(2)

    def __print_arguments(self, **kwargs):
        desc = "## Configuration parameters to stratifying a multi-label " \
               "dataset splitting based on clustering embeddings obtained from " \
               "GAN2Embed model:"

        argdict = dict()
        argdict.update({'dimension_size': 'The dimension size of embeddings: {0}'.format(self.dimension_size)})
        argdict.update(
            {'num_examples2gen': 'The number of samples for the generator.: {0}'.format(self.num_examples2gen)})
        argdict.update({'num_subsamples': 'Subsampling input size: {0}'.format(self.num_subsamples)})
        argdict.update({'num_clusters': 'Number of communities: {0}'.format(self.num_clusters)})
        argdict.update({'sigma': 'Constant that scales the amount of '
                                 'laplacian norm regularization: {0}'.format(self.sigma)})
        argdict.update({'update_ratio': 'Updating ratio when choose the trees: {0}'.format(self.display_interval)})
        argdict.update({'window_size': 'Window size to skip.: {0}'.format(self.window_size)})
        argdict.update({'swap_probability': 'A hyper-parameter: {0}'.format(self.swap_probability)})
        argdict.update({'threshold_proportion': 'A hyper-parameter: {0}'.format(self.threshold_proportion)})
        argdict.update({'decay': 'A hyper-parameter: {0}'.format(self.decay)})
        argdict.update({'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update({'split_size': 'Split size: {0}'.format(self.split_size)})
        argdict.update({'batch_size': 'Number of examples to use in '
                                      'each iteration: {0}'.format(self.batch_size)})
        argdict.update({'max_iter_gen': 'The number of inner loops for the '
                                        'generator: {0}'.format(self.max_iter_gen)})
        argdict.update({'max_iter_dis': 'The number of inner loops for the '
                                        'discriminator: {0}'.format(self.max_iter_dis)})
        argdict.update({'num_epochs': 'Number of loops over training set: {0}'.format(self.num_epochs)})
        argdict.update({'lambda_gen': 'The l2 loss regulation weight for '
                                      'the generator: {0}'.format(self.lambda_gen)})
        argdict.update({'lambda_dis': 'The l2 loss regulation weight for '
                                      'the discriminator: {0}'.format(self.lambda_dis)})
        argdict.update({'lr': 'Learning rate: {0}'.format(self.lr)})
        argdict.update({'display_interval': 'Sample new nodes for the '
                                            'discriminator for every '
                                            'disinterval iterations: {0}'.format(self.display_interval)})
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

    def fit(self, X, y, split_type: str = "extreme"):
        """Split multi-label y dataset into train and test subsets.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_examples, n_features).

        y : {array-like, sparse matrix} of shape (n_examples, n_labels).

        split_type : Splitting type of {naive, extreme, iterative}.

        Returns
        -------
        data partition : two lists of indices representing the resulted data split
        """

        if y is None:
            raise Exception("Please provide a dataset.")

        check, y = check_type(y, False)
        if not check:
            tmp = "The method only supports scipy.sparse and numpy.ndarray type of data"
            raise Exception(tmp)

        num_examples, num_labels = y.shape

        # check whether data is singly labeled
        if num_labels == 1:
            # transform it to multi-label data
            classes = list(set([i[0] if i else 0 for i in y.data]))
            mlb = LabelBinarizer(labels=classes)
            y = mlb.transform(y)

        if not self.is_fit:
            desc = '\t>> Building Graph...'
            print(desc)
            # Construct graph
            idx = np.random.choice(a=list(range(num_examples)), size=self.num_subsamples, replace=True)
            A = y[idx].T.dot(y[idx])
            A = self.__normalize_laplacian(A=A, return_adj=True, norm_adj=True)
            A[A <= 0.05] = 0.0
            A = lil_matrix(A)
            gan2embed = GAN2Embed(dimension_size=self.dimension_size, num_examples2gen=self.num_examples2gen,
                                  update_ratio=self.update_ratio, window_size=self.window_size, shuffle=self.shuffle,
                                  batch_size=self.batch_size, num_epochs=self.num_epochs,
                                  max_iter_gen=self.max_iter_gen, max_iter_dis=self.max_iter_dis,
                                  lambda_gen=self.lambda_gen, lambda_dis=self.lambda_dis, lr=self.lr,
                                  display_interval=self.display_interval, num_jobs=self.num_jobs, verbose=False)
            generator, discriminator = gan2embed.fit(adjacency=A)
            del generator

            desc = '\t>> Extracting clusters...'
            print(desc)
            # Construct graph
            _, self.clusters_labels = kmeans2(data=discriminator.weights[0].numpy(), k=self.num_clusters,
                                              iter=self.num_epochs, minit='++')
            del discriminator

        mlb = LabelBinarizer(labels=list(range(self.num_clusters)))
        y = mlb.reassign_labels(y, mapping_labels=self.clusters_labels)
        self.is_fit = True

        # perform splitting
        if split_type == "extreme":
            st = ExtremeStratification(swap_probability=self.swap_probability,
                                       threshold_proportion=self.threshold_proportion, decay=self.decay,
                                       shuffle=self.shuffle, split_size=self.split_size,
                                       num_epochs=self.num_epochs, verbose=False)
            train_list, test_list = st.fit(X=X, y=y)
        elif split_type == "iterative":
            st = IterativeStratification(shuffle=self.shuffle, split_size=self.split_size, verbose=False)
            train_list, test_list = st.fit(y=y)
        else:
            st = NaiveStratification(shuffle=self.shuffle, split_size=self.split_size, batch_size=self.batch_size,
                                     num_jobs=self.num_jobs, verbose=False)
            train_list, test_list = st.fit(y=y)
        return train_list, test_list


if __name__ == "__main__":
    from utils import DATASET_PATH, RESULT_PATH, DATASET

    model_name = "gan2split"
    split_type = "extreme"
    split_size = 0.80
    num_epochs = 5
    num_jobs = 10

    for dsname in DATASET:
        X_name = dsname + "_X.pkl"
        y_name = dsname + "_y.pkl"

        file_path = os.path.join(DATASET_PATH, y_name)
        with open(file_path, mode="rb") as f_in:
            y = pkl.load(f_in)
            idx = list(set(y.nonzero()[0]))
            y = y[idx]

        file_path = os.path.join(DATASET_PATH, X_name)
        with open(file_path, mode="rb") as f_in:
            X = pkl.load(f_in)
            X = X[idx]

        st = GANStratification(dimension_size=50, num_examples2gen=20, update_ratio=1, window_size=2,
                               num_subsamples=10000, num_clusters=5, sigma=2, swap_probability=0.1,
                               threshold_proportion=0.1, decay=0.1, shuffle=True, split_size=split_size,
                               batch_size=1000, max_iter_gen=num_epochs, max_iter_dis=num_epochs, num_epochs=num_epochs,
                               lambda_gen=1e-5, lambda_dis=1e-5, lr=1e-3, display_interval=2,
                               num_jobs=num_jobs)
        training_idx, test_idx = st.fit(X=X, y=y, split_type=split_type)

        data_properties(y=y, selected_examples=[training_idx, test_idx], num_tails=5, dataset_name=dsname,
                        model_name=model_name, rspath=RESULT_PATH, display_dataframe=False)
        print("\n{0}\n".format(60 * "-"))
