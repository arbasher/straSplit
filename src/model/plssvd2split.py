'''
Clustering based stratified multi-label data splitting
'''

import os
import pickle as pkl
import sys
import textwrap
import time
import warnings

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.sparse import lil_matrix
from sklearn.cross_decomposition import PLSSVD

from src.model.extreme2split import ExtremeStratification
from src.model.naive2split import NaiveStratification
from src.utility.file_path import DATASET_PATH
from src.utility.utils import check_type, LabelBinarizer

np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class ClusterStratification(object):
    def __init__(self, num_clusters: int = 20, swap_probability: float = 0.1, threshold_proportion: float = 0.1,
                 decay: float = 0.1, shuffle: bool = True, split_size: float = 0.75, batch_size: int = 100,
                 num_epochs: int = 5, lr: float = 0.0001, num_jobs: int = 2):

        """Clustering based stratified based multi-label data splitting.

        Parameters
        ----------
        num_clusters : int, default=5
            The number of communities to form. It should be greater than 1.

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

        num_epochs : int, default=50
            The number of iterations of the k-means algorithm to run.

        lr : float, default=0.0001
            Learning rate.

        num_jobs : int, default=2
            The number of parallel jobs to run for splitting.
            ``-1`` means using all processors.
        """

        if num_clusters < 1:
            num_clusters = 20
        self.num_clusters = num_clusters

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
        desc = "## Configuration parameters to stratifying a multi-label " \
               "dataset splitting based on clustering the covariance of X " \
               "and y using PLSSVD:"

        argdict = dict()
        argdict.update({'num_clusters': 'Number of clusters to form: {0}'.format(self.num_clusters)})
        argdict.update({'swap_probability': 'A hyper-parameter: {0}'.format(self.swap_probability)})
        argdict.update({'threshold_proportion': 'A hyper-parameter: {0}'.format(self.threshold_proportion)})
        argdict.update({'decay': 'A hyper-parameter: {0}'.format(self.decay)})
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
        print(textwrap.TextWrapper(width=75, subsequent_indent='   ').fill(desc), file=sys.stderr)
        print('\t\t{0}'.format(args), file=sys.stderr)

    def __optimal_learning_rate(self, alpha):
        """Learning rate optimizer.

        Parameters
        ----------
        alpha : the learning rate.

        Returns
        -------
        learning rate : optimized rate
        """

        def _loss(p, y):
            z = p * y
            # approximately equal and saves the computation of the log
            if z > 18:
                return np.exp(-z)
            if z < -18:
                return -z
            return np.log(1.0 + np.exp(-z))

        typw = np.sqrt(1.0 / np.sqrt(alpha))
        # computing lr0, the initial learning rate
        initial_eta0 = typw / max(1.0, _loss(-typw, 1.0))
        # initialize t such that lr at first sample equals lr0
        optimal_init = 1.0 / (initial_eta0 * alpha)
        return optimal_init

    def fit(self, X, y, use_extreme: bool = False):
        """Split multi-label y dataset into train and test subsets.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features).

        y : {array-like, sparse matrix} of shape (n_samples, n_labels).

        use_extreme : whether to apply stratification for extreme
        multi-label datasets.

        Returns
        -------
        data partition : two lists of indices representing the resulted data split
        """

        if X is None:
            raise Exception("Please provide a dataset.")
        if y is None:
            raise Exception("Please provide labels for the dataset.")
        assert X.shape[0] == y.shape[0]

        check, X = check_type(X=X, return_list=False)
        if not check:
            tmp = "The method only supports scipy.sparse, numpy.ndarray, and list type of data"
            raise Exception(tmp)

        check, y = check_type(X=y, return_list=False)
        if not check:
            tmp = "The method only supports scipy.sparse, numpy.ndarray, and list type of data"
            raise Exception(tmp)

        num_examples, num_features, num_labels = X.shape[0], X.shape[1], y.shape[1]

        # check whether data is singly labeled
        if num_labels == 1:
            # transform it to multi-label data
            classes = list(set([i[0] if i else 0 for i in y.data]))
            mlb = LabelBinarizer(labels=classes)
            y = mlb.transform(y)

        # 1)- Compute covariance of X and y using SVD
        if not self.is_fit:
            model = PLSSVD(n_components=self.num_clusters, scale=True, copy=False)
            optimal_init = self.__optimal_learning_rate(alpha=self.lr)
            list_batches = np.arange(start=0, stop=num_examples, step=self.batch_size)
            total_progress = self.num_epochs * len(list_batches)
            for epoch in np.arange(start=1, stop=self.num_epochs + 1):
                for idx, batch_idx in enumerate(list_batches):
                    current_progress = epoch * (idx + 1)
                    desc = '\t>> Computing the covariance of X and y using PLSSVD: {0:.2f}%...'.format(
                        (current_progress / total_progress) * 100)
                    if total_progress == current_progress:
                        print(desc)
                    else:
                        print(desc, end="\r")
                    model.fit(X[batch_idx:batch_idx + self.batch_size].toarray(),
                              y[batch_idx:batch_idx + self.batch_size].toarray())
                    U = model.x_weights_
                    learning_rate = 1.0 / (self.lr * (optimal_init + epoch - 1))
                    U = U + learning_rate * (0.5 * 2 * U)
                    U = U + learning_rate * (0.5 * np.sign(U))
                    model.x_weights_ = U
            self.U = lil_matrix(model.x_weights_)
            del U, model

        # 2)- Project X onto a low dimension via U orthonormal basis obtained from SVD
        #     using SVD
        desc = '\t>> Projecting examples onto the obtained low dimensional U orthonormal basis...'
        print(desc)
        Z = X.dot(self.U)

        # 3)- Cluster low dimensional examples
        if not self.is_fit:
            desc = '\t>> Clustering the resulted low dimensional examples...'
            print(desc)
            self.centroid_kmeans, label_kmeans = kmeans2(data=Z.toarray(), k=self.num_clusters,
                                                         iter=self.num_epochs, minit='++')
        else:
            label_kmeans = np.array([np.argmin(z.dot(self.centroid_kmeans), 1)[0] for z in Z])

        mlb = LabelBinarizer(labels=list(range(self.num_clusters)))
        y = mlb.reassign_labels(y, mapping_labels=label_kmeans)
        self.is_fit = True

        # perform splitting
        if use_extreme:
            extreme = ExtremeStratification(swap_probability=self.swap_probability,
                                            threshold_proportion=self.threshold_proportion,
                                            decay=self.decay, shuffle=self.shuffle,
                                            split_size=self.split_size, num_epochs=self.num_epochs,
                                            verbose=False)
            train_list, test_list = extreme.fit(X=X, y=y)
        else:
            naive = NaiveStratification(shuffle=self.shuffle, split_size=self.split_size,
                                        batch_size=self.batch_size, num_jobs=self.num_jobs,
                                        verbose=False)
            train_list, test_list = naive.fit(y=y)
        return train_list, test_list


if __name__ == "__main__":
    X_name = "Xbirds_train.pkl"
    y_name = "Ybirds_train.pkl"
    use_extreme = False

    file_path = os.path.join(DATASET_PATH, y_name)
    with open(file_path, mode="rb") as f_in:
        y = pkl.load(f_in)
        idx = list(set(y.nonzero()[0]))
        y = y[idx]

    file_path = os.path.join(DATASET_PATH, X_name)
    with open(file_path, mode="rb") as f_in:
        X = pkl.load(f_in)
        X = X[idx]

    st = ClusterStratification(num_clusters=5, swap_probability=0.1, threshold_proportion=0.1,
                               decay=0.1, shuffle=True, split_size=0.75, batch_size=100,
                               num_epochs=5, lr=0.0001, num_jobs=2)
    training_idx, test_idx = st.fit(X=X, y=y, use_extreme=use_extreme)
    training_idx, dev_idx = st.fit(X=X[training_idx], y=y[training_idx],
                                   use_extreme=use_extreme)

    print("\n{0}".format(60 * "-"))
    print("## Summary...")
    print("\t>> Training set size: {0}".format(len(training_idx)))
    print("\t>> Validation set size: {0}".format(len(dev_idx)))
    print("\t>> Test set size: {0}".format(len(test_idx)))
