'''
The label enhancement algorithm based on label propagation
to recover the label distributions from logical labels by
using iterative label propagation technique. After discovering
distributions, the stratified multi-label dataset splitting is
performed.
'''

import os
import pickle as pkl
import sys
import textwrap
import time
import warnings

import igraph
import numpy as np
from scipy.sparse import triu, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity

from src.model.extreme2split import ExtremeStratification
from src.model.naive2split import NaiveStratification
from src.utility.file_path import DATASET_PATH
from src.utility.utils import check_type, custom_shuffle, normalize_laplacian, LabelBinarizer

np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class LabelEnhancementStratification(object):
    def __init__(self, num_subsamples: int = 10000, num_communities: int = 5, walk_size: int = 4,
                 sigma: float = 2, alpha: float = 0.2, swap_probability: float = 0.1,
                 threshold_proportion: float = 0.1, decay: float = 0.1, shuffle: bool = True,
                 split_size: float = 0.75, batch_size: int = 100, num_epochs: int = 50,
                 num_jobs: int = 2):
        """Label enhancement based stratified based multi-label data splitting.

        Parameters
        ----------
        num_subsamples : int, default=10000
            The number of subsamples to use for detecting communities.
            It should be greater than 100.

        num_communities : int, default=5
            The number of communities to form. It should be greater than 1.

        walk_size : int, default=4
            The length of random walks to perform, It should be greater than 2.

        sigma : float, default=2
            Scaling component to the graph degree matrix.
            It should be greater than 0.

        alpha : float, default=0.2
            A hyperparameter to balancing parameter which controls the fraction
            of the information inherited from the label propagation and the
            label matrix. It should be in (0,1).

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

        num_jobs : int, default=2
            The number of parallel jobs to run for splitting.
            ``-1`` means using all processors.
        """

        if num_subsamples < 100:
            num_subsamples = 100
        self.num_subsamples = num_subsamples

        if num_communities < 1:
            num_communities = 5
        self.num_communities = num_communities

        if walk_size < 2:
            walk_size = 4
        self.walk_size = walk_size

        if sigma < 0.0:
            sigma = 2
        self.sigma = sigma

        if alpha > 1.0 or alpha < 0.0:
            alpha = 0.2
        self.alpha = alpha

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
            num_epochs = 5
        self.num_epochs = num_epochs

        if num_jobs <= 0:
            num_jobs = 2
        self.num_jobs = num_jobs
        self.is_fit = False

        warnings.filterwarnings("ignore", category=Warning)

        self.__print_arguments()
        time.sleep(2)

    def __print_arguments(self, **kwargs):
        desc = "## Configuration parameters to stratifying a multi-label " \
               "dataset splitting based on label enhancement approach:"
        print(desc)

        argdict = dict()
        argdict.update({'num_subsamples': 'Subsampling input size: {0}'.format(self.num_subsamples)})
        argdict.update({'num_communities': 'Number of communities: {0}'.format(self.num_communities)})
        argdict.update({'walk_size': 'The length of random walks to perform: {0}'.format(self.walk_size)})
        argdict.update({'sigma': 'Constant that scales the amount of '
                                 'laplacian norm regularization: {0}'.format(self.sigma)})
        argdict.update({'alpha': 'A hyperparameter to balancing parameter'
                                 'which controls the fraction of the information '
                                 'inherited from the label propagation and the '
                                 'label matrix.: {0}'.format(self.alpha)})
        argdict.update({'swap_probability': 'A hyper-parameter: {0}'.format(self.swap_probability)})
        argdict.update({'threshold_proportion': 'A hyper-parameter: {0}'.format(self.threshold_proportion)})
        argdict.update({'decay': 'A hyper-parameter: {0}'.format(self.decay)})
        argdict.update({'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update({'split_size': 'Split size: {0}'.format(self.split_size)})
        argdict.update({'batch_size': 'Number of examples to use in '
                                      'each iteration: {0}'.format(self.batch_size)})
        argdict.update({'num_epochs': 'Number of loops over training set: {0}'.format(self.num_epochs)})
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

    def __graph_construction(self, X):
        """Clustering labels after constructing graph adjacency matrix empirically.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_labels)
            Matrix `X`.

        Returns
        -------
        community labels : a list of communities defining a community to a label association
        """

        A = X.T.dot(X)
        A = normalize_laplacian(A=A, sigma=self.sigma, return_adj=True, norm_adj=True)
        A = triu(A)
        # Create the graph
        vertices = [i for i in range(A.shape[0])]
        edges = list(zip(*A.nonzero()))
        weight = A.data.tolist()
        g = igraph.Graph(vertex_attrs={"label": vertices}, edges=edges)
        g = g.community_walktrap(weights=weight, steps=self.walk_size)
        communities = g.as_clustering(n=self.num_communities)
        communities = np.array(communities.membership)
        return communities

    def fit(self, X, y, use_extreme: bool = False):
        """Split multi-label y dataset into train and test subsets.

        Parameters
        ----------
        y : {array-like, sparse matrix} of shape (n_samples, n_labels).

        X : {array-like, sparse matrix} of shape (n_samples, n_features).

        use_extreme : whether to apply stratification for extreme
        multi-label datasets.

        Returns
        -------
        data partition : two lists of indices representing the resulted data split
        """

        check, X = check_type(X=X, return_list=False)
        if not check:
            tmp = "The method only supports scipy.sparse, numpy.ndarray, and list type of data"
            raise Exception(tmp)
        check, y = check_type(X=y, return_list=False)
        if not check:
            tmp = "The method only supports scipy.sparse, numpy.ndarray, and list type of data"
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
            if self.shuffle:
                sample_idx = custom_shuffle(num_examples)
                X = X[sample_idx, :]
                y = y[sample_idx, :]
            P = lil_matrix(cosine_similarity(X=X))
            P = normalize_laplacian(A=P, sigma=self.sigma, return_adj=True, norm_adj=True)
            P = triu(P)
            D = y
            for epoch in range(self.num_epochs):
                D = self.alpha * P * D + (1 - self.alpha) * y
            idx = np.random.choice(a=list(range(num_examples)), size=self.num_subsamples, replace=True)
            self.community_labels = self.__graph_construction(D[idx])
        mlb = LabelBinarizer(labels=list(range(self.num_communities)))
        y = mlb.reassign_labels(y, mapping_labels=self.community_labels)
        self.is_fit = True

        # perform splitting
        if use_extreme:
            extreme = ExtremeStratification(swap_probability=self.swap_probability,
                                            threshold_proportion=self.threshold_proportion, decay=self.decay,
                                            shuffle=self.shuffle, split_size=self.split_size,
                                            num_epochs=self.num_epochs, verbose=False)
            train_list, test_list = extreme.fit(X=X, y=y)
        else:
            naive = NaiveStratification(shuffle=self.shuffle, split_size=self.split_size,
                                        batch_size=self.batch_size,
                                        num_jobs=self.num_jobs, verbose=False)
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

    st = LabelEnhancementStratification(num_subsamples=10000, num_communities=5, walk_size=5, sigma=2, alpha=0.3,
                                        shuffle=True, split_size=0.8, batch_size=100, num_jobs=10)
    training_idx, test_idx = st.fit(X=X, y=y, use_extreme=use_extreme)
    training_idx, dev_idx = st.fit(X=X[training_idx], y=y[training_idx], use_extreme=use_extreme)

    print("\n{0}".format(60 * "-"))
    print("## Summary...")
    print("\t>> Training set size: {0}".format(len(training_idx)))
    print("\t>> Validation set size: {0}".format(len(dev_idx)))
    print("\t>> Test set size: {0}".format(len(test_idx)))
