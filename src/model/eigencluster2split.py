'''
Clustering eigenvalues of the label adjacency matrix
to perform stratified split of a multi-label dataset
'''

import os
import pickle as pkl
import sys
import textwrap
import time
import warnings

import numpy as np
from scipy import linalg
from scipy.cluster.vq import kmeans2

from src.model.utils import normalize_laplacian
from src.model.extreme2split import ExtremeStratification
from src.model.iterative2split import IterativeStratification
from src.model.naive2split import NaiveStratification
from src.model.utils import DATASET_PATH, RESULT_PATH, DATASET
from src.model.utils import check_type, data_properties, LabelBinarizer

np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class ClusteringEigenStratification(object):
    def __init__(self, num_subsamples: int = 10000, num_clusters: int = 5, sigma: float = 2,
                 swap_probability: float = 0.1, threshold_proportion: float = 0.1, decay: float = 0.1,
                 shuffle: bool = True, split_size: float = 0.75, batch_size: int = 100,
                 num_epochs: int = 50, num_jobs: int = 2):
        """Community based stratified based multi-label data splitting.

        Parameters
        ----------
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

        num_epochs : int, default=50
            The number of iterations of the k-means algorithm to run.

        num_jobs : int, default=2
            The number of parallel jobs to run for splitting.
            ``-1`` means using all processors.
        """

        if num_subsamples < 100:
            num_subsamples = 100
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
               "dataset splitting based on clustering eigen values of the " \
               "label adjacency matrix:"

        argdict = dict()
        argdict.update({'num_subsamples': 'Subsampling input size: {0}'.format(self.num_subsamples)})
        argdict.update({'num_clusters': 'Number of communities: {0}'.format(self.num_clusters)})
        argdict.update({'sigma': 'Constant that scales the amount of '
                                 'laplacian norm regularization: {0}'.format(self.sigma)})
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

    def __cluster_generation(self, y):
        """Clustering labels after constructing graph adjacency matrix empirically.

        Parameters
        ----------
        y : {array-like, sparse matrix} of shape (n_samples, n_labels)
            Matrix `y`.

        Returns
        -------
        clusters labels : a list of clusters defining a cluster to a label association
        """

        A = y.T.dot(y)
        A = normalize_laplacian(A=A, sigma=self.sigma, return_adj=False, norm_adj=False)
        _, V = linalg.eigh(A.toarray())
        V = V[:, -self.num_clusters:]
        _, label = kmeans2(data=V, k=self.num_clusters, iter=self.num_epochs, minit='++')
        return label

    def fit(self, y, X=None, split_type: str = "extreme"):
        """Split multi-label y dataset into train and test subsets.

        Parameters
        ----------
        y : {array-like, sparse matrix} of shape (n_samples, n_labels).

        X : {array-like, sparse matrix} of shape (n_samples, n_features).

        split_type : Splitting type of {naive, extreme, iterative}.

        Returns
        -------
        data partition : two lists of indices representing the resulted data split
        """

        if y is None:
            raise Exception("Please provide labels for the dataset.")
        assert X.shape[0] == y.shape[0]
        check, y = check_type(X=y, return_list=False)
        if not check:
            tmp = "The method only supports scipy.sparse, numpy.ndarray, and list type of data"
            raise Exception(tmp)

        if split_type == "extreme":
            if X is None:
                raise Exception("Please provide a dataset.")
            check, X = check_type(X=X, return_list=False)
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
            desc = '\t>> Extracting clusters...'
            print(desc)
            # Construct graph
            idx = np.random.choice(a=list(range(num_examples)), size=self.num_subsamples, replace=True)
            self.clusters_labels = self.__cluster_generation(y[idx])
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
    model_name = "eigencluster2split"
    split_type = "extreme"
    split_size = 0.80
    num_epochs = 5
    num_jobs = 10

    for dsname in sorted(DATASET):
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

        st = ClusteringEigenStratification(num_subsamples=10000, num_clusters=5, sigma=2, swap_probability=0.1,
                                           threshold_proportion=0.1, decay=0.1, shuffle=True, split_size=split_size,
                                           batch_size=500, num_epochs=num_epochs, num_jobs=num_jobs)
        training_idx, test_idx = st.fit(y=y, X=X, split_type=split_type)

        data_properties(y=y.toarray(), selected_examples=training_idx, num_tails=1, display_full_properties=True,
                        dataset_name=dsname, model_name=model_name, split_set_name="training",
                        rspath=RESULT_PATH)
        data_properties(y=y.toarray(), selected_examples=test_idx, num_tails=1, display_full_properties=False,
                        dataset_name=dsname, model_name=model_name, split_set_name="test", rspath=RESULT_PATH,
                        mode="a")
        print("\n{0}\n".format(60 * "-"))
