import os
import pickle as pkl
import random
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import linalg, sparse
from scipy.cluster.vq import kmeans2
from scipy.sparse import lil_matrix

from src.utility.file_path import DATASET_PATH
from src.utility.utils import check_type, LabelBinarizer

np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class LabelEnhancementStratification(object):
    def __init__(self, num_subsamples: int = 10000, num_clusters: int = 5, sigma: float = 2, shuffle: bool = True,
                 split_size: float = 0.75, batch_size: int = 100, num_epochs: int = 50, num_jobs: int = 2):
        """Community based stratified based multi-label data splitting.

        Parameters
        ----------
        num_subsamples : int, default=10000
            The number of subsamples to use for detecting communities.
            It should be greater than 100.

        num_clusters : int, default=5
            The number of communities to form. It should be greater than 1.

        sigma : float, default=2
            Scaling component to the graph degree matrix.
            It should be greater than 0.

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
        desc = "## Split multi-label data using community based approach..."
        print(desc)

        argdict = dict()
        argdict.update({'num_subsamples': 'Subsampling input size: {0}'.format(self.num_subsamples)})
        argdict.update({'num_clusters': 'Number of communities: {0}'.format(self.num_clusters)})
        argdict.update({'sigma': 'Constant that scales the amount of '
                                 'laplacian norm regularization: {0}'.format(self.sigma)})
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

    def __graph_construction(self, X):
        """Clustering labels after constructing graph adjacency matrix empirically.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_labels)
            Matrix `X`.

        Returns
        -------
        clusters labels : a list of clusters defining a cluster to a label association
        """

        A = X.T.dot(X)
        A = self.__normalize_laplacian(A=A, return_adj=False, norm_adj=False)
        _, V = linalg.eigh(A.toarray())
        V = V[:, -self.num_clusters:]
        centroid, label = kmeans2(data=V, k=self.num_clusters, iter=self.num_epochs, minit='++')
        return centroid, label

    def __batch_fit(self, examples, check_list):
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

    def fit(self, X):
        """Split multi-label y dataset into train and test subsets.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_labels)
            Matrix `X`.

        Returns
        -------
        data partition : two lists of the resulted data split
        """

        check, X = check_type(X, False)
        if not check:
            tmp = "The method only supports scipy.sparse and numpy.ndarray type of data"
            raise Exception(tmp)

        num_examples, num_labels = X.shape

        # check whether data is singly labeled
        if num_labels == 1:
            # transform it to multi-label data
            classes = list(set([i[0] if i else 0 for i in X.data]))
            mlb = LabelBinarizer(labels=classes)
            X = mlb.transform(X)

        if not self.is_fit:
            desc = '\t>> Building Graph...'
            print(desc)
            # Construct graph
            idx = np.random.choice(a=list(range(num_examples)), size=self.num_subsamples, replace=True)
            clusters_centroids, self.clusters_labels = self.__graph_construction(X[idx])
            del clusters_centroids
        mlb = LabelBinarizer(labels=list(range(self.num_clusters)))
        X = mlb.reassign_labels(X, mapping_labels=self.clusters_labels)
        num_labels = self.num_clusters

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
            list_batches = np.arange(start=0, stop=len(examples), step=self.batch_size)
            results = parallel(delayed(self.__batch_fit)(examples[batch_idx:batch_idx + self.batch_size],
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


if __name__ == "__main__":
    X_name = "Ybirds_train.pkl"
    file_path = os.path.join(DATASET_PATH, X_name)
    with open(file_path, mode="rb") as f_in:
        X = pkl.load(f_in)
        X = lil_matrix(X[X.getnnz(axis=1) != 0][:, X.getnnz(axis=0) != 0].A)

    st = LabelEnhancementStratification(num_subsamples=10000, num_clusters=5, sigma=2,
                                        shuffle=True, split_size=0.8, batch_size=100,
                                        num_jobs=10)
    training_set, test_set = st.fit(X=X)
    training_set, dev_set = st.fit(X=X[training_set])

    print("training set size: {0}".format(len(training_set)))
    print("validation set size: {0}".format(len(dev_set)))
    print("test set size: {0}".format(len(test_set)))
