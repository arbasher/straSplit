'''
Clustering based stratified multi-label data splitting
'''

import os
import pickle as pkl
import random
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.cluster.vq import kmeans2
from scipy.sparse import lil_matrix
from sklearn.cross_decomposition import PLSSVD

from src.split_data.utility import DIRECTORY_PATH, check_type, LabelBinarizer

np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class StratifiedClustering(object):
    def __init__(self, num_clusters: int = 20, shuffle: bool = True, split_size: float = 0.75, batch_size: int = 100,
                 num_epochs: int = 5, lr: float = 0.0001, num_jobs: int = 2):

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
        """

        if num_clusters < 1:
            num_clusters = 20
        self.num_clusters = num_clusters

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

        # 1)- Compute covariance of X and y using SVD
        if not self.is_fit:
            plsca = PLSSVD(n_components=self.num_clusters, scale=True, copy=False)
            lam_reg = 0.5
            mu_reg = 0.5
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
                    plsca.fit(X[batch_idx:batch_idx + self.batch_size].toarray(),
                              y[batch_idx:batch_idx + self.batch_size].toarray())
                    U = plsca.x_weights_
                    learning_rate = 1.0 / (self.lr * (optimal_init + epoch - 1))
                    U = U + learning_rate * (lam_reg * 2 * U)
                    U = U + learning_rate * (mu_reg * np.sign(U))
                    plsca.x_weights_ = U
            self.U = lil_matrix(plsca.x_weights_)
            del U, plsca

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
            examples = list(y[:, label_idx].nonzero()[0])
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
    log_path = os.path.join(DIRECTORY_PATH, 'log')

    X_name = "Xbirds_train.pkl"
    y_name = "ybirds_train.pkl"

    file_path = os.path.join(DIRECTORY_PATH, 'data', X_name)
    with open(file_path, mode="rb") as f_in:
        X = pkl.load(f_in)

    file_path = os.path.join(DIRECTORY_PATH, 'data', y_name)
    with open(file_path, mode="rb") as f_in:
        y = pkl.load(f_in)

    st = StratifiedClustering(num_clusters=5, shuffle=True, split_size=0.8, batch_size=100, num_epochs=5, lr=0.0001,
                              num_jobs=2)
    training_set, test_set = st.fit(X=X, y=y)
    training_set, dev_set = st.fit(X=X[training_set], y=y[training_set])

    print("training set size: {0}".format(len(training_set)))
    print("validation set size: {0}".format(len(dev_set)))
    print("test set size: {0}".format(len(test_set)))
