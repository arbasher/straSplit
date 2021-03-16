'''
Naive online based stratified multi-label data splitting
'''

import os
import pickle as pkl
import random
import sys
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import lil_matrix

from src.utility.file_path import DATASET_PATH
from src.utility.utils import check_type, LabelBinarizer

np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class NaiveStratification(object):
    def __init__(self, shuffle: bool = True, split_size: float = 0.75, batch_size: int = 100, num_jobs: int = 2):
        """Naive online stratified based multi-label data splitting.

        Parameters
        ----------
        shuffle : bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None.

        split_size : float, default=0.75
            It should be between 0.0 and 1.0 and represents the proportion
            of data to include in the test split.

        batch_size : int, default=100
            It should be a positive integer and represents the size of
            batch during splitting process.

        num_jobs : int, default=2
            The number of parallel jobs to run for splitting.
            ``-1`` means using all processors.
        """

        if split_size >= 1.0 or split_size <= 0.0:
            split_size = 0.8
        self.split_size = split_size

        if batch_size <= 0:
            batch_size = 100
        self.batch_size = batch_size

        if batch_size <= 0:
            batch_size = 100
        self.batch_size = batch_size

        self.shuffle = shuffle

        if num_jobs <= 0:
            num_jobs = 2
        self.num_jobs = num_jobs

        warnings.filterwarnings("ignore", category=Warning)

        self.__print_arguments()
        time.sleep(2)

    def __print_arguments(self, **kwargs):
        desc = "## Split multi-label data using naive stratified approach..."
        print(desc)

        argdict = dict()
        argdict.update({'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update({'split_size': 'Split size: {0}'.format(self.split_size)})
        argdict.update({'batch_size': 'Number of examples to use in '
                                      'each iteration: {0}'.format(self.batch_size)})
        argdict.update({'num_jobs': 'Number of parallel workers: {0}'.format(self.num_jobs)})

        for key, value in kwargs.items():
            argdict.update({key: value})
        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print('\t>> The following arguments are applied:\n\t\t{0}'.format(args), file=sys.stderr)

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

        check, X = check_type(X)
        if not check:
            tmp = "The method only supports scipy.sparse and numpy.ndarray type of data"
            raise Exception(tmp)

        num_examples, num_labels = X.shape

        # check whether data is singly labeled
        if num_labels == 1:
            # transform it to multi-label data
            classes = list(set([i[0] if i else 0 for i in X.data]))
            num_labels = len(classes)
            mlb = LabelBinarizer(labels=classes)
            X = mlb.transform(X)

        check_list = list()
        train_list = list()
        test_list = list()
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads", verbose=0)

        desc = '>> Stratified Split...'
        print(desc)

        # Find the label with the fewest (but at least one) remaining
        # examples, breaking ties randomly
        for label_idx in range(num_labels):
            examples = list(X[:, label_idx].nonzero()[0])
            if len(examples) == 0:
                continue
            list_batches = np.arange(start=0, stop=len(examples), step=self.batch_size)
            results = parallel(delayed(self.__parallel_split)(examples[batch_idx:batch_idx + self.batch_size],
                                                              check_list)
                               for idx, batch_idx in enumerate(list_batches))
            desc = '\t--> Splitting progress: {0:.2f}%...'.format(((label_idx + 1) / num_labels) * 100)
            if label_idx + 1 == num_labels:
                print(desc)
            else:
                print(desc, end="\r")

            results = list(zip(*results))
            train_list.extend([i for item in results[0] for i in item])
            test_list.extend([i for item in results[1] for i in item])
            del results
        del check_list
        return sorted(train_list), sorted(test_list)


if __name__ == "__main__":
    X_name = "Ybirds_train.pkl"
    file_path = os.path.join(DATASET_PATH, X_name)
    with open(file_path, mode="rb") as f_in:
        X = pkl.load(f_in)
        X = lil_matrix(X[X.getnnz(axis=1) != 0][:, X.getnnz(axis=0) != 0].A)

    st = NaiveStratification(shuffle=True, split_size=0.8, batch_size=1000,
                             num_jobs=10)
    training_set, test_set = st.fit(X=X)
    training_set, dev_set = st.fit(X=X[training_set])

    print("training set size: {0}".format(len(training_set)))
    print("validation set size: {0}".format(len(dev_set)))
    print("test set size: {0}".format(len(test_set)))
