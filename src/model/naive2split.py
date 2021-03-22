'''
Naive online based stratified multi-label data splitting
'''

import os
import pickle as pkl
import random
import sys
import textwrap
import time
import warnings

import numpy as np
from joblib import Parallel, delayed

from utils import DATASET_PATH, RESULT_PATH, DATASET
from utils import check_type, data_properties, LabelBinarizer

np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class NaiveStratification(object):
    def __init__(self, shuffle: bool = True, split_size: float = 0.75, batch_size: int = 100,
                 num_jobs: int = 2, verbose: bool = True):
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

        verbose : bool, default=True
            Display arguments.
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

        if verbose:
            self.__print_arguments()
            time.sleep(2)

    def __print_arguments(self, **kwargs):
        desc = "## Configuration parameters to naive based stratified multi-label " \
               "dataset splitting:"
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
        print(textwrap.TextWrapper(width=75, subsequent_indent='   ').fill(desc), file=sys.stderr)
        print('\t\t{0}'.format(args), file=sys.stderr)

    def __split(self, examples, check_list):
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

    def fit(self, y):
        """Split multi-label y dataset into train and test subsets.

        Parameters
        ----------
        y : {array-like, sparse matrix} of shape (n_samples, n_labels).

        Returns
        -------
        data partition : two lists of indices representing the resulted data split
        """

        if y is None:
            raise Exception("Please provide labels for the dataset.")

        check, y = check_type(X=y, return_list=False)
        if not check:
            tmp = "The method only supports scipy.sparse, numpy.ndarray, and list type of data"
            raise Exception(tmp)

        num_examples, num_labels = y.shape

        # check whether data is singly labeled
        if num_labels == 1:
            # transform it to multi-label data
            classes = list(set([i[0] if i else 0 for i in y.data]))
            num_labels = len(classes)
            mlb = LabelBinarizer(labels=classes)
            y = mlb.transform(y)

        desc = '\t>> Perform splitting...'
        print(desc)
        check_list = list()
        train_list = list()
        test_list = list()
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads", verbose=0)
        # Find the label with the fewest (but at least one) remaining examples
        for label_idx in range(num_labels):
            examples = list(y[:, label_idx].nonzero()[0])
            if len(examples) == 0:
                continue
            list_batches = np.arange(start=0, stop=len(examples), step=self.batch_size)
            results = parallel(delayed(self.__split)(examples[batch_idx:batch_idx + self.batch_size],
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
        return sorted(train_list), sorted(test_list)


if __name__ == "__main__":
    model_name = "naive2split"
    split_size = 0.80
    num_jobs = 10

    for dsname in DATASET:
        y_name = dsname + "_y.pkl"
        file_path = os.path.join(DATASET_PATH, y_name)
        with open(file_path, mode="rb") as f_in:
            y = pkl.load(f_in)
            idx = list(set(y.nonzero()[0]))
            y = y[idx]

        st = NaiveStratification(shuffle=True, split_size=split_size, batch_size=500, num_jobs=num_jobs)
        training_idx, test_idx = st.fit(y=y)

        data_properties(y=y.toarray(), selected_examples=training_idx, num_tails=1, display_full_properties=True,
                        dataset_name=dsname, model_name=model_name, split_set_name="training",
                        rspath=RESULT_PATH)
        data_properties(y=y.toarray(), selected_examples=test_idx, num_tails=1, display_full_properties=False,
                        dataset_name=dsname, model_name=model_name, split_set_name="test", rspath=RESULT_PATH,
                        mode="a")
        print("\n{0}\n".format(60 * "-"))
