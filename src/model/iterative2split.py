'''
Iterative stratification for multi-mabel data.

1. Sechidis, K., Tsoumakas, G. and Vlahavas, I., 2011,
September. On the stratification of multi-label data.
In Joint European Conference on Machine Learning and
Knowledge Discovery in Databases (pp. 145-158).
Springer, Berlin, Heidelberg.
'''

import os
import pickle as pkl
import random
import sys
import textwrap
import time
import warnings

import numpy as np

from utils import check_type, custom_shuffle, data_properties, LabelBinarizer

random.seed(12345)


class IterativeStratification(object):

    def __init__(self, shuffle: bool = True, split_size: float = 0.75, verbose: bool = True):
        """Splitting a large scale multi-label data using the iterative stratification approach.

        Parameters
        ----------
        shuffle : bool, default=True
            Whether or not to shuffle the data before splitting. If shuffle=False
            then stratify must be None.

        split_size : float, default=0.75
            It should be between 0.0 and 1.0 and represents the proportion
            of data to include in the test split.

        verbose : bool, default=True
            Display arguments.
        """

        self.shuffle = shuffle

        if split_size >= 1.0 or split_size <= 0.0:
            split_size = 0.8
        self.split_size = split_size

        warnings.filterwarnings("ignore", category=Warning)

        if verbose:
            self.__print_arguments()
            time.sleep(2)

    def __print_arguments(self, **kwargs):
        desc = "## Configuration parameters to iteratively stratifying " \
               "a multi-label dataset splitting:"
        argdict = dict()
        argdict.update({'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update({'split_size': 'Split size: {0}'.format(self.split_size)})

        for key, value in kwargs.items():
            argdict.update({key: value})
        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print(textwrap.TextWrapper(width=75, subsequent_indent='   ').fill(desc), file=sys.stderr)
        print('\t\t{0}'.format(args), file=sys.stderr)

    def __create_desired_examples(self, num_examples):
        subset_dict = {}
        r = [self.split_size, 1 - self.split_size]
        for idx in range(2):
            subset_dict[idx] = {'desired': int(round(num_examples * r[idx]))}
        return subset_dict

    def __create_desired_labels(self, y, subset_dict):
        r = [self.split_size, 1 - self.split_size]
        for label_idx in range(y.shape[1]):
            total_examples = y[:, label_idx].sum()
            for idx, item in subset_dict.items():
                item.update({label_idx: int(round(total_examples * r[idx]))})
        return subset_dict

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
            mlb = LabelBinarizer(labels=classes)
            y = mlb.transform(y)

        if self.shuffle:
            sample_idx = custom_shuffle(num_examples=num_examples)
            y = y[sample_idx, :]

        # 1. Calculate the desired number of examples at each subset.
        subset_dict = self.__create_desired_examples(num_examples=num_examples)

        # 2. Calculate the desired number of examples of each label at each subset.
        subset_dict = self.__create_desired_labels(y=y, subset_dict=subset_dict)

        examples_dict = dict([(i, []) for i in range(2)])
        selected_examples = []
        # Main loop to create stratified k-fold splits
        total_examples = list(range(num_examples))
        sorted_list = list(zip(list(range(num_labels)), np.sum(y.toarray(), 0)))
        sorted_list = sorted(sorted_list, key=lambda x: x[1])
        desc = '\t>> Perform splitting (iterative)...'
        print(desc)

        while len(total_examples) > 0:
            # 1. Find the label with the fewest (but at least one) remaining examples,
            # breaking ties randomly
            examples = y[:, sorted_list[0][0]].nonzero()[0]
            for example in examples:
                if example in selected_examples:
                    continue
                # 2. Find the subset(s) with the largest number of desired examples for this
                # label
                m = np.argmax([item[sorted_list[0][0]] for k, item in subset_dict.items()])
                examples_dict.update({m: examples_dict[m] + [example]})
                selected_examples.append(example)
                # 3. Update the desired number of examples
                labels = y[example].nonzero()[1]
                for label_idx in labels:
                    subset_dict[m][label_idx] -= 1
                subset_dict[m]['desired'] -= 1
                temp = total_examples.index(example)
                total_examples.pop(temp)
            sorted_list.pop(0)
            desc = '\t\t--> Splitting progress: {0:.2f}%...'.format(((1 - (len(total_examples) / num_examples)) * 100))
            if len(examples) == num_examples:
                print(desc)
            else:
                print(desc, end="\r")

        # Prepare train_list, test_list
        train_list = examples_dict[0]
        test_list = examples_dict[1]
        return sorted(train_list), sorted(test_list)


if __name__ == "__main__":
    from utils import DATASET_PATH, RESULT_PATH, DATASET

    model_name = "iterative2split"
    split_size = 0.80

    for dsname in DATASET:
        y_name = dsname + "_y.pkl"
        file_path = os.path.join(DATASET_PATH, y_name)
        with open(file_path, mode="rb") as f_in:
            y = pkl.load(f_in)
            idx = list(set(y.nonzero()[0]))
            y = y[idx]

        st = IterativeStratification(shuffle=True, split_size=split_size)
        training_idx, test_idx = st.fit(y=y)

        data_properties(y=y, selected_examples=[training_idx, test_idx], num_tails=5, dataset_name=dsname,
                        model_name=model_name, rspath=RESULT_PATH, display_dataframe=False)
        print("\n{0}\n".format(60 * "-"))
