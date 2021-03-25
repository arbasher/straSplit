'''
Stratified Sampling for extreme Multi-Label Data
adapted from https://github.com/maxitron93/stratified_sampling_for_XML
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


class ExtremeStratification(object):

    def __init__(self, swap_probability: float = 0.1, threshold_proportion: float = 0.1, decay: float = 0.1,
                 shuffle: bool = True, split_size: float = 0.75, num_epochs: int = 50, verbose: bool = True):
        """Splitting a large scale multi-label data using the
            stratification approach.

        Parameters
        ----------
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

        num_epochs : int, default=50
            The number of iterations of the k-means algorithm to run.

        verbose : bool, default=True
            Display arguments.
        """

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

        if num_epochs <= 0:
            num_epochs = 5
        self.num_epochs = num_epochs

        warnings.filterwarnings("ignore", category=Warning)

        if verbose:
            self.__print_arguments()
            time.sleep(2)

    def __print_arguments(self, **kwargs):
        desc = "## Configuration parameters to stratifying a large scale " \
               "multi-label dataset splitting:"
        argdict = dict()
        argdict.update(
            {'swap_probability': 'A hyper-parameter for extreme stratification: {0}'.format(self.swap_probability)})
        argdict.update({'threshold_proportion': 'A hyper-parameter for extreme stratification: {0}'.format(
            self.threshold_proportion)})
        argdict.update({'decay': 'A hyper-parameter for extreme stratification: {0}'.format(self.decay)})
        argdict.update({'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update({'split_size': 'Split size: {0}'.format(self.split_size)})
        argdict.update({'num_epochs': 'Number of loops over a dataset: {0}'.format(self.num_epochs)})

        for key, value in kwargs.items():
            argdict.update({key: value})
        args = list()
        for key, value in argdict.items():
            args.append(value)
        args = [str(item[0] + 1) + '. ' + item[1] for item in zip(list(range(len(args))), args)]
        args = '\n\t\t'.join(args)
        print(textwrap.TextWrapper(width=75, subsequent_indent='   ').fill(desc), file=sys.stderr)
        print('\t\t{0}'.format(args), file=sys.stderr)

    # 1. Create instances_dict to keep track of instance information:
    def __create_instances_dict(self, X, y):
        instances_dict = {}
        for idx in range(X.shape[0]):
            train_or_test = 'train'
            if random.uniform(0, 1) > self.split_size:
                train_or_test = 'test'
            instances_dict[idx] = {'labels': y[idx].nonzero()[1].tolist(),
                                   'train_or_test': train_or_test,
                                   'instance_score': 0}  # instance_score: float, adjusted sum of label scores
        return instances_dict

    # 2. Create labels_dict to keep track of label information:
    def __create_labels_dict(self, instances_dict):
        labels_dict = {}
        for _, instance_dict in instances_dict.items():
            train_or_test = instance_dict['train_or_test']
            for label in instance_dict['labels']:
                try:
                    if train_or_test == 'train':
                        labels_dict[label]['train'] += 1
                    else:
                        labels_dict[label]['test'] += 1
                except:
                    if train_or_test == 'train':
                        labels_dict[label] = {'train': 1, 'test': 0, 'label_score': 0}
                    else:
                        labels_dict[label] = {'train': 0, 'test': 1, 'label_score': 0}
        return labels_dict

    # 3. Calculate the label score for each label in labels_dict
    def __score_labels(self, labels_dict, average_labels_per_instance):
        target_size = 1 - self.split_size
        for label, label_dict in labels_dict.items():
            label_score = 0
            label_count = label_dict['train'] + label_dict['test']
            if label_count > 1:
                actual_test_proportion = label_dict['test'] / label_count
                if actual_test_proportion >= target_size:  # Too much of the label is in the test set
                    label_score = (actual_test_proportion - target_size) / self.split_size
                    if actual_test_proportion > 0.999:
                        label_score += average_labels_per_instance
                else:  # Too much of the label is in the train set
                    label_score = (actual_test_proportion - target_size) / target_size
                    if actual_test_proportion < 0.001:
                        label_score -= average_labels_per_instance
            labels_dict[label]['label_score'] = label_score

    # 4. Calculate the instance score for each instance in instances_dict
    def __score_instances(self, instances_dict, labels_dict, examples_scores=None):
        for instance_id, instance_dict in instances_dict.items():
            instance_score = 0
            train_or_test = instance_dict['train_or_test']
            for label in instance_dict['labels']:
                label_score = labels_dict[label]['label_score']
                if label_score > 0:  # If too much of the label is in the test set 
                    if train_or_test == 'test':
                        instance_score += label_score  # If instance in test, increase score
                    elif train_or_test == 'train':
                        instance_score -= label_score  # If instance in train, decrease score
                    else:
                        print('\t\t--> Something went wrong: {}'.format(instance_id))
                elif label_score < 0:  # If too much of the label is in the train set
                    if train_or_test == 'train':
                        instance_score -= label_score  # If instance in train, increase score
                    elif train_or_test == 'test':
                        instance_score += label_score  # If instance in test, decrease score
                    else:
                        print('\t\t--> Something went wrong: {}'.format(instance_id))
            temp = 0
            if examples_scores:
                temp = examples_scores[instance_id]
            instances_dict[instance_id]['instance_score'] = instance_score + temp

    # 5. Calculate the total score
    # The higher the score, the more 'imbalanced' the distribution of labels between train and test sets
    def __calculate_total_score(self, instances_dict):
        total_score = 0
        for _, instance_dict in instances_dict.items():
            total_score += instance_dict['instance_score']
        return total_score

    # 6. Calculate the threshold score for swapping
    def __calculte_threshold_score(self, instances_dict, average_labels_per_instance, epoch):
        instance_scores = [instance_dict['instance_score'] for _, instance_dict in instances_dict.items() if
                           instance_dict['instance_score'] < average_labels_per_instance]
        threshold_score = np.quantile(instance_scores, (1 - (self.threshold_proportion / ((1 + self.decay) ** epoch))))
        if threshold_score < 0:
            threshold_score = 0
        return threshold_score

    # 7. Swap the instances with instance_score that is greater than the threshold score
    # Probability of swapping an instance is swap_probability
    def __swap_instances(self, instances_dict, threshold_score, swap_counter, average_labels_per_instance, epoch):
        for instance_id, instance_dict in instances_dict.items():
            instance_score = instance_dict['instance_score']
            if instance_score >= average_labels_per_instance:
                if random.uniform(0, 1) <= 0.25 / (1.05 ** epoch):
                    current_group = instance_dict['train_or_test']
                    if current_group == 'train':
                        instances_dict[instance_id]['train_or_test'] = 'test'
                        swap_counter['to_test'] += 1
                    elif current_group == 'test':
                        instances_dict[instance_id]['train_or_test'] = 'train'
                        swap_counter['to_train'] += 1
            elif instance_score > threshold_score:
                if random.uniform(0, 1) <= self.swap_probability / ((1 + self.decay) ** epoch):
                    current_group = instance_dict['train_or_test']
                    if current_group == 'train':
                        instances_dict[instance_id]['train_or_test'] = 'test'
                        swap_counter['to_test'] += 1
                    elif current_group == 'test':
                        instances_dict[instance_id]['train_or_test'] = 'train'
                        swap_counter['to_train'] += 1

    def fit(self, X, y, examples_scores=None):
        """Split multi-label y dataset into train and test subsets.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features).

        y : {array-like, sparse matrix} of shape (n_samples, n_labels).

        examples_scores : a dictionary of shape (n_samples, 1) that contains
            uncertainty score to each example.

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

        num_examples, num_labels = X.shape

        # check whether data is singly labeled
        if num_labels == 1:
            # transform it to multi-label data
            classes = list(set([i[0] if i else 0 for i in X.data]))
            mlb = LabelBinarizer(labels=classes)
            y = mlb.transform(y)

        if self.shuffle:
            sample_idx = custom_shuffle(num_examples=num_examples)
            X = X[sample_idx, :]
            y = y[sample_idx, :]

        # Keep track how how many instances have been swapped to train or test
        swap_counter = {'to_train': 0, 'to_test': 0}

        # 1. Create instances_dict to keep track of instance information:
        instances_dict = self.__create_instances_dict(X, y)

        # 2 Get average number of labels per instance
        labels_per_instance = [len(instance_dict['labels']) for idx, instance_dict in instances_dict.items()]
        average_labels_per_instance = sum(labels_per_instance) / len(labels_per_instance)

        # 3. Create labels_dict to keep track of label information:
        labels_dict = self.__create_labels_dict(instances_dict)

        # 4. Calculate the label score for each label in labels_dict
        # Positive score if too much of the label is in the test set
        # Negative score if too much of the label is in the train set
        self.__score_labels(labels_dict, average_labels_per_instance)

        # 5. Calculate the instance score for each instance in instances_dict
        # A high score means the instance is a good candidate for swapping
        self.__score_instances(instances_dict, labels_dict, examples_scores=examples_scores)

        # 6. Calculate the total score
        # The higher the score, the more 'imbalanced' the distribution of labels between train and test sets
        total_score = self.__calculate_total_score(instances_dict)
        desc = '\t>> Perform splitting (extreme)...'
        print(desc)
        print('\t\t--> Starting score: {0}'.format(round(total_score)))

        # Main loop to create stratified train-test split  
        for epoch in range(self.num_epochs):
            # To keep track of how long each iteration takes

            # 1. Calculate the threshold score for swapping
            threshold_score = self.__calculte_threshold_score(instances_dict=instances_dict,
                                                              average_labels_per_instance=average_labels_per_instance,
                                                              epoch=epoch)

            # 2. Swap the instances with instance_score that is greater than the threshold score
            # Probability of swapping an instance is swap_probability
            self.__swap_instances(instances_dict=instances_dict, threshold_score=threshold_score,
                                  swap_counter=swap_counter, average_labels_per_instance=average_labels_per_instance,
                                  epoch=epoch)

            # 3. Recreate labels_dict with updated train-test split
            labels_dict = self.__create_labels_dict(instances_dict=instances_dict)

            # 4. Recalculate the label score for each label in labels_dict
            self.__score_labels(labels_dict=labels_dict, average_labels_per_instance=average_labels_per_instance)

            # 5. Recalculate the instance score for each instance in instances_dict
            self.__score_instances(instances_dict=instances_dict, labels_dict=labels_dict,
                                   examples_scores=examples_scores)

            # 6. Recalculate the total score
            total_score = self.__calculate_total_score(instances_dict=instances_dict)
            desc = '\t\t--> Splitting progress: {0:.2f}%; score: {1:.2f}'.format(((epoch + 1) / self.num_epochs * 100),
                                                                                 total_score)
            if epoch + 1 == self.num_epochs:
                print(desc)
            else:
                print(desc, end="\r")

        # Prepare train_list, test_list
        train_list = []
        test_list = []
        for idx, instance_dict in instances_dict.items():
            if instance_dict['train_or_test'] == 'train':
                train_list.append(idx)
            elif instance_dict['train_or_test'] == 'test':
                test_list.append(idx)
            else:
                print(f'Something went wrong: {idx}')
        return sorted(train_list), sorted(test_list)


if __name__ == "__main__":
    from utils import DATASET_PATH, RESULT_PATH, DATASET

    model_name = "extreme2split"
    split_size = 0.80
    num_epochs = 5

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

        st = ExtremeStratification(swap_probability=0.1, threshold_proportion=0.1, decay=0.1,
                                   shuffle=True, split_size=split_size, num_epochs=num_epochs)
        training_idx, test_idx = st.fit(X=X, y=y)

        data_properties(y=y, selected_examples=[training_idx, test_idx], num_tails=5, dataset_name=dsname,
                        model_name=model_name, rspath=RESULT_PATH, display_dataframe=False)
        print("\n{0}\n".format(60 * "-"))
