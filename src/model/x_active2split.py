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
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import lil_matrix
from scipy.sparse import triu
from sklearn.metrics.pairwise import cosine_similarity

from src.utility.file_path import DATASET_PATH
from src.utility.utils import check_type, LabelBinarizer
from src.utility.utils import custom_shuffle, normalize_laplacian

EPSILON = np.finfo(np.float).eps
np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class ActiveStratification(object):

    def __init__(self, subsample_input_size=0.3, subsample_labels_size=50, calc_ads=True,
                 acquisition_type="variation", top_k=20, ads_percent=0.7, tol_labels_iter=10,
                 delay_factor=1.0, forgetting_rate=0.9, num_subsamples: int = 10000,
                 num_communities: int = 5, walk_size: int = 4, sigma: float = 2,
                 alpha: float = 0.2, swap_probability: float = 0.1,
                 threshold_proportion: float = 0.1, decay: float = 0.1, shuffle: bool = True,
                 split_size: float = 0.75, batch_size: int = 100, max_inner_iter=100,
                 num_epochs: int = 50, num_jobs: int = 2):
        """Community based stratified based multi-label data splitting.

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

        self.subsample_input_size = subsample_input_size
        self.subsample_labels_size = subsample_labels_size
        self.calc_ads = calc_ads
        self.ads_percent = ads_percent
        self.acquisition_type = acquisition_type  # entropy, mutual, variation, psp
        self.top_k = top_k
        self.tol_labels_iter = tol_labels_iter
        self.forgetting_rate = forgetting_rate
        self.delay_factor = delay_factor
        self.max_inner_iter = max_inner_iter

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
        desc = "## Split extreme multi-label data using stratified partitioning..."
        print(desc)

        argdict = dict()
        argdict.update({'swap_probability': 'A hyper-parameter: {0}'.format(self.top_k)})
        argdict.update({'threshold_proportion': 'A hyper-parameter: {0}'.format(self.tol_labels_iter)})
        argdict.update({'decay': 'A hyper-parameter: {0}'.format(self.delay_factor)})
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
        print('\t>> The following arguments are applied:\n\t\t{0}'.format(args), file=sys.stderr)

    def __entropy(self, score):
        log_prob_bag = np.log(score + EPSILON)
        if len(score.shape) > 1:
            entropy_ = -np.diag(np.dot(score, log_prob_bag.T))
        else:
            entropy_ = -np.multiply(score, log_prob_bag)
        np.nan_to_num(entropy_, copy=False)
        entropy_ = entropy_ + EPSILON
        return entropy_

    def __mutual_information(self, H_m, H):
        mean_entropy = np.mean(H_m, axis=0)
        mutual_info = H - mean_entropy
        return mutual_info

    def __variation_ratios(self, score, sample_idx):
        mlb = preprocessing.MultiLabelBinarizer()
        V = score[:, sample_idx, :]
        V = mlb.fit_transform(V)
        V = mlb.classes_[np.argsort(-np.sum(V, axis=0))][:self.top_k]
        total_sum = np.intersect1d(score[0, sample_idx], V).shape[0]
        D = 1 - total_sum / self.top_k
        return D

    def __psp(self, score, samples_idx, y_true):
        num_labels = y_true.shape[1]

        # propensity of all labels
        N_j = y_true[samples_idx].toarray()
        labels_sum = np.sum(N_j, axis=0)
        g = 1 / (labels_sum + 1)
        psp_label = 1 / (1 + g)

        # retrieve the top k labels
        top_k = y_true.shape[1] if self.top_k > num_labels else self.top_k
        labels_idx = np.argsort(-score)[:, :top_k]

        # compute normalized psp@k
        psp = N_j / psp_label
        tmp = [psp[s_idx, labels_idx[s_idx]] for s_idx in np.arange(psp.shape[0])]
        psp = (1 / top_k) * np.sum(tmp, axis=1)
        min_psp = np.min(psp) + EPSILON
        max_psp = np.max(psp) + EPSILON
        psp = psp - min_psp
        psp = psp / (max_psp - min_psp)
        psp = 1 - psp + EPSILON
        return psp

    def fit(self, X, y, use_extreme, score, samples_idx):
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

        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads", verbose=0)
        if self.acquisition_type == "entropy":
            models_entropy = np.array([np.mean(score[samples_idx == idx], axis=(0)) for idx in np.unique(samples_idx)])
            H = self.__entropy(score=models_entropy)
        elif self.acquisition_type == "mutual":
            models_entropy = np.array([np.mean(score[samples_idx == idx], axis=(0)) for idx in np.unique(samples_idx)])
            H = self.__entropy(score=models_entropy)
            results = parallel(delayed(self.__entropy)(score[model_idx], model_idx)
                               for model_idx in np.arange(self.num_models))
            H_m = np.vstack(zip(*results))
            H_m = np.array([[H_m[np.argwhere(samples_idx[m] == s_idx)[0][0], m]
                             if s_idx in samples_idx[m] else EPSILON for idx, s_idx in
                             enumerate(np.unique(samples_idx))]
                            for m in np.arange(self.num_models)]).T
            H = self.__mutual_information(H_m=H_m.T, H=H, model_idx=self.num_models - 1)
        elif self.acquisition_type == "variation":
            labels = self.num_labels
            score = np.array([[[score[m][np.argwhere(samples_idx[m] == s_idx)[0][0], l_idx]
                                if s_idx in samples_idx[m] else EPSILON
                                for l_idx in np.arange(labels)]
                               for idx, s_idx in enumerate(np.unique(samples_idx))]
                              for m in np.arange(self.num_models)])
            num_samples = score.shape[1]
            score = np.argsort(-score)[:, :, :self.top_k]
            results = parallel(delayed(self.__variation_ratios)(score, sample_idx, num_samples)
                               for sample_idx in np.arange(num_samples))
            H = np.vstack(results).reshape(num_samples, )
        else:
            results = parallel(delayed(self.__psp)(score[model_idx], model_idx,
                                                   samples_idx[model_idx], y_true)
                               for model_idx in np.arange(self.num_models))
            H = np.hstack(zip(*results))
            samples_idx = np.hstack(samples_idx)
            H = np.array([np.mean(H[samples_idx == idx]) for idx in np.unique(samples_idx)])
        return H


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
    # shuffle = True, split_size = 0.8, batch_size = 1000, num_jobs = 10
    st = ActiveStratification(shuffle=True, split_size=0.8, batch_size=100, num_epochs=5, num_jobs=2)
    training_idx, test_idx = st.fit(X=X, y=y, use_extreme=use_extreme)
    training_idx, dev_idx = st.fit(X=X[training_idx], y=y[training_idx],
                                   use_extreme=use_extreme)

    print("\n{0}".format(60 * "-"))
    print("## Summary...")
    print("\t>> Training set size: {0}".format(len(training_idx)))
    print("\t>> Validation set size: {0}".format(len(dev_idx)))
    print("\t>> Test set size: {0}".format(len(test_idx)))
