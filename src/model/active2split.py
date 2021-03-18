import os
import pickle as pkl
import time
import warnings

import numpy as np
from joblib import Parallel, delayed

from src.utility.file_path import DATASET_PATH

EPSILON = np.finfo(np.float).eps
np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class ActiveStratification(object):

    def __init__(self, subsample_input_size=0.3, subsample_labels_size=50,
                 calc_ads=True, acquisition_type="variation", top_k=20, ads_percent=0.7,
                 tol_labels_iter=10, delay_factor=1.0, forgetting_rate=0.9, split_size=30, max_inner_iter=100,
                 num_epochs=3,
                 num_jobs=2, shuffle=True):

        self.subsample_input_size = subsample_input_size
        self.subsample_labels_size = subsample_labels_size
        self.calc_ads = calc_ads
        self.ads_percent = ads_percent
        self.acquisition_type = acquisition_type  # entropy, mutual, variation, psp
        self.top_k = top_k
        self.tol_labels_iter = tol_labels_iter
        self.forgetting_rate = forgetting_rate
        self.delay_factor = delay_factor
        self.split_size = split_size
        self.max_inner_iter = max_inner_iter
        self.num_epochs = num_epochs
        self.num_jobs = num_jobs
        self.shuffle = shuffle

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

    def __batch_predictive_uncertainty(self, score, samples_idx, y_true=None):
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

    file_path = os.path.join(DATASET_PATH, y_name)
    with open(file_path, mode="rb") as f_in:
        y = pkl.load(f_in)
        idx = list(set(y.nonzero()[0]))
        y = y[idx]

    file_path = os.path.join(DATASET_PATH, X_name)
    with open(file_path, mode="rb") as f_in:
        X = pkl.load(f_in)
        X = X[idx]

    st = ExtremeStratification(split_size=0.8)
    training_idx, test_idx = st.fit(y=X)
    training_idx, dev_idx = st.fit(y=X[training_idx])

    print("training set size: {0}".format(len(training_idx)))
    print("validation set size: {0}".format(len(dev_idx)))
    print("test set size: {0}".format(len(test_idx)))
