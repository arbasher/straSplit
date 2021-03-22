'''
The predictive uncertainty approach to group examples with
high informativeness into training set using a calibrated
stratified splitting an extreme multi-label dataset algorithm.
'''
import os
import pickle as pkl
import sys
import textwrap
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.linear_model import SGDClassifier

from extreme2split import ExtremeStratification
from utils import DATASET_PATH, RESULT_PATH, DATASET
from utils import check_type, data_properties, LabelBinarizer
from utils import custom_shuffle

EPSILON = np.finfo(np.float).eps
UPPER_BOUND = np.log(sys.float_info.max) * 10
LOWER_BOUND = np.log(sys.float_info.min) * 10
np.random.seed(12345)
np.seterr(divide='ignore', invalid='ignore')


class ActiveStratification(object):
    def __init__(self, subsample_labels_size: int = 50, acquisition_type: str = "psp", top_k: int = 20,
                 calc_ads: bool = True, ads_percent: float = 0.7, use_solver: bool = True, loss_function: str = "hinge",
                 swap_probability: float = 0.1, threshold_proportion: float = 0.1, decay: float = 0.1,
                 penalty: str = 'elasticnet', alpha_elastic: float = 0.0001, l1_ratio: float = 0.65,
                 alpha_l21: float = 5, loss_threshold: float = 0.05, shuffle: bool = True, split_size: float = 0.75,
                 batch_size: int = 100, num_epochs: int = 50, lr: float = 1e-3, display_interval: int = 1,
                 num_jobs: int = 2):
        """The predictive uncertainty approach to splitting a multi-label
            data using the stratification approach.

        Parameters
        ----------
        subsample_labels_size : int, default=50
            Subsampling labels.

        acquisition_type: str, default="psp"
            The acquisition function for estimating the predictive
            uncertainty. {"entropy" or "psp"}

        top_k : int, default=20
            Top k labels to be considered for calculating the model
            for psp acquisition function.

        calc_ads : bool, default=False
            A boolean variable indicating whether to subsample dataset
            using ADS.

        ads_percent : float, default=0.7
            Proportion of active dataset subsampling.

        use_solver : bool, default=True
            Whether to apply sklearn optimizers.

        loss_function : str, default="hinge"
            The loss function to be used.

        swap_probability : float, default=0.1
            A hyper-parameter for stratification.

        threshold_proportion : float, default=0.1
            A hyper-parameter for stratification.

        decay : float, default=0.1
            A hyper-parameter for stratification.

        penalty: str, default="elasticnet"
            The penalty (aka regularization term). {"l1", "l2", "elasticnet", or "l21"}

        alpha_elastic : float, default=0.0001
            Constant controlling elastic term.

        l1_ratio : float, default=0.0001
            The elastic net mixing parameter.

        alpha_l21 : float, default=5
            Constant controlling l21 term.

        loss_threshold : float, default=0.05
            A cutoff threshold between two consecutive rounds.

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

        display_interval : int, default=1
            The frequency of cost calculation.

        num_jobs : int, default=2
            The number of parallel jobs to run for splitting.
            ``-1`` means using all processors.
        """

        if subsample_labels_size < 0:
            swap_probability = 50
        self.subsample_labels_size = subsample_labels_size

        if acquisition_type == "entropy" or acquisition_type == "psp":
            self.acquisition_type = acquisition_type
        else:
            self.acquisition_type = "psp"

        if top_k < 0:
            top_k = 20
        self.top_k = top_k

        self.calc_ads = calc_ads
        if calc_ads:
            if ads_percent < 0.0 or ads_percent >= 1.0:
                ads_percent = 0.7
            self.ads_percent = ads_percent

        self.use_solver = use_solver
        self.loss_function = loss_function

        if swap_probability < 0.0:
            swap_probability = 0.1
        self.swap_probability = swap_probability

        if threshold_proportion < 0.0:
            threshold_proportion = 0.1
        self.threshold_proportion = threshold_proportion

        if decay < 0.0:
            decay = 0.1
        self.decay = decay

        if penalty in {'l2', 'l1', 'elasticnet'}:
            self.penalty = penalty
        elif penalty == 'l21':
            self.penalty = penalty
        else:
            self.penalty = "elasticnet"

        if alpha_elastic < 0.0:
            alpha_elastic = 0.0001
        self.alpha_elastic = alpha_elastic

        if l1_ratio >= 1.0 or l1_ratio <= 0.0:
            l1_ratio = 0.65
        self.l1_ratio = l1_ratio

        if alpha_l21 < 0.0:
            alpha_l21 = 0.01
        self.alpha_l21 = alpha_l21

        if loss_threshold < 0.0:
            loss_threshold = 0.05
        self.loss_threshold = loss_threshold

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

        if lr <= 0.0:
            lr = 0.0001
        self.lr = lr

        if display_interval <= 0:
            display_interval = 1
        self.display_interval = display_interval

        if num_jobs <= 0:
            num_jobs = 2
        self.num_jobs = num_jobs
        self.is_fit = False

        warnings.filterwarnings("ignore", category=Warning)

        self.__print_arguments()
        time.sleep(2)

    def __print_arguments(self, **kwargs):
        desc = "## Configuration parameters to estimating examples " \
               "predictive uncertainty scores to group example with " \
               "high informativeness into training set using a modified " \
               "approach to splitting an extreme large scale multi-label " \
               "dataset:"

        argdict = dict()
        argdict.update({'subsample_labels_size': 'Subsampling labels: {0}'.format(self.subsample_labels_size)})
        argdict.update({'acquisition_type': 'The acquisition function for estimating the predictive '
                                            'uncertainty: {0}'.format(self.acquisition_type)})
        if self.acquisition_type == "psp":
            argdict.update({'top_k': 'Top k labels to be considered for calculating the model '
                                     'for psp acquisition function: {0}'.format(self.top_k)})
        if self.calc_ads:
            argdict.update({'calc_ads': 'Whether subsample dataset: {0}'.format(self.calc_ads)})
            argdict.update({'ads_percent': 'Proportion of active dataset subsampling: {0}'.format(self.ads_percent)})
        argdict.update({'use_solver': 'Apply sklearn optimizers? {0}'.format(self.use_solver)})
        argdict.update({'loss_function': 'The loss function: {0}'.format(self.loss_function)})
        argdict.update({'swap_probability': 'A hyper-parameter for '
                                            'extreme stratification: {0}'.format(self.swap_probability)})
        argdict.update({'threshold_proportion': 'A hyper-parameter for '
                                                'extreme stratification: {0}'.format(self.threshold_proportion)})
        argdict.update({'decay': 'A hyper-parameter for extreme stratification: {0}'.format(self.decay)})
        argdict.update({'penalty': 'The penalty (aka regularization term): {0}'.format(self.penalty)})
        if self.penalty == "elasticnet":
            argdict.update({'alpha_elastic': 'Constant controlling the elastic term: {0}'.format(self.alpha_elastic)})
            argdict.update({'l1_ratio': 'The elastic net mixing parameter: {0}'.format(self.l1_ratio)})
        if self.penalty == "l21":
            argdict.update({'alpha_l21': 'Constant controlling the l21 term: {0}'.format(self.alpha_l21)})
        argdict.update({'loss_threshold': 'A cutoff threshold between '
                                          'two consecutive rounds: {0}'.format(self.loss_threshold)})
        argdict.update({'shuffle': 'Shuffle the dataset? {0}'.format(self.shuffle)})
        argdict.update({'split_size': 'Split size: {0}'.format(self.split_size)})
        argdict.update({'batch_size': 'Number of examples to use in '
                                      'each iteration: {0}'.format(self.batch_size)})
        argdict.update({'num_epochs': 'Number of loops over training set: {0}'.format(self.num_epochs)})
        argdict.update({'lr': 'Learning rate: {0}'.format(self.lr)})
        argdict.update({'display_interval': 'How often to evaluate? {0}'.format(self.display_interval)})
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

    def __check_bounds(self, X):
        X = np.clip(X, LOWER_BOUND, UPPER_BOUND)
        if len(X.shape) > 1:
            if X.shape[0] == X.shape[1]:
                min_x = np.min(X) + EPSILON
                max_x = np.max(X) + EPSILON
                X = X - min_x
                X = X / (max_x - min_x)
                X = 2 * X - 1
        return X

    def __init_variables(self, num_labels, num_features):
        """Initialize latent variables.
        """
        # initialize parameters
        self.coef = np.zeros(shape=(num_labels, num_features))
        self.intercept = np.zeros(shape=(num_labels, 1))

    def __optimal_learning_rate(self, alpha):
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

    def __sigmoid(self, X):
        return expit(X)

    def __log_logistic(self, X, negative=True):
        param = 1
        if negative:
            param = -1
        X = np.clip(X, EPSILON, 1 - EPSILON)
        X = param * np.log(1 + np.exp(X))
        return X

    def __grad_l21_norm(self, M):
        if len(M.shape) == 2:
            D = 1 / (2 * np.linalg.norm(M, axis=1))
            ret = np.dot(np.diag(D), M)
        else:
            D = (2 * np.linalg.norm(M) + EPSILON)
            ret = M / D
        return ret

    def __solver(self, X, y, coef, intercept):
        if self.penalty == 'l21':
            penalty = "none"
        else:
            penalty = self.penalty
        estimator = SGDClassifier(loss=self.loss_function, penalty=penalty, alpha=self.alpha_elastic,
                                  l1_ratio=self.l1_ratio, shuffle=self.shuffle, n_jobs=self.num_jobs,
                                  random_state=12345, warm_start=True, average=True)
        estimator.fit(X=X, y=y, coef_init=coef, intercept_init=intercept)
        return estimator.coef_[0], estimator.intercept_

    def __optimize_theta(self, X, y, learning_rate, batch_idx, total_progress):
        num_examples = X.shape[0]
        X = X.toarray()
        y = np.array(y.toarray(), dtype=np.float32)

        labels = np.arange(self.num_labels)
        if self.num_labels > self.subsample_labels_size:
            labels = np.random.choice(labels, self.subsample_labels_size, replace=False)
            labels = np.sort(labels)
        count = 1
        current_progress = batch_idx * len(labels)
        for label_idx in labels:
            desc = '\t\t\t--> Optimizing {0}: {1:.2f}%...'.format("Theta",
                                                                  ((current_progress + count) / total_progress) * 100)
            if total_progress == current_progress + count:
                print(desc)
            else:
                print(desc, end="\r")
            count += 1
            gradient = 0.0

            if self.use_solver:
                coef = np.reshape(self.coef[label_idx], newshape=(1, self.coef[label_idx].shape[0]))
                intercept = self.intercept[label_idx]
                coef, intercept = self.__solver(X=X, y=y[:, label_idx], coef=coef, intercept=intercept)
                self.coef[label_idx] = coef
                self.intercept[label_idx] = intercept
            else:
                coef_intercept = self.coef[label_idx]
                X_tmp = np.concatenate((np.ones((num_examples, 1)), X), axis=1)
                coef_intercept = np.hstack((self.intercept[label_idx], coef_intercept))
                cond = -(2 * y[:, label_idx] - 1)
                coef = np.dot(X_tmp, coef_intercept)
                coef = np.multiply(coef, cond)
                logit = 1 / (np.exp(-coef) + 1)
                coef = np.multiply(X_tmp, cond[:, np.newaxis])
                coef = np.multiply(coef, logit[:, np.newaxis])
                coef = np.mean(coef, axis=0)
                del logit, coef_intercept
                self.coef[label_idx] = self.coef[label_idx] - learning_rate * coef[1:]
                self.intercept[label_idx] = coef[0]
                if self.penalty != "l21":
                    l1 = self.l1_ratio * np.sign(self.coef[label_idx])
                    l2 = (1 - self.l1_ratio) * 2 * self.coef[label_idx]
                    if self.penalty == "elasticnet":
                        gradient += self.alpha_elastic * (l1 + l2)
                    if self.penalty == "l1":
                        gradient += self.alpha_elastic * l1
                    if self.penalty == "l2":
                        gradient += self.alpha_elastic * l2

            # compute the constraint lambda_5 * D_Theta^path * Theta^path
            if self.penalty == "l21":
                gradient += self.alpha_l21 * self.__grad_l21_norm(M=self.coef[label_idx])

            # gradient of Theta^path = Theta^path_old + learning_rate * gradients
            tmp = self.coef[label_idx] - learning_rate * gradient
            self.coef[label_idx] = self.__check_bounds(tmp)

    def __parallel_backward(self, X, y, learning_rate, examples_idx):
        print('  \t\t<<<------------<<<------------<<<')
        print('  \t\t>> Feed-Backward...')

        X_tmp = X[examples_idx]
        y_tmp = y[examples_idx]
        list_batches = np.arange(start=0, stop=len(examples_idx), step=self.batch_size)
        num_labels = self.num_labels
        if num_labels > self.subsample_labels_size:
            num_labels = self.subsample_labels_size
        total_progress = len(list_batches) * num_labels
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads", verbose=0)
        parallel(delayed(self.__optimize_theta)(X_tmp[batch:batch + self.batch_size],
                                                y_tmp[batch:batch + self.batch_size],
                                                learning_rate, batch_idx, total_progress)
                 for batch_idx, batch in enumerate(list_batches))

    def __label_prob(self, X, labels, transform=False):
        if len(labels) == 0:
            labels = np.arange(self.num_labels)
        coef_intercept = self.coef[labels]
        coef_intercept = np.hstack((self.intercept[labels], coef_intercept))
        prob_label = self.__sigmoid(np.dot(X, coef_intercept.T))
        if not transform:
            prob_label = np.mean(prob_label, axis=0)
        return prob_label

    def __forward(self, X, y, batch_idx, total_progress):
        num_examples = X.shape[0]
        X = np.concatenate((np.ones((num_examples, 1)), X), axis=1)
        num_labels_example = np.sum(y, axis=0)
        weight_labels = 1 / num_labels_example
        weight_labels[weight_labels == np.inf] = 0.0
        weight_labels = weight_labels / np.sum(weight_labels)
        labels = np.unique(np.where(y == 1)[1])
        if labels.shape[0] > self.subsample_labels_size:
            labels = np.random.choice(labels, self.subsample_labels_size, replace=False, p=weight_labels[labels])
            labels = np.sort(labels)

        # compute probability of labels
        prob = self.__label_prob(X=X, labels=labels, transform=True)

        # compute probability of bags based on labels
        tmp = np.zeros((num_examples, self.num_labels)) + EPSILON
        tmp[:, labels] = prob
        prob = tmp
        prob[np.where(y == 0)] = EPSILON
        desc = '\t\t\t--> Computed {0:.4f}%...'.format(((batch_idx + 1) / total_progress * 100))
        print(desc, end="\r")
        return prob

    def __parallel_forward(self, X, y, example_idx):
        print('  \t\t>>>------------>>>------------>>>')
        print('  \t\t>> Feed-Forward...')

        X_tmp = X[example_idx].toarray()
        y_tmp = y[example_idx].toarray()
        list_batches = np.arange(start=0, stop=len(example_idx), step=self.batch_size)
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads", verbose=0)
        prob = parallel(delayed(self.__forward)(X_tmp[batch:batch + self.batch_size],
                                                y_tmp[batch:batch + self.batch_size],
                                                batch_idx, len(list_batches))
                        for batch_idx, batch in enumerate(list_batches))
        # merge result
        prob = np.vstack(prob)
        return prob

    def __entropy(self, prob):
        log_prob_bag = np.log(prob + EPSILON)
        if len(prob.shape) > 1:
            entropy_ = -np.diag(np.dot(prob, log_prob_bag.T))
        else:
            entropy_ = -np.multiply(prob, log_prob_bag)
        np.nan_to_num(entropy_, copy=False)
        entropy_ = entropy_ + EPSILON
        return entropy_

    def __psp(self, y_true, prob):
        num_labels = y_true.shape[1]

        # propensity of all labels
        N_j = y_true.toarray()
        labels_sum = np.sum(N_j, axis=0)
        g = 1 / (labels_sum + 1)
        psp_label = 1 / (1 + g)

        # retrieve the top k labels
        top_k = y_true.shape[1] if self.top_k > num_labels else self.top_k
        labels_idx = np.argsort(-prob)[:, :top_k]

        # compute normalized psp@k
        psp = N_j / psp_label
        tmp = [psp[s_idx, labels_idx[s_idx]] for s_idx in range(psp.shape[0])]
        psp = (1 / top_k) * np.sum(tmp, axis=1)
        min_psp = np.min(psp) + EPSILON
        max_psp = np.max(psp) + EPSILON
        psp = psp - min_psp
        psp = psp / (max_psp - min_psp)
        psp = 1 - psp + EPSILON
        return psp

    def __predictive_uncertainty(self, prob, y=None):
        desc = '  \t\t>> Predictive uncertainty using {0}...'.format(self.acquisition_type)
        print(desc)
        if self.acquisition_type == "entropy":
            H = self.__entropy(prob=np.mean(prob, axis=1))
        else:
            H = self.__psp(y_true=y, prob=prob)
        return H

    def __subsample_strategy(self, H, num_examples):
        sub_sampled_size = int(self.ads_percent * num_examples)
        sorted_idx = np.argsort(H)[::-1]
        init_samples = sorted_idx[:sub_sampled_size]
        return init_samples

    def __cost(self, X, y, label_idx):
        desc = '\t\t\t--> Calculating cost: {0:.2f}%...'.format((((label_idx + 1) / self.num_labels) * 100))
        print(desc, end="\r")
        coef_intercept = self.coef[label_idx]
        coef_intercept = np.hstack((self.intercept[label_idx], coef_intercept))
        cond = -(2 * y[:, label_idx] - 1)
        coef = np.dot(X, coef_intercept)
        coef = np.multiply(coef, cond)
        cost_label = -np.mean(self.__log_logistic(coef))
        return cost_label

    def __parallel_cost(self, X, y):
        print('  \t\t>> Compute cost...')
        # properties of dataset
        num_examples = X.shape[0]
        X_tmp = X.toarray()
        y_tmp = y.toarray()
        X_tmp = np.concatenate((np.ones((num_examples, 1)), X_tmp), axis=1)

        # estimate expected cost over all labels
        parallel = Parallel(n_jobs=self.num_jobs, prefer="threads", verbose=0)
        results = parallel(delayed(self.__cost)(X_tmp, y_tmp, label_idx)
                           for label_idx in np.arange(self.num_labels))
        cost_label = np.mean(results)
        del results
        cost = cost_label + EPSILON
        return cost

    def fit(self, X, y):
        """Split multi-label y dataset into train and test subsets.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_examples, n_features).

        y : {array-like, sparse matrix} of shape (n_examples, n_labels).

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
            temp = "The method only supports scipy.sparse, numpy.ndarray, and list type of data"
            raise Exception(temp)
        check, y = check_type(X=y, return_list=False)
        if not check:
            temp = "The method only supports scipy.sparse, numpy.ndarray, and list type of data"
            raise Exception(temp)

        # collect properties from data
        num_examples, num_features = X.shape
        self.num_labels = y.shape[1]

        # check whether data is singly labeled
        if self.num_labels == 1:
            # transform it to multi-label data
            classes = list(set([i[0] if i else 0 for i in y.data]))
            mlb = LabelBinarizer(labels=classes)
            y = mlb.transform(y)
            self.num_labels = y.shape[1]

        if not self.is_fit:
            print('\t>> Training to learn a model...')
            self.__init_variables(num_labels=self.num_labels, num_features=num_features)

            old_cost = np.inf
            optimal_init = self.__optimal_learning_rate(alpha=self.lr)
            n_epochs = self.num_epochs + 1
            timeref = time.time()

            for epoch in np.arange(start=1, stop=n_epochs):
                desc = '\t   {0:d})- Epoch count ({0:d}/{1:d})...'.format(epoch, n_epochs - 1)
                print(desc)

                # shuffle dataset
                if epoch == 1:
                    example_idx = custom_shuffle(num_examples=num_examples)
                    example_idx = list(example_idx)
                    X = X[example_idx, :]
                    y = y[example_idx, :]
                else:
                    if self.calc_ads:
                        temp = [s for s in range(num_examples) if s not in example_idx]
                        sub_sampled_size = int(self.ads_percent * len(temp))
                        temp = list(np.random.choice(a=temp, size=sub_sampled_size, replace=False))
                        example_idx.extend(temp)

                # usual optimization technique
                learning_rate = 1.0 / (self.lr * (optimal_init + epoch - 1))

                # set epoch time
                start_epoch = time.time()

                self.__parallel_backward(X=X, y=y, learning_rate=learning_rate, examples_idx=example_idx)
                prob = self.__parallel_forward(X=X, y=y, example_idx=example_idx)
                H = self.__predictive_uncertainty(prob=prob, y=y[example_idx])
                if self.calc_ads:
                    example_idx = self.__subsample_strategy(H=H, num_examples=num_examples)
                    example_idx = list(example_idx)
                    H = H[example_idx]

                end_epoch = time.time()
                self.is_fit = True

                # Save models parameters based on test frequencies
                if (epoch % self.display_interval) == 0 or epoch == n_epochs - 1:
                    # compute loss
                    new_cost = self.__parallel_cost(X=X[example_idx], y=y[example_idx])
                    print('\t\t\t--> New cost: {0:.4f}; Old cost: {1:.4f}'.format(new_cost, old_cost))
                    if old_cost >= new_cost or epoch == n_epochs - 1:
                        old_cost = new_cost
                print('\t\t\t--> Epoch {0} took {1} seconds...'.format(epoch, round(end_epoch - start_epoch, 3)))
            print('\t  --> Training consumed %.2f mintues' % (round((time.time() - timeref) / 60., 3)))
        else:
            print('\t>> Estimating examples scores...')
            example_idx = list(range(num_examples))
            prob = self.__parallel_forward(X=X, y=y, example_idx=example_idx)
            H = self.__predictive_uncertainty(prob=prob, y=y[example_idx])
            if self.calc_ads:
                example_idx = self.__subsample_strategy(H=H, num_examples=num_examples)
                example_idx = list(example_idx)
                H = H[example_idx]

        X = X[example_idx]
        y = y[example_idx]
        example_idx = list(range(len(example_idx)))
        examples_scores = dict(list(zip(example_idx, H)))

        # perform calibrated splitting
        extreme = ExtremeStratification(swap_probability=self.swap_probability,
                                        threshold_proportion=self.threshold_proportion,
                                        decay=self.decay, shuffle=self.shuffle,
                                        split_size=self.split_size, num_epochs=self.num_epochs,
                                        verbose=False)
        train_list, test_list = extreme.fit(X=X, y=y, examples_scores=examples_scores)
        return train_list, test_list


if __name__ == "__main__":
    model_name = "active2split"
    split_type = "extreme"
    split_size = 0.80
    num_epochs = 5
    num_jobs = 10
    use_solver = False

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

        st = ActiveStratification(subsample_labels_size=10, acquisition_type="entropy", top_k=5, calc_ads=False,
                                  ads_percent=0.7, use_solver=use_solver, loss_function="hinge", swap_probability=0.1,
                                  threshold_proportion=0.1, decay=0.1, penalty='elasticnet', alpha_elastic=0.0001,
                                  l1_ratio=0.65, alpha_l21=0.01, loss_threshold=0.05, shuffle=True,
                                  split_size=split_size, batch_size=500, num_epochs=num_epochs, lr=1e-3,
                                  display_interval=1, num_jobs=num_jobs)
        training_idx, test_idx = st.fit(X=X, y=y)

        data_properties(y=y.toarray(), selected_examples=[training_idx, test_idx], num_tails=5, dataset_name=dsname,
                        model_name=model_name, rspath=RESULT_PATH, display_dataframe=False)
        print("\n{0}\n".format(60 * "-"))
