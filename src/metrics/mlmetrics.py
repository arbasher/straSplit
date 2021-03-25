import math
import os
import pickle as pkl
from collections import Counter

import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import chi2_contingency, pearsonr, kurtosis, skew


###********************          Dimensionality metrics         ********************###

def instances(X):
    '''
    This metric indicates the number of input instances.

    Parameters
    ----------
    X

    Returns
    -------
    An integer representing the number of instances.
    '''

    return X.shape[0]


def attributes(X):
    '''
    This metric indicates the number of input features of the dataset.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    Returns
    -------
    An integer representing the number of features.
    '''

    return X.shape[1]


def labels(y):
    '''
    This metric indicates the number of labels.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    An integer representing the number of labels.
    '''

    return y.shape[1]


def distinct_labels(y):
    '''
    This metric indicates the number of distinct labels in a data.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    An integer representing the number of distinct labels.
    '''

    return lil_matrix(y.sum(0)).nnz


def distinct_labelsets(y, return_labels: bool = False):
    '''
    This metric indicates the number of distinct label set in a data.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    return_labels: a boolean variable indicating whether to return a
    list of distinct labelsets or not.

    Returns
    -------
    An integer representing the number of distinct label set.
    '''

    distinct = list()
    for idx in range(y.shape[0]):
        temp = sorted(list(y[idx].nonzero()[1]))
        if temp in distinct:
            continue
        distinct.extend([temp])
    if return_labels:
        return distinct
    else:
        return len(distinct)


def lxixf(X, y):
    '''
    This metric indicates the number of labels * number of instances
    * number of features in a data.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    An integer representing the multiplication of labels * instances * features.
    '''

    return labels(y) * instances(X) * attributes(X)


def ratio_instances_features(X):
    '''
    This metric calculates the proportion between the number of instances
    and the number of attributes in a data.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    Returns
    -------
    A numerical value representing the proportion.
    '''
    return instances(X) / attributes(X)


###********************        Label distribution metrics       ********************###

def cardinality(y):
    '''
    This metric is defined as the mean number of labels associated
    for instance.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the cardinality.
    '''

    temp = int(y.sum())
    card = temp / y.shape[0]
    return card


def density(y):
    '''
    This metric is defined as cardinality divided by the number
    of labels.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the density.
    '''

    dens = cardinality(y) / labels(y)
    return dens


def frequency(y, l: int):
    '''
    This metric is defined as the number of appearances of this
    label divided by the total number of instances.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    l : an integer index representing a class label.

    Returns
    -------
    A numerical value representing the frequency of a label.
    '''

    temp = y[:, l]
    temp = int(temp.sum())
    freq_l = temp / y.shape[0]
    return freq_l


def std_label_cardinality(y):
    '''
    This metric calculates the standard deviation of the number of
    labels associated with each instance.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).


    Returns
    -------
    A numerical value representing the standard deviation of label
    cardinality.
    '''

    card = cardinality(y=y)
    std_card = 0
    for idx in range(instances(y)):
        std_card += (y[idx].nnz - card) ** 2
    std_card /= y.shape[0]
    std_card = math.sqrt(std_card)
    return std_card


def custom_entropy(dist):
    '''
    This metrics returns the entropy of a distribution.

    Parameters
    ----------
    dist : a distribution.

    Returns
    -------
    A numerical value corresponding the entropy of the distribution.
    '''

    p = dist / np.sum(dist, axis=0)
    pc = np.clip(p, 1e-15, 1)
    H = np.sum(np.sum(-p * np.log(pc), axis=0) * np.sum(dist, axis=0) / np.sum(dist))
    return H


def entropy_(X):
    '''
    This metric calculates the entropy of a nominal attribute or a label.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features
        or n_labels).

    Returns
    -------
    A numerical value indicating the mean entropy.
    '''

    dist = Counter([int(item) for item in X])
    dist = np.array([v for k, v in dist.items()])
    H = custom_entropy(dist=dist)
    return H


def min_max_mean_entropy_labels(y):
    '''
    This metrics returns the minimum, maximum and mean of entropies of
    labels.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    Three values corresponding minimum, maximum, and mean entropy of labels.
    '''

    H = []
    for l in range(labels(y)):
        temp = entropy_(y[:, l].toarray().astype(int))
        H.append((l, temp))
    H = sorted(H, key=lambda x: x[1])
    min_entropy = H[0][0]
    max_entropy = H[::-1][0][0]
    mean_entropy = sum([item[1] for item in H]) / len(H)
    return min_entropy, max_entropy, mean_entropy


###********************             Imbalance metrics           ********************###

def imbalance_ratio_inter_class(y, l: int):
    '''
    This imbalance ratio inter-class metric is obtained by dividing the
    number of positive examples of most frequent label by the number of
    positive examples of the given label.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    l : an integer index representing a class label.

    Returns
    -------
    A numerical value representing the imbalance ratio inter-class of a label.
    '''

    max_freq = list(y.sum(0))[0].max()
    ir = 0.0
    if y[:, l].sum() >= 1:
        ir = max_freq / y[:, l].sum()
    return ir


def mean_ir_inter_class(y):
    '''
    This metric is defined as the average imbalance ratio for all labels.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the mean imbalance ratio inter-class for
    all labels.
    '''

    mean_ir = sum([imbalance_ratio_inter_class(y=y, l=l) for l in range(labels(y))])
    mean_ir /= labels(y)
    return mean_ir


def max_ir_inter_class(y):
    '''
    This metric returns the mode of imbalance ratio inter-class corresponding
    the most imbalanced label.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    An integer value representing the mode of imbalance ratio inter-class.
    '''

    max_ir = [(l, imbalance_ratio_inter_class(y=y, l=l)) for l in range(labels(y))]
    max_ir = sorted(max_ir, key=lambda x: x[1], reverse=True)[0][0]
    return max_ir


def cvir_inter_class(y):
    '''
    This metric is defined as the coefficient of variation of imbalance ratio
    inter-class.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the cvir value.
    '''
    mean_ir = mean_ir_inter_class(y)
    sigma_ir = 0.0
    for l in range(labels(y)):
        sigma_ir += (imbalance_ratio_inter_class(y=y, l=l) - mean_ir) ** 2
    sigma_ir /= (labels(y) - 1)
    cvir_ir = math.sqrt(sigma_ir) / mean_ir
    return cvir_ir


def imbalance_ratio_intra_class(y, l: int):
    '''
    This imbalance ratio intra-class metric measures the degree of
    imbalance inside a label.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    l : an integer index representing a class label.

    Returns
    -------
    A numerical value representing the imbalance ratio intra-class of a label.
    '''
    n = instances(y)
    max_label = max(n, y[:, l].sum())
    min_label = min(n, y[:, l].sum())
    ir = 0.0
    if min_label >= 1:
        ir = max_label / min_label
    return ir


def mean_ir_intra_class(y):
    '''
    This metric is defined as the average imbalance ratio for all labels.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the mean imbalance ratio intra-class for
    all labels.
    '''

    mean_ir = sum([imbalance_ratio_intra_class(y=y, l=l) for l in range(labels(y))])
    mean_ir /= labels(y)
    return mean_ir


def max_ir_intra_class(y):
    '''
    This metric returns the mode of imbalance ratio intra-class corresponding
    the most imbalanced label.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    An integer value representing the mode of imbalance ratio intra-class.
    '''

    max_ir = [(l, imbalance_ratio_intra_class(y=y, l=l)) for l in range(labels(y))]
    max_ir = sorted(max_ir, key=lambda x: x[1], reverse=True)[0][0]
    return max_ir


def std_ir_intra_class(y, l: list):
    '''
    This metric computes the standard deviations intra-class of a label.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    l : an integer index representing a class label.

    Returns
    -------
    A numerical value representing the standard deviations of a label.
    '''
    n = instances(y)
    L_p = y[:, l].sum()
    L_n = n - L_p
    std_ir = (L_p - n / 2) ** 2 + (L_n - n / 2) ** 2
    std_ir = math.sqrt(std_ir / 2)
    return std_ir


def mean_std_ir_intra_class(y):
    '''
    This metric computes the mean of the standard deviations intra-class of
    each label.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the mean of the standard deviations of each
    label.
    '''

    mean_std = sum([std_ir_intra_class(y=y, l=l) for l in range(labels(y))])
    mean_std /= labels(y)
    return mean_std


def imbalance_ratio_labelset(l: int, dl: list, calc_dl: bool = True):
    '''
    This metric is calculated by dividing the number of instances associated
    with the most frequent labelsets by the number of instances of the current
    labelset.

    Parameters
    ----------
    l : an integer index representing a class label.

    calc_dl : whether to calculate distinct label sets.

    Returns
    -------
    A numerical value representing the imbalance ratio label set of a label.
    '''
    if calc_dl:
        dl = distinct_labelsets(y, return_labels=True)
        dl = [y[:, item].sum() / labels(y) for idx, item in enumerate(dl)]

    max_freq = max(dl)
    max_freq = dl.index(max_freq)
    ir = 0.0
    if dl[l] > 0:
        ir = max_freq / dl[l]
    return ir


def mean_ir_labelset(y):
    '''
    This metric is defined as the average imbalance ratio for all labelsets.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the mean imbalance ratio labelsets for
    all labels.
    '''
    dl = distinct_labelsets(y, return_labels=True)
    dl = [y[:, item].sum() / labels(y) for idx, item in enumerate(dl)]
    mean_ir = sum([imbalance_ratio_labelset(l=l, dl=dl, calc_dl=False) for l in range(len(dl))])
    mean_ir /= len(dl)
    return mean_ir


def max_ir_labelset(y):
    '''
    This metric is defined as the maximum imbalance ratio for all labelsets.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    An integer value representing the maximum imbalance ratio label set for
    all labels.
    '''

    dl = distinct_labelsets(y, return_labels=True)
    dl = [y[:, item].sum() / labels(y) for idx, item in enumerate(dl)]
    max_ir = max(dl)
    max_ir = dl.index(max_ir)
    return max_ir


def kurtosis_cardinality(y):
    '''
    This metric applies the Kurtosis formula to the number of associated
    labels of each instance.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the Kurtosis cardinality.
    '''

    n = instances(y)
    card = cardinality(y)
    m2 = [(y[idx].nnz - card) ** 2 for idx in range(instances(y))]
    m4 = [item ** 2 for item in m2]
    m2 = sum(m2) / n
    m4 = sum(m4) / n
    g_ku = (m4 / m2 ** 2) - 3
    kurtosis = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g_ku + 6)
    return kurtosis


def skewness_cardinality(y):
    '''
    This metric measures the skewness of the number of labels associated
    with each instance, using the skewness metric.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the skewness metric.
    '''

    n = instances(y)
    card = cardinality(y)
    temp = sum([(y[idx].nnz - card) ** 3 for idx in range(instances(y))])
    sigma_3 = temp / (n - 1)
    sigma_3 = math.pow(sigma_3, 3 / 2)
    skewness = temp / sigma_3
    skewness = (n / ((n - 1) * (n - 2))) * skewness
    return skewness


def pmax(y):
    '''
    This metric calculates the proportion of instances associated with the
    most frequent labelset.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    An integer value representing the mode of the proportion of maxim label
    combination value.
    '''

    dl = distinct_labelsets(y, return_labels=True)
    n = instances(y)
    dl = [y[:, item].sum() / n for idx, item in enumerate(dl)]
    max_ir = max(dl)
    p_max = dl.index(max_ir)
    return p_max


###********************       Labels relationship metrics       ********************###

def bound(y):
    '''
    This metric represents the maximum number of labelsets that may exist
    in the dataset.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    An integer value representing the bound.
    '''

    return 2 ** labels(y)


def diversity(y):
    '''
    This metric represents the percentage of labelsets present in the
    dataset divided by the number of possible labelsets.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the diversity.
    '''

    dl = distinct_labelsets(y)
    bnd = bound(y)
    return dl / bnd


def scumble(y):
    '''
    This metric aims to quantify the variation of the imbalance among
    labels of each instance.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the variation.
    '''

    scumble_value = 0.0
    labels_value = labels(y)
    n = instances(y)
    for idx in range(n):
        ir_d = [imbalance_ratio_inter_class(y[idx], l) for l in range(labels_value)]
        ir_mean = sum(ir_d) / len(ir_d)
        ir_d = math.prod(ir_d)
        ir_d = math.pow(ir_d, 1 / labels_value)
        if ir_mean > 0:
            ir_d /= ir_mean
        ir_d = 1 - ir_d
        scumble_value += ir_d
    scumble_value /= n
    return scumble_value


def propportion_distinct_labelsets(y):
    '''
    This metric normalizes distinct label sets by the number of instances.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the proportion of distinct labelsets.
    '''

    pdl = distinct_labelsets(y) / y.shape[0]
    return pdl


def number_labelsets_to_n_instances(y, n: int):
    '''
    This metric returns the number of labelsets appearing the up to n
    times in the dataset.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    n : an integer value representing upto n examples.

    Returns
    -------
    An integer value indicating the the number of labelsets .
    '''

    dl = distinct_labelsets(y, return_labels=True)
    num_labelsets = [l for l in dl if y[:, l].sum() <= n]
    return len(num_labelsets)


def ratio_labelsets_to_n_instances(y, n: int):
    '''
    This metric returns the ratio of labelsets appearing the up to n
    times in the dataset.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    n : an integer value representing upto n examples.

    Returns
    -------
    A numerical value indicating the ratio of labelsets.
    '''

    dl = distinct_labelsets(y, return_labels=True)
    num_labelsets = [l for l in dl if y[:, l].sum() <= n]
    return len(num_labelsets) / len(dl)


def average_instances_per_labelset(y):
    '''
    This metric is calculated as the average number of instances
    associated with each labelset.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value indicating the average number of instances.
    '''

    dl = distinct_labelsets(y, return_labels=True)
    return instances(y) / len(dl)


def std_instances_per_labelset(y):
    '''
    This metric calculates the standard deviation of the number of
    instances per labelset.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the standard deviation.
    '''
    dl = distinct_labelsets(y, return_labels=True)
    average_labelsets = average_instances_per_labelset(y=y)
    std_labelsets = sum([(y[:, item].sum() - average_labelsets) ** 2 for item in dl]) / len(dl)
    std_labelsets = math.sqrt(std_labelsets)
    return std_labelsets


def number_unique_labelsets(y):
    '''
    This metric returns the number of labelsets that are only associated
    with an instance of the dataset.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    An integer value indicating the number of unique labelsets .
    '''

    dl = distinct_labelsets(y, return_labels=True)
    num_labelsets = [l for l in dl if y[:, l].sum() == 1]
    return len(num_labelsets)


def ratio_labelsets_half_attributes(X, y):
    '''
    This metric returns the ratio of labelsets which number of appearances
    is less or equal to the half of the number of attributes.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value indicating the ratio of labelsets.
    '''

    dl = distinct_labelsets(y, return_labels=True)
    half_attributes = attributes(X) / 2
    ratio_labelsets = len([l for l in dl if y[:, l].sum() <= half_attributes]) / len(dl)
    return ratio_labelsets


def number_unconditionally_dependent_label_pairs(y):
    '''
    This metric represents the number of pairs of labels unconditionally
    dependent that reject the null hypotesis of the Chi-square test at
    99% confidence.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    An integer value indicating the number of labels pairs.
    '''

    count_labels = labels(y)
    pairwise_label = lil_matrix((count_labels, count_labels))
    for i in range(count_labels):
        for j in range(i + 1, count_labels):
            temp = np.array(y[:, [i, j]].toarray().T, dtype=np.float32)
            temp[temp == 0] = 0.01
            temp = chi2_contingency(temp)[0]
            pairwise_label[i, j] = 1 if temp > 6.635 else 0
    return pairwise_label.sum()


def ratio_unconditionally_dependent_label_pairs(y):
    '''
    This metric the proportion of pairs of labels dependent at 99%
    confidence divided by the number of existing pairs.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing the ratio of labels pairs.
    '''

    ratio_dependent = number_unconditionally_dependent_label_pairs(y)
    ratio_dependent = ratio_dependent * math.pow((labels(y) * (labels(y) - 1)) / 2, -1)
    return ratio_dependent


def average_unconditionally_dependent_label_pairs(y):
    '''
    This metric returns the average value of chi-square test for the
    pairs of labels that reject the null hypothesis.

    Parameters
    ----------
    y : {array-like, sparse matrix} of shape (n_instances, n_labels).

    Returns
    -------
    A numerical value representing average value of chi-square test.
    '''

    count_labels = labels(y)
    pairwise_label = lil_matrix((count_labels, count_labels))
    for i in range(count_labels):
        for j in range(i + 1, count_labels):
            temp = np.array(y[:, [i, j]].toarray().T, dtype=np.float32)
            temp[temp == 0] = 0.01
            temp = chi2_contingency(temp)[0]
            pairwise_label[i, j] = temp if temp > 6.635 else 0
    return pairwise_label.sum()


###********************           Attributes metrics            ********************###

def number_binary_attributes(X):
    '''
    This metric calculates the number of binary attributes of a dataset.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    Returns
    -------
    An integer value indicating the number of binary attributes.
    '''

    num_attributes = attributes(X)
    binary_attributes = [idx for idx in range(num_attributes) if
                         ((X[:, idx].toarray() == 0) | (X[:, idx].toarray() == 1)).all()]
    return len(binary_attributes)


def number_nominal_attributes(X):
    '''
    This metric calculates the number of nominal attributes of a dataset.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    Returns
    -------
    An integer value indicating the number of nominal attributes.
    '''

    num_attributes = attributes(X)
    nominal_attributes = [idx for idx in range(num_attributes) if X[:, idx].toarray().dtype == int]
    return len(nominal_attributes)


def number_numeric_attributes(X):
    '''
    This metric calculates the number of numeric attributes of a dataset.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    Returns
    -------
    An integer value indicating the number of numeric attributes.
    '''

    num_attributes = attributes(X)
    numeric_attributes = [idx for idx in range(num_attributes) if X[:, idx].toarray().dtype == float]
    return len(numeric_attributes)


def proportion_binary_attributes(X):
    '''
    This metric calculates the proposition of binary attributes of a dataset.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    Returns
    -------
    A numerical value indicating the proposition of binary attributes.
    '''

    num_attributes = attributes(X)
    binary_attributes = [idx for idx in range(num_attributes) if
                         ((X[:, idx].toarray() == 0) | (X[:, idx].toarray() == 1)).all()]
    return len(binary_attributes) / num_attributes


def proportion_nominal_attributes(X):
    '''
    This metric calculates the proposition of nominal attributes of a dataset.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    Returns
    -------
    A numerical value indicating the proposition of nominal attributes.
    '''

    num_attributes = attributes(X)
    nominal_attributes = [idx for idx in range(num_attributes) if sum(X[:, idx].toarray().astype(int)) != 0]
    return len(nominal_attributes) / num_attributes


def proportion_numeric_attributes(X):
    '''
    This metric calculates the proposition of numeric attributes of a dataset.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    Returns
    -------
    A numerical value indicating the proposition of numeric attributes.
    '''

    num_attributes = attributes(X)
    numeric_attributes = [idx for idx in range(num_attributes) if X[:, idx].toarray().dtype == float]
    return len(numeric_attributes) / num_attributes


def mean_entropies_nominal_attributes(X, t: int):
    '''
    This metric calculates the entropy of each nominal attribute.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    t : an integer representing the index of the attribute.

    Returns
    -------
    A numerical value indicating the mean entropy.
    '''

    num_instances = instances(X)
    H = entropy_(X[:, t].toarray().astype(int)) / num_instances
    return H


def mean_mean_numeric_attributes(X):
    '''
    This metric calculates the mean of means of all numeric attributes.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    Returns
    -------
    A numerical value indicating the mean-mean numeric attributes.
    '''

    num_instances = instances(X)
    num_attributes = attributes(X)
    mean_mean_numeric = [X[:, idx].sum() / num_instances for idx in range(num_attributes) if
                         X[:, idx].toarray().dtype == float]
    mean_mean_numeric = sum(mean_mean_numeric) / num_instances
    return mean_mean_numeric


def mean_std_numeric_attributes(X):
    '''
    This metric calculates the average of standard deviations of all
    numeric attributes.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    Returns
    -------
    A numerical value indicating the average of standard deviations
    numeric attributes.
    '''

    num_instances = instances(X)
    num_attributes = attributes(X)
    mean_numeric = [X[:, idx].sum() / num_instances for idx in range(num_attributes) if
                    X[:, idx].toarray().dtype == float]
    std_numeric = [math.sqrt(sum((X[:, 0].toarray() - mean_numeric[0]) ** 2) / (num_instances - 1)) for idx in
                   range(num_attributes) if X[:, idx].toarray().dtype == float]
    std_numeric = sum(std_numeric) / num_instances
    return std_numeric


def gain_ratio(X, y, l: int, t: int):
    '''
    This metric calculates the gain ratio.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    y : {array-like, sparse matrix} of shape (n_samples, n_labels).

    l : an integer representing the index of a label.

    t : an integer representing the index of a attribute.

    Returns
    -------
    A numerical value indicating the gain ratio.
    '''

    h_label = entropy_(y[:, l].toarray().astype(int))
    h_attribute = entropy_(X[:, t].toarray().astype(int))
    h_label_attribute = entropy_(np.compress(X[:, t].toarray().reshape((X.shape[0],)), a=y[:, l].toarray(), axis=0))
    return (h_label - h_label_attribute) / h_attribute


def average_gain_ratio(X, y):
    '''
    This metric calculates the average value of gain ratio values for
    all attributes and each label.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).

    y : {array-like, sparse matrix} of shape (n_samples, n_labels).

    Returns
    -------
    A numerical value indicating the average gain ratio.
    '''

    num_attributes = attributes(X)
    num_labels = labels(y)
    total_gain_ratio = 0.0
    for l in range(num_labels):
        for t in range(num_attributes):
            total_gain_ratio += gain_ratio(X=X, y=y, l=l, t=t)
    total_gain_ratio /= num_attributes
    total_gain_ratio /= num_labels
    return total_gain_ratio


def average_absolute_correlation_numeric_attributes(X):
    '''
    This metric calculates the average of the correlation coefficient
    among all pairs of numeric attributes.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).


    Returns
    -------
    A numerical value indicating the average of the correlation coefficient.
    '''

    num_instances = instances(X)
    num_attributes = attributes(X)
    mean_correlation = 0.0
    for i in range(num_attributes):
        for j in range(i + 1, num_attributes):
            mean_correlation += \
                pearsonr(x=X[:, i].toarray().reshape((num_instances,)), y=X[:, j].toarray().reshape((num_instances,)))[
                    0]
    temp = (num_attributes * (num_attributes - 1)) / 2
    temp = math.pow(temp, -1)
    mean_correlation *= temp
    return mean_correlation


def average_kurtosis_numeric_attributes(X):
    '''
    This metric calculates the Kurtosis for each numeric attribute
    and then, it calculates the average.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).


    Returns
    -------
    A numerical value indicating the average of kurtosis for numeric attributes.
    '''

    num_attributes = attributes(X)
    mean_kurtosis = 0.0
    for i in range(num_attributes):
        mean_kurtosis += kurtosis(X[:, i].toarray())[0]
    mean_kurtosis /= num_attributes
    return mean_kurtosis


def average_skewness_numeric_attributes(X):
    '''
    This metric calculates calculates the mean of skewness of numeric attributes.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features).


    Returns
    -------
    A numerical value indicating the average of skewness for numeric attributes.
    '''

    num_attributes = attributes(X)
    mean_skewness = 0.0
    for i in range(num_attributes):
        mean_skewness += skew(X[:, i].toarray())[0]
    mean_skewness /= num_attributes
    return mean_skewness


if __name__ == "__main__":
    from src.model.utils import DATASET_PATH

    dsname = "birds"
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

    temp = mean_ir_intra_class(y)
    print(temp)
    temp = mean_ir_inter_class(y)
    print(temp)
