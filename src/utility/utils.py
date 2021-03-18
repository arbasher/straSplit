import numpy as np
from scipy.sparse import lil_matrix, issparse


def check_type(X, return_list: bool = False):
    checked = False
    if issparse(X):
        checked = True
    elif isinstance(X, np.ndarray):
        X = lil_matrix(X)
        checked = True
    elif isinstance(X, list):
        X = lil_matrix(X)
        checked = True

    if return_list:
        X = X.toarray().tolist()
    return checked, X


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    temp = e_x / e_x.sum()
    return temp.tolist()


class LabelBinarizer(object):
    def __init__(self, labels):
        check, labels = check_type(X=labels, return_list=True)
        if not check:
            tmp = "The classes only supports scipy.sparse, numpy.ndarray, and list type of data"
            raise Exception(tmp)
        self.labels = labels

    def transform(self, X):
        num_labels = len(self.labels)
        X_transform = lil_matrix((X.shape[0], num_labels), dtype="int")
        for idx, item in enumerate(X.data):
            temp = 0 if not item else item[0]
            X_transform[idx, self.labels.index(temp)] = 1
        return X_transform

    def reassign_labels(self, X, mapping_labels):
        num_labels = len(self.labels)
        X_transform = lil_matrix((X.shape[0], num_labels), dtype="int")
        for idx, item in enumerate(X):
            X_transform[idx, mapping_labels[item.nonzero()[1]]] = 1
        return X_transform
