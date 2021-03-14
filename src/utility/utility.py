import os

import numpy as np
import scipy as sp

DIRECTORY_PATH = os.getcwd()
DIRECTORY_PATH = DIRECTORY_PATH.split(os.sep)
DIRECTORY_PATH = os.sep.join(DIRECTORY_PATH[:-3])


def check_type(X):
    checked = False
    if isinstance(X, sp.sparse.bsr_matrix):
        checked = True
    elif isinstance(X, sp.sparse.csc_matrix):
        checked = True
    elif isinstance(X, sp.sparse.csr_matrix):
        checked = True
    elif isinstance(X, sp.sparse.lil_matrix):
        checked = True
    elif isinstance(X, np.ndarray):
        X = sp.sparse.lil_matrix(X)
        checked = True
    return checked, X


class LabelBinarizer(object):
    def __init__(self, labels):
        if not isinstance(labels, list):
            tmp = "The classes only supports list type of data."
            raise Exception(tmp)
        self.labels = labels

    def transform(self, X):
        num_labels = len(self.labels)
        X_transform = sp.sparse.lil_matrix((X.shape[0], num_labels), dtype="int")
        for idx, item in enumerate(X.data):
            temp = 0 if not item else item[0]
            X_transform[idx, self.labels.index(temp)] = 1
        return X_transform

    def reassign_labels(self, X, mapping_labels):
        num_labels = len(self.labels)
        X_transform = sp.sparse.lil_matrix((X.shape[0], num_labels), dtype="int")
        for idx, item in enumerate(X):
            X_transform[idx, mapping_labels[item.nonzero()[1]]] = 1
        return X_transform
