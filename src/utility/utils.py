import numpy as np
from scipy.sparse import lil_matrix, issparse, eye


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


def normalize_laplacian(A, sigma: float = 2.0, return_adj: bool = False, norm_adj: bool = False):
    """Normalize graph Laplacian matrix.

    Parameters
    ----------
    A : {array-like, sparse matrix} of shape (n_labels, n_labels)
        Matrix `A`.

    sigma : float, default=2.0
        Scaling component to the graph degree matrix.
        It should be greater than 0.

    return_adj : bool, default=False
        Whether or not to return adjacency matrix or normalized Laplacian matrix.

    norm_adj : bool, default=False
        Whether or not to normalize adjacency matrix.

    Returns
    -------
    cluster labels : a list of clusters defining a cluster to a label association
    """

    if sigma < 0.0:
        sigma = 2

    def __scale_diagonal(D):
        D = D.sqrt()
        D = D.power(-1)
        return D

    A.setdiag(values=0)
    D = lil_matrix(A.sum(axis=1))
    D = D.multiply(eye(D.shape[0]))
    if return_adj:
        if norm_adj:
            D_inv = D.power(-0.5)
            A = D_inv.dot(A).dot(D_inv)
        return A
    else:
        L = D - A
        D = __scale_diagonal(D=D) / sigma
        return D.dot(L.dot(D))


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
        self.labels = labels[0]

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
