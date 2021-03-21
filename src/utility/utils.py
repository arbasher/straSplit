import os
import random

import altair as alt
import numpy as np
import pandas as pd
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


def custom_shuffle(num_examples):
    idx = list(range(num_examples))
    random.shuffle(idx)
    return idx


def data_properties(y, selected_examples, num_tails: int = 2, display_full_properties=True, data_name="test",
                    selected_name:str="training set", file_name: str = "tails_bar", rspath: str = "."):
    if display_full_properties:
        L_S = int(np.sum(y))
        LCard_S = L_S / y.shape[0]
        LDen_S = LCard_S / L_S
        DL_S = np.nonzero(np.sum(y, axis=0))[0].size
        PDL_S = DL_S / y.shape[0]
        print('## DATA PROPERTIES for {0}...'.format(data_name))
        print('\t>> Number of examples: {0}'.format(y.shape[0]))
        print('\t>> Number of labels: {0}'.format(L_S))
        print('\t>> Label cardinality: {0:.4f}'.format(LCard_S))
        print('\t>> Label density: {0:.4f}'.format(LDen_S))
        print('\t>> Distinct label sets: {0}'.format(DL_S))
        print('\t>> Proportion of distinct label sets: {0:.4f}'.format(PDL_S))
        tail = np.sum(y, axis=0)
        tail = tail[np.nonzero(tail)[0]]
        tail[tail <= num_tails] = 1
        tail[tail > num_tails] = 0
        print('\t>> Number of tail labels of size {0}: {1}'.format(
            num_tails, int(tail.sum())))
        tail[tail == 0] = -1
        tail[tail == 1] = 0
        print('\t>> Number of dominant labels of size {0}: {1}'.format(
            num_tails + 1, int(np.count_nonzero(tail))))

    tail = np.sum(y, axis=0)
    ntail_idx = np.nonzero(tail)[0]
    tail = tail[ntail_idx]
    tail_idx = np.argsort(tail)
    tail = tail[tail_idx]

    y = y[selected_examples]
    tail_selected = np.sum(y, axis=0)
    tail_selected = tail_selected[ntail_idx]
    tail_selected = tail_selected[tail_idx]

    L_S = int(np.sum(y))
    LCard_S = L_S / y.shape[0]
    LDen_S = LCard_S / L_S
    DL_S = np.nonzero(np.sum(y, axis=0))[0].size
    PDL_S = DL_S / y.shape[0]
    print('## SELECTED ({0}) DATA PROPERTIES for {1}...'.format(selected_name, data_name))
    print('\t>> Number of examples: {0}'.format(y.shape[0]))
    print('\t>> Number of labels: {0}'.format(L_S))
    print('\t>> Label cardinality: {0:.4f}'.format(LCard_S))
    print('\t>> Label density: {0:.4f}'.format(LDen_S))
    print('\t>> Distinct label sets: {0}'.format(DL_S))
    print('\t>> Proportion of distinct label sets: {0:.4f}'.format(PDL_S))

    # print
    df_comp = pd.DataFrame(
        {"Class": np.arange(1, 1 + tail.shape[0]), "All": tail, "Selected": tail_selected})
    df_comp = df_comp.melt(['Class'], var_name='Split', value_name='Sum')

    # Prob bar
    alt.themes.enable('none')
    chart = alt.Chart(df_comp).properties(width=600, height=350).mark_bar(color="grey").encode(
        x=alt.X('Class:O', title="Class ID", sort='ascending'),
        y=alt.Y('Sum:Q', title="Number of Examples"),
        color=alt.Color('Split:N', scale=alt.Scale(range=['grey', 'black'])),
    ).configure_header(
        titleFontSize=20,
        labelFontSize=15
    ).configure_axis(
        labelLimit=500,
        titleFontSize=20,
        labelFontSize=12,
        labelPadding=5,
    ).configure_axisY(
        grid=False
    ).configure_legend(
        strokeColor='gray',
        fillColor='white',
        padding=10,
        cornerRadius=10).resolve_scale(x='independent')
    # save
    chart.save(os.path.join(rspath, file_name.lower() + '_tails' + '.html'))


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
