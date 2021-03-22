import os
import random

import altair as alt
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, issparse, eye
from scipy.stats import entropy

###********************    Path and datasets arguments    ********************###
DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split(os.sep)
REPO_PATH = os.sep.join(REPO_PATH[:-3])

DATASET_PATH = os.path.join(REPO_PATH, 'data')
RESULT_PATH = os.path.join(REPO_PATH, 'result')
LOG_PATH = os.path.join(REPO_PATH, 'log')

DATASET = ['bibtex', 'birds', 'Corel5k', 'delicious', 'emotions',
           'enron', 'genbase', 'mediamill', 'medical', 'rcv1subset1',
           'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'rcv1subset5',
           'scene', 'tmc2007_500', 'yeast']


###********************          Utilty functions         ********************###

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


def custom_shuffle(num_examples):
    idx = list(range(num_examples))
    random.shuffle(idx)
    return idx


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    temp = e_x / e_x.sum()
    return temp.tolist()


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


def data_properties(y, selected_examples, num_tails: int = 2, display_full_properties=True, dataset_name="test",
                    model_name: str = "model", split_set_name: str = "training set", rspath: str = ".",
                    mode: str = "w"):
    args_list = []
    if display_full_properties:
        L_S = int(np.sum(y))
        LCard_S = L_S / y.shape[0]
        LDen_S = LCard_S / L_S
        DL_S = np.nonzero(np.sum(y, axis=0))[0].size
        PDL_S = DL_S / y.shape[0]
        tail = np.sum(y, axis=0)
        tail = tail[np.nonzero(tail)[0]]
        tail[tail <= num_tails] = 1
        tail[tail > num_tails] = 0
        tail_sum = int(tail.sum())
        tail[tail == 0] = -1
        tail[tail == 1] = 0
        tail_count = int(np.count_nonzero(tail))

        args_list.append('## DATA PROPERTIES for {0}...'.format(dataset_name))
        args_list.append('\t>> Number of examples: {0}'.format(y.shape[0]))
        args_list.append('\t>> Number of labels: {0}'.format(L_S))
        args_list.append('\t>> Label cardinality: {0:.6f}'.format(LCard_S))
        args_list.append('\t>> Label density: {0:.6f}'.format(LDen_S))
        args_list.append('\t>> Distinct label sets: {0}'.format(DL_S))
        args_list.append('\t>> Proportion of distinct label sets: {0:.6f}'.format(PDL_S))
        args_list.append('\t>> Number of tail labels of size {0}: {1}'.format(num_tails, tail_sum))
        args_list.append('\t>> Number of dominant labels of size {0}: {1}'.format(num_tails + 1, tail_count))

    distr_y = np.sum(y, axis=0)
    ntail_idx = np.nonzero(distr_y)[0]
    tail = distr_y[ntail_idx]
    tail_idx = np.argsort(tail)
    tail = tail[tail_idx]
    distr_y = distr_y / np.sum(y)

    y = y[selected_examples]
    distr_y_selected = np.sum(y, axis=0)
    tail_selected = distr_y_selected[ntail_idx]
    tail_selected = tail_selected[tail_idx]
    distr_y_selected = distr_y_selected / np.sum(y)

    L_S_selected = int(np.sum(y))
    LCard_S_selected = L_S_selected / y.shape[0]
    LDen_S_selected = LCard_S_selected / L_S_selected
    DL_S_selected = np.nonzero(np.sum(y, axis=0))[0].size
    PDL_S_selected = DL_S_selected / y.shape[0]
    kl = entropy(pk=distr_y_selected, qk=distr_y)

    args_list.append('## SELECTED ({0} set) DATA PROPERTIES for {1}...'.format(split_set_name, dataset_name))
    args_list.append('\t>> Number of examples: {0}'.format(y.shape[0]))
    args_list.append('\t>> Number of labels: {0}'.format(L_S_selected))
    args_list.append('\t>> Label cardinality: {0:.6f}'.format(LCard_S_selected))
    args_list.append('\t>> Label density: {0:.6f}'.format(LDen_S_selected))
    args_list.append('\t>> Distinct label sets: {0}'.format(DL_S_selected))
    args_list.append('\t>> Proportion of distinct label sets: {0:.6f}'.format(PDL_S_selected))
    args_list.append(
        '\t>> KL difference between two full and selected examples labels distributions: {0:.6f}'.format(kl))

    save_name = model_name.lower() + "_" + dataset_name.lower()
    temp = os.path.join(rspath, save_name + "_properties.txt")
    with open(file=temp, mode=mode) as fout:
        for args in args_list:
            print(args)
            fout.write(args + "\n")

    # Plotting utilities
    df_comp = pd.DataFrame({"Class": np.arange(1, 1 + tail.shape[0]), "All": tail, "Selected": tail_selected})
    df_comp = df_comp.melt(['Class'], var_name='Split', value_name='Sum')

    # Bar plot
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
    chart.save(os.path.join(rspath, save_name + "_" + split_set_name.lower() + '_tails' + '.html'))


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
