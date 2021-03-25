import random

import altair as alt
import pandas as pd
from scipy.sparse import issparse, eye
from scipy.stats import entropy

from src.metrics.mlmetrics import *

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


def data_properties(y, selected_examples, num_tails: int = 2, dataset_name="test", model_name: str = "model",
                    rspath: str = ".", display_dataframe: bool = False, display_figure: bool = False):
    save_name = model_name.lower() + "_" + dataset_name.lower()
    args_list = []
    hold_list = [['Number of examples', 'Number of labels', 'Label cardinality',
                  'Label density', 'Distinct labels', 'Distinct label sets',
                  'Frequency of distinct label sets',
                  'Mean imbalance ratio intra-class for all labels',
                  'Mean imbalance ratio inter-class for all labels',
                  'Mean imbalance ratio labelsets for all labels',
                  'Labels having less than or equal to {0} examples'.format(num_tails),
                  'Labels having more than {0} examples'.format(num_tails + 1),
                  'KL difference between complete and data partition']]

    # 1. Compute properties of complete data
    L_S = int(np.sum(y))
    LCard_S = cardinality(y)
    LDen_S = density(y)
    DL_S = distinct_labels(y)
    DLS_S = distinct_labelsets(y)
    PDL_S = propportion_distinct_labelsets(y)
    IR_intra = mean_ir_intra_class(y)
    IR_inter = mean_ir_inter_class(y)
    IR_labelset = mean_ir_labelset(y)

    # 1.1. Compute tail labels properties for the complete data
    tail = np.sum(y.toarray(), axis=0)
    tail = tail[np.nonzero(tail)[0]]
    tail[tail <= num_tails] = 1
    tail[tail > num_tails] = 0
    tail_sum = int(tail.sum())
    tail[tail == 0] = -1
    tail[tail == 1] = 0
    tail_count = int(np.count_nonzero(tail))

    args_list.append('## PROPERTIES for {0}...'.format(dataset_name))
    args_list.append('\t>> Number of examples: {0}'.format(y.shape[0]))
    args_list.append('\t>> Number of labels: {0}'.format(L_S))
    args_list.append('\t>> Label cardinality: {0:.6f}'.format(LCard_S))
    args_list.append('\t>> Label density: {0:.6f}'.format(LDen_S))
    args_list.append('\t>> Distinct labels: {0}'.format(DL_S))
    args_list.append('\t>> Distinct label sets: {0}'.format(DLS_S))
    args_list.append('\t>> Frequency of distinct label sets: {0:.6f}'.format(PDL_S))
    args_list.append('\t>> Mean imbalance ratio intra-class for all labels: {0:.6f}'.format(IR_intra))
    args_list.append('\t>> Mean imbalance ratio inter-class for all labels: {0:.6f}'.format(IR_inter))
    args_list.append('\t>> Mean imbalance ratio labelsets for all labels: {0:.6f}'.format(IR_labelset))
    args_list.append('\t>> Labels having less than or equal to {0} examples: {1}'.format(num_tails, tail_sum))
    args_list.append('\t>> Labels having more than {0} examples: {1}'.format(num_tails + 1, tail_count))

    hold_list.append([y.shape[0], L_S, LCard_S, LDen_S, DL_S, DLS_S, PDL_S, IR_intra,
                      IR_inter, IR_labelset, tail_sum, tail_count, 0])

    # 2. Compute properties of complete data
    distr_y = np.sum(y.toarray(), axis=0)
    ntail_idx = np.nonzero(distr_y)[0]
    tail = distr_y[ntail_idx]
    tail_idx = np.argsort(tail)
    tail = tail[tail_idx]
    distr_y = distr_y / np.sum(y.toarray())

    # 3. Iteratively calculate properties of training and test data, respectively
    split_set_name = ["training set", "test set"]
    tail_selected_list = []
    for idx in range(len(selected_examples)):
        y_tmp = y[selected_examples[idx]]
        distr_y_selected = np.sum(y_tmp.toarray(), axis=0)
        tail_selected = distr_y_selected[ntail_idx]
        tail_selected = tail_selected[tail_idx]
        distr_y_selected = distr_y_selected / np.sum(y.toarray())
        tail_selected_list.append(tail_selected)

        L_S_selected = int(y_tmp.sum())
        LCard_S_selected = cardinality(y_tmp)
        LDen_S_selected = density(y_tmp)
        DL_S_selected = distinct_labels(y_tmp)
        DLS_S_selected = distinct_labelsets(y_tmp)
        PDL_S_selected = propportion_distinct_labelsets(y)
        IR_intra_selected = mean_ir_intra_class(y_tmp)
        IR_inter_selected = mean_ir_inter_class(y_tmp)
        IR_labelset_selected = mean_ir_labelset(y_tmp)
        kl = entropy(pk=distr_y_selected, qk=distr_y)

        # 3.1. Compute tail labels properties for the complete data
        temp = np.sum(y_tmp.toarray(), axis=0)
        temp = temp[np.nonzero(temp)[0]]
        temp[temp <= num_tails] = 1
        temp[temp > num_tails] = 0
        temp_sum = int(temp.sum())
        temp[temp == 0] = -1
        temp[temp == 1] = 0
        temp_count = int(np.count_nonzero(temp))

        args_list.append('## PROPERTIES for {0} ({1})...'.format(dataset_name, split_set_name[idx]))
        args_list.append('\t>> Number of examples: {0}'.format(y_tmp.shape[0]))
        args_list.append('\t>> Number of labels: {0}'.format(L_S_selected))
        args_list.append('\t>> Label cardinality: {0:.6f}'.format(LCard_S_selected))
        args_list.append('\t>> Label density: {0:.6f}'.format(LDen_S_selected))
        args_list.append('\t>> Distinct labels: {0}'.format(DL_S_selected))
        args_list.append('\t>> Distinct label sets: {0}'.format(DLS_S_selected))
        args_list.append('\t>> Frequency of distinct label set: {0:.6f}'.format(PDL_S_selected))
        args_list.append('\t>> Mean imbalance ratio intra-class for all labels: {0:.6f}'.format(IR_intra_selected))
        args_list.append('\t>> Mean imbalance ratio inter-class for all labels: {0:.6f}'.format(IR_inter_selected))
        args_list.append('\t>> Mean imbalance ratio labelsets for all labels: {0:.6f}'.format(IR_labelset_selected))
        args_list.append('\t>> Labels having less than or equal to {0} examples: {1}'.format(num_tails, temp_sum))
        args_list.append('\t>> Labels having more than {0} examples: {1}'.format(num_tails + 1, temp_count))
        args_list.append('\t>> KL difference between complete '
                         'and data partition: {0:.6f}'.format(kl))
        hold_list.append([y_tmp.shape[0], L_S_selected, LCard_S_selected, LDen_S_selected,
                          DL_S_selected, DLS_S_selected, PDL_S_selected, IR_intra_selected,
                          IR_inter_selected, IR_labelset_selected, temp_sum, temp_count, kl])

    if not display_dataframe:
        for args in args_list:
            print(args)

    # Plotting utilities
    df_comp = pd.DataFrame({"Label": np.arange(1, 1 + tail.shape[0]), "Complete": tail,
                            "Train": tail_selected_list[0], "Test": tail_selected_list[1]})
    df_comp = df_comp.melt(['Label'], var_name='Dataset', value_name='Sum')

    # Bar plot
    alt.themes.enable('none')
    chart = alt.Chart(df_comp).properties(width=600, height=350).mark_bar(color="grey").encode(
        x=alt.X('Label:O', title="Label ID", sort='ascending'),
        y=alt.Y('Sum:Q', title="Number of Examples", stack=None),
        color=alt.Color('Dataset:N', scale=alt.Scale(range=['red', 'black', 'blue'])),
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
    chart.save(os.path.join(rspath, save_name + '.html'))

    df = pd.DataFrame(hold_list).T
    df.columns = ['Properties for {0}'.format(dataset_name), 'Complete set', 'Training set', 'Test set']
    df.to_csv(path_or_buf=os.path.join(rspath, save_name + ".tsv"), sep='\t')
    if display_dataframe and display_figure:
        return df, chart
    elif display_dataframe and not display_figure:
        return df
    elif not display_dataframe and display_figure:
        return chart


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
