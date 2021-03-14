import os
import shutil
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import lil_matrix
from sklearn.metrics import confusion_matrix, coverage_error
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import jaccard_score, hamming_loss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from utility.access_file import save_data

EPSILON = np.finfo(np.float).eps

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')


def create_remove_dir(folder_path):
    if os.path.isdir(folder_path):
        try:
            shutil.rmtree(path=folder_path)
        except OSError as e:
            print("\t\t## Cannot remove the directory: {0}".format(
                folder_path), file=sys.stderr)
            raise e
    try:
        os.mkdir(path=folder_path)
    except OSError as e:
        print("\t\t## Creation of the directory {0} failed...".format(
            folder_path), file=sys.stderr)
        raise e


def extract_labels(data_id, use_all_labels, num_sample, file_name, dataset_path):
    classlabels = list()
    class_labels_ids = dict()
    if use_all_labels:
        classlabels = [item for item, idx in data_id.items()]
        class_labels_ids = data_id
    else:
        file_name = file_name + '_' + str(num_sample) + '_mapping.txt'
        fileClasslabels = os.path.join(dataset_path, file_name)
        with open(fileClasslabels, 'r') as f_in:
            for tmp in f_in:
                try:
                    if not tmp.startswith('#'):
                        label = tmp.split('\t')[0]
                        classlabels.append(label)
                        class_labels_ids.update({label: data_id[label]})
                except IOError:
                    break
    return class_labels_ids, classlabels


###***************************   Stratified Partition Dataset   ***************************###

def stratified_dataset(data_id, f_object, num_sample, use_all_labels=False, train_size=0.8, val_size=0.2,
                       dataset_path=''):
    fName = f_object + '_' + str(num_sample)
    X = os.path.join(dataset_path, fName + '_X.pkl')
    y = os.path.join(dataset_path, fName + '_y.pkl')
    class_labels_ids = dict()
    labels = dict()
    classlabels = list()

    if use_all_labels:
        classlabels = [item for item, idx in data_id.items()]
        class_labels_ids = data_id

    with open(X, 'rb') as f_in:
        while True:
            try:
                data = pkl.load(f_in)
                if type(data) is tuple and len(data) == 10:
                    nTotalSamples = data[1]
                    nTotalComponents = data[3]
                    nTotalClassLabels = data[5]
                    nTotalEvidenceFeatures = data[7]
                    nTotalClassEvidenceFeatures = data[9]
                    break
            except IOError:
                break

    with open(y, 'rb') as f_in:
        while True:
            try:
                data = pkl.load(f_in)
                if type(data) is tuple:
                    y_true, sample_ids = data
                    for sidx in np.arange(y_true.shape[0]):
                        for label in y_true[sidx]:
                            if label not in labels:
                                labels.update({label: [sidx]})
                                if not use_all_labels:
                                    classlabels.append(label)
                                    class_labels_ids.update(
                                        {label: data_id[label]})
                            else:
                                labels[label].extend([sidx])
                    break
            except IOError:
                break

    train_samples = list()
    dev_samples = list()
    test_samples = list()
    for label in labels.items():
        trn = np.random.choice(a=label[1], size=int(
            np.ceil(train_size * len(label[1]))), replace=False)
        dev = np.random.choice(a=trn, size=int(
            np.ceil(val_size * len(trn))), replace=False)
        if len(dev) > 0:
            trn = [x for x in trn if x not in dev]
        tst = [i for i in label[1] if i not in trn and i not in dev]
        for sidx in trn:
            if sidx not in train_samples and sidx not in dev_samples and sidx not in test_samples:
                train_samples.append(sidx)
        for sidx in dev:
            if sidx not in train_samples and sidx not in dev_samples and sidx not in test_samples:
                dev_samples.append(sidx)
        for sidx in tst:
            if sidx not in train_samples and sidx not in dev_samples and sidx not in test_samples:
                test_samples.append(sidx)

    train_samples = np.unique(train_samples)
    dev_samples = np.unique(dev_samples)
    test_samples = np.unique(test_samples)

    # X training set and X label set
    file_desc = '# The training set is stored as X\n'
    X_train_file = fName + '_Xtrain' + '.pkl'
    save_data(data=file_desc, file_name=X_train_file,
              save_path=dataset_path, tag='X training samples', mode='w+b')
    save_data(data=('nTotalSamples', len(train_samples),
                    'num_components', nTotalComponents,
                    'nTotalClassLabels', nTotalClassLabels,
                    'nTotalEvidenceFeatures', nTotalEvidenceFeatures,
                    'nTotalClassEvidenceFeatures', nTotalClassEvidenceFeatures), file_name=X_train_file,
              save_path=dataset_path, mode='a+b', print_tag=False)
    file_desc = '# This file stores the labels of the training set as (X, ids)\n'
    ytrainFile = fName + '_ytrain' + '.pkl'
    save_data(data=file_desc, file_name=ytrainFile,
              save_path=dataset_path, tag='X training samples', mode='w+b')

    # X development set and X label set
    file_desc = '# The development set is stored as X\n'
    X_dev_file = fName + '_Xdev' + '.pkl'
    save_data(data=file_desc, file_name=X_dev_file,
              save_path=dataset_path, tag='X development samples', mode='w+b')
    save_data(data=('nTotalSamples', len(dev_samples),
                    'num_components', nTotalComponents,
                    'nTotalClassLabels', nTotalClassLabels,
                    'nTotalEvidenceFeatures', nTotalEvidenceFeatures,
                    'nTotalClassEvidenceFeatures', nTotalClassEvidenceFeatures), file_name=X_dev_file,
              save_path=dataset_path, mode='a+b', print_tag=False)
    file_desc = '# This file stores the labels of the development set as (X, ids)\n'
    ydevFile = fName + '_ydev' + '.pkl'
    save_data(data=file_desc, file_name=ydevFile,
              save_path=dataset_path, tag='X development samples', mode='w+b')

    # X test set and X label set
    file_desc = '# The test set is stored as X\n'
    XtestFile = fName + '_Xtest' + '.pkl'
    save_data(data=file_desc, file_name=XtestFile,
              save_path=dataset_path, tag='X test samples', mode='w+b')
    save_data(data=('nTotalSamples', len(test_samples),
                    'num_components', nTotalComponents,
                    'nTotalClassLabels', nTotalClassLabels,
                    'nTotalEvidenceFeatures', nTotalEvidenceFeatures,
                    'nTotalClassEvidenceFeatures', nTotalClassEvidenceFeatures), file_name=XtestFile,
              save_path=dataset_path, mode='a+b', print_tag=False)
    file_desc = '# This file stores the labels of the test set as (X, ids)\n'
    ytestFile = fName + '_ytest' + '.pkl'
    save_data(data=file_desc, file_name=ytestFile,
              save_path=dataset_path, tag='X test samples', mode='w+b')

    with open(X, 'rb') as f_in:
        sidx = 0
        while True:
            try:
                item = pkl.load(f_in)
                if type(item) is np.ndarray:
                    if sidx in train_samples:
                        save_data(data=item, file_name=X_train_file, save_path=dataset_path, mode='a+b',
                                  print_tag=False)
                        save_data(data=(y_true[sidx], sample_ids[sidx]), file_name=ytrainFile, save_path=dataset_path,
                                  mode='a+b', print_tag=False)
                    elif sidx in dev_samples:
                        save_data(data=item, file_name=X_dev_file,
                                  save_path=dataset_path, mode='a+b', print_tag=False)
                        save_data(data=(y_true[sidx], sample_ids[sidx]), file_name=ydevFile, save_path=dataset_path,
                                  mode='a+b', print_tag=False)
                    else:
                        save_data(data=item, file_name=XtestFile,
                                  save_path=dataset_path, mode='a+b', print_tag=False)
                        save_data(data=(y_true[sidx], sample_ids[sidx]), file_name=ytestFile, save_path=dataset_path,
                                  mode='a+b', print_tag=False)
                    sidx += 1
                if sidx == nTotalSamples:
                    break
            except IOError:
                break

    return [classlabels, class_labels_ids, nTotalComponents, nTotalClassLabels,
            nTotalEvidenceFeatures, nTotalClassEvidenceFeatures,
            X_train_file, ytrainFile, X_dev_file, ydevFile, XtestFile, ytestFile]


###***************************          Detailed Report         ***************************###


def detail_header_file(header=True):
    if header:
        desc = "{0}{1}\n".format("# ", "=" * 52)
        desc += "# Description of attributes in this file.\n"
        desc += "{0}{1}\n".format("# ", "=" * 52)
        desc += "# Sample-id: a unique identifier of the sample.\n"
        desc += "# Total Pathways: total set of pathways for the \n#\tassociated sample.\n"
        desc += "# Pathway Frame-id: a unique identifier of the \n#\tpathway (as in the PGDB).\n"
        desc += "# Pathway Score: a number from 0-1 indicating the \n#\tstrength of the evidence " \
                "supporting the \n#\tinference of this pathway, where 1.0 means \n#\tvery strong evidence.\n"
        desc += "# Pathway Abundance: the abundance of the pathway \n#\tgiven the abundance values of the " \
                "enzymes \n#\tfor this pathway in the annotation file.\n"
        desc += "{0}{1}\n\n".format("# ", "=" * 52)
    else:
        desc = "{0}\n".format("-" * 111)
        desc += " {1:10}{0}{2:15}{0}{3:40}{0}{4:15}{0}{5:18}\n".format(" | ", "Sample-id", "Total Pathways",
                                                                       "Pathway Frame-id", "Pathway Score",
                                                                       "Pathway Abundance")
        desc += "{0}\n".format("-" * 111)
    return desc


def list_header_file(header=True):
    if header:
        desc = "{0}{1}\n".format("# ", "=" * 47)
        desc += "# Description of attributes in this file.\n"
        desc += "{0}{1}\n".format("# ", "=" * 47)
        desc += "# Sample-id: a unique identifier of the sample.\n"
        desc += "# Total Pathways: total set of pathways for the \n#\tassociated sample.\n"
        desc += "# Pathway Frame-id: a unique identifier of the \n#\tpathway (as in the PGDB).\n"
        desc += "{0}{1}\n\n".format("# ", "=" * 47)
    else:
        desc = "{0}\n".format("-" * 60)
        desc += " {1:10}{0}{2:15}{0}{3:40}\n".format(
            " | ", "Sample-id", "Total Pathways", "Pathway Frame-id")
        desc += "{0}\n".format("-" * 60)
    return desc


def compute_abd_cov(X, labels_components, class_dict, component_dict=None, batch_idx=-1, total_progress=-1):
    '''
    Predict a list of pathways for a given set
        of features extracted from input set

    :param forTraining:
    :type X_test: list
    :param X_test: feature list generated by DataObject, shape array-like
    '''
    exp2labels_abun = np.zeros(shape=(X.shape[0], len(class_dict.keys())))
    exp2labels_cov = np.zeros(shape=(X.shape[0], len(class_dict.keys())))
    if component_dict:
        ec_mapped = np.zeros(shape=(X.shape[0], len(
            class_dict.keys()), len(component_dict.keys())))
    batch_idx = batch_idx * len(class_dict.keys())
    for class_idx, class_label in class_dict.items():
        ref_label = labels_components[class_idx, :].toarray()
        tmp = np.copy(X.toarray())
        tmp[:, np.where(ref_label == 0)[1]] = 0
        abd_ecs = np.divide(tmp, ref_label)
        np.nan_to_num(abd_ecs, copy=False)
        exp2labels_abun[:, class_idx] = np.sum(abd_ecs, axis=1)
        tmp[tmp > 0] = 1.
        tmp = np.multiply(tmp, ref_label)
        cov_ecs = np.divide(np.sum(tmp, axis=1), np.sum(ref_label))
        cov_ecs[cov_ecs == np.inf] = 0
        np.nan_to_num(cov_ecs, copy=False)
        exp2labels_cov[:, class_idx] = cov_ecs
        if component_dict:
            row, col = np.nonzero(tmp)
            ec_mapped[row, class_idx, col] = 1
        if total_progress > 0:
            current_progress = batch_idx + class_idx
            desc = '\t\t--> Building {0:.4f}%...'.format(
                ((current_progress / total_progress) * 100))
            print(desc, end="\r")
    if component_dict:
        return exp2labels_abun, exp2labels_cov, ec_mapped
    return exp2labels_abun, exp2labels_cov


def __synthesize_report(X, sample_ids, y_pred_score, y_pred, y_dict_ids, y_common_name, component_dict,
                        labels_components, save_path,
                        batch_idx, total_progress):
    desc = '\t\t--> Synthesizing pathway reports {0:.4f}%...'.format(
        ((batch_idx + 1) / total_progress * 100))
    print(desc, end="\r")
    file_detail_name = 'pathway_report.tsv'
    pathway_abun, pathway_cov, ec_mapped = compute_abd_cov(X=X, labels_components=labels_components,
                                                           class_dict=y_dict_ids,
                                                           component_dict=component_dict)
    batch_data_lst = list()
    y_pred = y_pred.toarray()
    if y_pred_score is not None:
        y_pred_score = y_pred_score.toarray()
    for sidx, item in enumerate(y_pred):
        lst_ptwy = [y_dict_ids[idx] for idx in np.nonzero(item)[0]]
        lst_ptwy_name = [y_common_name[idx] for idx in np.nonzero(item)[0]]
        lst_ptwy_idx = [idx for idx in np.nonzero(item)[0]]
        if len(lst_ptwy_idx) == 0:
            continue
        if y_pred_score is not None:
            label_prob = list(y_pred_score[sidx, np.array(lst_ptwy_idx)])
        label_prediction = list(item[np.array(lst_ptwy_idx)])
        label_abun = [pathway_abun[sidx, pidx] for pidx in lst_ptwy_idx]
        label_cov = [pathway_cov[sidx, pidx] for pidx in lst_ptwy_idx]
        ec2label = [', '.join([component_dict[i] for i in np.nonzero(ec_mapped[sidx, pidx])[0] if component_dict[i]])
                    for pidx in lst_ptwy_idx]
        if y_pred_score is not None:
            sample_data = np.c_[lst_ptwy, lst_ptwy_name, label_prob,
                                label_prediction, label_abun, label_cov, ec2label]
        else:
            sample_data = np.c_[lst_ptwy, lst_ptwy_name,
                                label_prediction, label_abun, label_cov, ec2label]
        # Delete the previous folder and recreate a new one
        sample_folder_path = os.path.join(save_path, str(sample_ids[sidx]))
        create_remove_dir(folder_path=sample_folder_path)
        df_sample_pathways = pd.DataFrame(sample_data)
        if y_pred_score is not None:
            df_sample_pathways.columns = [
                'FrameID', 'Name', 'Score', 'Predicted', 'Abundance', 'Coverage', 'MappedEC']
        else:
            df_sample_pathways.columns = [
                'FrameID', 'Name', 'Predicted', 'Abundance', 'Coverage', 'MappedEC']
        df_sample_pathways.to_csv(path_or_buf=os.path.join(
            sample_folder_path, file_detail_name), sep='\t')
        batch_data_lst.append(lst_ptwy)
    return batch_data_lst


def synthesize_report(X, sample_ids, y_pred, y_dict_ids, y_common_name, component_dict, labels_components,
                      y_pred_score=None,
                      batch_size=30, num_jobs=1, rsfolder="Results", rspath="../.", dspath="../.", file_name='labels'):
    if y_pred is None:
        raise Exception("Please provide two matrices as numpy matrix format: "
                        "(num_samples, num_labels), representing pathway scores "
                        "and the status of prediction as binary values.")

    num_samples = len(sample_ids)
    main_folder_path = os.path.join(rspath, rsfolder)
    list_batches = np.arange(start=0, stop=num_samples, step=batch_size)
    parallel = Parallel(n_jobs=num_jobs, verbose=0)

    # Delete the previous main folder and recreate a new one
    create_remove_dir(folder_path=main_folder_path)
    if y_pred_score is not None:
        results = parallel(delayed(__synthesize_report)(X[batch:batch + batch_size],
                                                        sample_ids[batch:batch +
                                                                         batch_size],
                                                        y_pred_score[batch:batch +
                                                                           batch_size],
                                                        y_pred[batch:batch +
                                                                     batch_size],
                                                        y_dict_ids, y_common_name, component_dict,
                                                        labels_components, main_folder_path, batch_idx,
                                                        len(list_batches))
                           for batch_idx, batch in enumerate(list_batches))
    else:
        results = parallel(delayed(__synthesize_report)(X[batch:batch + batch_size],
                                                        sample_ids[batch:batch +
                                                                         batch_size],
                                                        y_pred_score, y_pred[batch:batch +
                                                                                   batch_size],
                                                        y_dict_ids, y_common_name, component_dict,
                                                        labels_components, main_folder_path, batch_idx,
                                                        len(list_batches))
                           for batch_idx, batch in enumerate(list_batches))
    desc = '\t\t--> Synthesizing pathway reports {0:.4f}%...'.format(100)
    print(desc)
    y = list(zip(*results))
    y = [item for lst in y for item in lst]
    print(
        '\t\t--> Storing predictions (label) to: {0:s}'.format(file_name + '_labels.pkl'))
    save_data(data=y, file_name=file_name + '_labels.pkl', save_path=dspath, mode="wb",
              print_tag=False)
    y_dict_ids = dict((y_id, y_idx) for y_idx, y_id in y_dict_ids.items())
    y_csr = np.zeros((len(y), len(y_dict_ids.keys())))
    for idx, lst in enumerate(y):
        for item in lst:
            if item in y_dict_ids:
                y_csr[idx, y_dict_ids[item]] = 1
    print(
        '\t\t--> Storing predictions (label index) to: {0:s}'.format(file_name + '_y.pkl'))
    save_data(data=lil_matrix(y_csr), file_name=file_name + "_y.pkl", save_path=dspath, mode="wb",
              print_tag=False)


###***************************           Report Scores          ***************************###

def psp(y_prob, y_true, A=1, B=1, C=1, top_k=50):
    # propensity of all labels
    N_j = y_true
    labels_sum = np.sum(N_j, axis=0)
    g = 1 / (C * (labels_sum + B)) ** A
    psp_label = 1 / (g + 1)

    # compute psp@k
    psp = N_j / psp_label
    labels_idx = np.flip(np.argsort(y_prob))[:, :top_k]
    tmp = [psp[s_idx, labels_idx[s_idx]] for s_idx in np.arange(psp.shape[0])]
    psp = (1 / top_k) * np.sum(tmp, axis=1)
    return psp.mean()


def psndcg(y_prob, y_true, A=1, B=1, C=1, top_k=50):
    # propensity of all labels
    N_j = y_true
    labels_sum = np.sum(N_j, axis=0)
    g = 1 / (C * (labels_sum + B)) ** A
    psp_label = 1 / (g + 1)
    log_psp_label = np.log(psp_label + 1)
    psp_label = np.multiply(psp_label, log_psp_label)

    # compute psdcg@k
    psdcg = N_j / psp_label
    labels_idx = np.flip(np.argsort(y_prob))[:, :top_k]
    tmp = np.array([psdcg[s_idx, labels_idx[s_idx]]
                    for s_idx in np.arange(psdcg.shape[0])])
    psdcg = np.sum(tmp, axis=1)
    sum_y = np.sum(1 / (np.log(y_true + 1) + EPSILON), axis=1)
    psdcg = psdcg / sum_y
    return psdcg.mean()


def score(y_true, y_pred, item_lst, six_db=False, A=1, B=1, C=1, top_k=150, mode='a',
          file_name='results.txt', save_path=''):
    idx_lst = [1]
    if six_db:
        item_lst = ['AraCyc', 'EcoCyc', 'HumanCyc',
                    'LeishCyc', 'TrypanoCyc', 'YeastCyc']
        if y_true.shape[0] == 4:
            item_lst = ['AraCyc', 'EcoCyc', 'HumanCyc', 'YeastCyc']
        idx_lst = [idx for idx in np.arange(len(item_lst))]
    print('\t>> Scores are saved to {0:s}...'.format(str(file_name)))
    for i, idx in enumerate(idx_lst):
        y = y_true
        y_hat = y_pred
        if six_db:
            y = y_true[idx]
            y_hat = y_pred[idx]
            y = y.reshape((1, y.shape[0]))
            y_hat = np.reshape(y_hat, (1, len(y_hat)))
            save_data(data='*** Scores for {0:s}...\n'.format(str(item_lst[i])), file_name=file_name,
                      save_path=save_path, mode=mode, w_string=True, print_tag=False)
        else:
            save_data(data='*** Scores for {0:s}...\n'.format(item_lst[i]), file_name=file_name, save_path=save_path,
                      mode='w', w_string=True, print_tag=False)
        ce_samples = coverage_error(y, y_hat)
        save_data(data='\t\t1)- Coverage error score: {0:.4f}\n'.format(ce_samples), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)

        lrl_samples = label_ranking_loss(y, y_hat)
        save_data(data='\t\t2)- Ranking loss score: {0:.4f}\n'.format(lrl_samples), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)

        lrap_samples = label_ranking_average_precision_score(y, y_hat)
        save_data(data='\t\t3)- Label ranking average precision score: {0:.4f}\n'.format(lrap_samples),
                  file_name=file_name, save_path=save_path, mode=mode, w_string=True, print_tag=False)

        if not np.array_equal(y_pred, y_pred.astype(bool)):
            top_k = y_true.shape[1] if top_k > y_true.shape[1] else top_k
            psp_samples = psp(y_prob=y_hat, y_true=y,
                              A=A, B=B, C=C, top_k=top_k)
            save_data(data='\t\t4)- Propensity Scored Precision at {0}: {1:.4f}\n'.format(top_k, psp_samples),
                      file_name=file_name, save_path=save_path, mode=mode, w_string=True, print_tag=False)

            ndcg_samples = psndcg(y_prob=y_hat, y_true=y,
                                  A=A, B=B, C=C, top_k=top_k)
            save_data(data='\t\t5)- Propensity Scored nDCG at {0}: {1:.4f}\n'.format(top_k, ndcg_samples),
                      file_name=file_name, save_path=save_path, mode=mode, w_string=True, print_tag=False)
            continue

        hl_samples = hamming_loss(y, y_hat)
        save_data(data='\t\t4)- Hamming-Loss score: {0:.4f}\n'.format(hl_samples), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)

        pr_samples_average = precision_score(y, y_hat, average='samples')
        pr_samples_micro = precision_score(y, y_hat, average='micro')
        pr_samples_macro = precision_score(y, y_hat, average='macro')
        save_data(data='\t\t5)- Precision...\n', file_name=file_name, save_path=save_path, mode=mode, w_string=True,
                  print_tag=False)
        save_data(data='\t\t\t--> Average sample precision: {0:.4f}\n'.format(pr_samples_average), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> Micro precision: {0:.4f}\n'.format(pr_samples_micro), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> Macro precision: {0:.4f}\n'.format(pr_samples_macro), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)

        rc_samples_average = recall_score(y, y_hat, average='samples')
        rc_samples_micro = recall_score(y, y_hat, average='micro')
        rc_samples_macro = recall_score(y, y_hat, average='macro')
        save_data(data='\t\t6)- Recall...\n', file_name=file_name, save_path=save_path, mode=mode, w_string=True,
                  print_tag=False)
        save_data(data='\t\t\t--> Average sample recall: {0:.4f}\n'.format(rc_samples_average), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> Micro recall: {0:.4f}\n'.format(rc_samples_micro), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> Macro recall: {0:.4f}\n'.format(rc_samples_macro), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)

        f1_samples_average = f1_score(y, y_hat, average='samples')
        f1_samples_micro = f1_score(y, y_hat, average='micro')
        f1_samples_macro = f1_score(y, y_hat, average='macro')
        save_data(data='\t\t7)- F1-score...\n', file_name=file_name, save_path=save_path, mode=mode, w_string=True,
                  print_tag=False)
        save_data(data='\t\t\t--> Average sample f1-score: {0:.4f}\n'.format(f1_samples_average), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> Micro f1-score: {0:.4f}\n'.format(f1_samples_micro), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> Macro f1-score: {0:.4f}\n'.format(f1_samples_macro), file_name=file_name,
                  save_path=save_path, mode=mode, w_string=True, print_tag=False)

        js_score_samples = jaccard_score(y, y_hat, average='samples')
        js_score_micro = jaccard_score(y, y_hat, average='micro')
        js_score_macro = jaccard_score(y, y_hat, average='macro')
        js_score_weighted = jaccard_score(y, y_hat, average='weighted')
        save_data(data='\t\t8)- Jaccard score...\n', file_name=file_name, save_path=save_path, mode=mode,
                  w_string=True, print_tag=False)
        save_data(data='\t\t\t--> Jaccard score (samples): {0:.4f}\n'.format(js_score_samples),
                  file_name=file_name, save_path=save_path, mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> Jaccard score (micro): {0:.4f}\n'.format(js_score_micro),
                  file_name=file_name, save_path=save_path, mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> Jaccard score (macro): {0:.4f}\n'.format(js_score_macro),
                  file_name=file_name, save_path=save_path, mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> Jaccard score (weighted): {0:.4f}\n'.format(js_score_weighted),
                  file_name=file_name, save_path=save_path, mode=mode, w_string=True, print_tag=False)

        tn, fp, fn, tp = confusion_matrix(y.flatten(), y_hat.flatten()).ravel()
        save_data(data='\t\t9)- Confusion matrix...\n', file_name=file_name, save_path=save_path, mode=mode,
                  w_string=True, print_tag=False)
        save_data(data='\t\t\t--> True positive: {0}\n'.format(tp), file_name=file_name, save_path=save_path,
                  mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> True negative: {0}\n'.format(tn), file_name=file_name, save_path=save_path,
                  mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> False positive: {0}\n'.format(fp), file_name=file_name, save_path=save_path,
                  mode=mode, w_string=True, print_tag=False)
        save_data(data='\t\t\t--> False negative: {0}\n'.format(fn), file_name=file_name, save_path=save_path,
                  mode=mode, w_string=True, print_tag=False)
