import os
import pickle as pkl

from scipy.sparse import lil_matrix
from skmultilearn.dataset import load_dataset, available_data_sets

from src.utility.file_path import DATASET_PATH

os.system('clear')

# Paths for the required files

print(">> Transform dataset to lil_matrix...")

temp = available_data_sets()
for ds in temp:
    if ds[1] == 'undivided':
        X_train, y_train, _, _ = load_dataset(ds[0], ds[1])
        X_file = os.path.join(DATASET_PATH, ds[0] + '_X.pkl')
        with open(file=X_file, mode="wb") as fout:
            pkl.dump(lil_matrix(X_train), fout)
        y_file = os.path.join(DATASET_PATH, ds[0] + '_y.pkl')
        with open(file=y_file, mode="wb") as fout:
            pkl.dump(lil_matrix(y_train), fout)
