import os
import pickle as pkl


def entropy_stratification(X, y, split_size: float = 0.75):
    pass


def npsp_stratification(X, y, split_size: float = 0.75):
    pass


def varitaion_ratios_stratification(X, y, split_size: float = 0.75):
    pass


def active_learning_stratification(X, y, split_size: float = 0.75):
    pass


if __name__ == "__main__":
    DIRECTORY_PATH = os.getcwd()
    DIRECTORY_PATH = DIRECTORY_PATH.split(os.sep)
    DIRECTORY_PATH = os.sep.join(DIRECTORY_PATH[:-3])

    X_name = "Xmediamill_train.pkl"
    file_path = os.path.join(DIRECTORY_PATH, 'data', X_name)
    with open(file_path, mode="rb") as f_in:
        y = pkl.load(f_in)
    training_set, test_set = stratification(y=y, split_size=0.8, batch_size=1000, num_jobs=10)
    training_set, dev_set = stratification(y=y[training_set], split_size=0.8, batch_size=500, num_jobs=10)
    print("training set size: {0}".format(len(training_set)))
    print("validation set size: {0}".format(len(dev_set)))
    print("test set size: {0}".format(len(test_set)))
