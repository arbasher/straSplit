import os.path

DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split(os.sep)
REPO_PATH = os.sep.join(REPO_PATH[:-2])

DATASET_PATH = os.path.join(REPO_PATH, 'data')
LOG_PATH = os.path.join(REPO_PATH, 'log')
RESULT_PATH = os.path.join(REPO_PATH, 'result')
MODEL_PATH = os.path.join(REPO_PATH, 'model')