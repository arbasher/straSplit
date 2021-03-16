import os.path

DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split(os.sep)
REPO_PATH = os.sep.join(REPO_PATH[:-3])

DATASET_PATH = os.path.join(REPO_PATH, 'data')
RESULT_PATH = os.path.join(REPO_PATH, 'result')
LOG_PATH = os.path.join(REPO_PATH, 'log')
