import os
import pickle as pkl
import sys
import traceback


def save_data(data, file_name: str, save_path: str, tag: str = '', mode: str = 'wb', w_string: bool = False,
              print_tag: bool = True):
    '''
    Save data into file
    :param mode:
    :param print_tag:
    :param data: the data file to be saved
    :param save_path: location of the file
    :param file_name: name of the file
    '''
    try:
        file = file_name
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        file_name = os.path.join(save_path, file_name)
        if print_tag:
            print('\t\t## Storing {0:s} into the file: {1:s}'.format(tag, file))
        with open(file=file_name, mode=mode) as fout:
            if not w_string:
                pkl.dump(data, fout)
            elif w_string:
                fout.write(data)
    except Exception as e:
        print('\t\t## The file {0:s} can not be saved'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e


def load_data(file_name, load_path, load_X=False, mode='rb', tag='data', print_tag=True):
    '''
    :param data:
    :param load_path: load file from a path
    :type file_name: string
    :param file_name:
    '''
    try:
        if print_tag:
            print('\t\t## Loading {0:s} from: {1:s}'.format(tag, file_name))
        file_name = os.path.join(load_path, file_name)
        with open(file_name, mode=mode) as f_in:
            if mode == "r":
                data = f_in.readlines()
            else:
                if load_X:
                    data = __load_sparse(f_in)
                else:
                    data = pkl.load(f_in)
            return data
    except Exception as e:
        print('\t\t## The file {0:s} can not be loaded or located'.format(file_name), file=sys.stderr)
        print(traceback.print_exc())
        raise e
