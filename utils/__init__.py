import os
import glob
import argparse


def get_all_files(data_dir, pattern='*'):
    files = [x for x in glob.iglob(os.path.join(data_dir, pattern))]
    return sorted(files)


def dict2namespace(langevin_config):
    namespace = argparse.Namespace()
    for key, value in langevin_config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace
