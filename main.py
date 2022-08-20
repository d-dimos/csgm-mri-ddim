import argparse
import copy
import logging
import os
import shutil
import sys

import numpy as np
import torch
import yaml

from samplers import guided_DDIM, guided_LD
from utils import dict2namespace


def get_args():
    parser = argparse.ArgumentParser(description='Template')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--maps_dir', type=str, required=True, help='Path to sensitivity maps directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained checkpoint')
    parser.add_argument('--sampler', type=str, required=True, help='Sampler type', choices=['ddim', 'LD'])
    parser.add_argument('--steps', type=int, required=True, help='Sampling steps')
    parser.add_argument('--exp', type=str, default='exp', help='Path to experiment logs')
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args


def main():

    # input arguments and configuration file
    args = get_args()
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    config = dict2namespace(config)
    config.device = args.device

    # experiment dir
    exp_name = args.sampler + '_' + str(args.steps) + '_R=' + str(config.R) + '_' + 'or=' + config.orientation
    args.log_path = os.path.join(args.exp, exp_name)
    if os.path.exists(args.log_path):
        response = input("Folder already exists. Overwrite? (Y/N)")
        if response.upper() == 'Y':
            shutil.rmtree(args.log_path)
            os.makedirs(args.log_path)
        else:
            print("Experiment exists!")
            sys.exit(0)
    else:
        os.makedirs(args.log_path)

    # set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # logging
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log_path, 'stdout.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(20)  # info

    logging.info(f'Device: {config.device}')
    logging.info(f'Anatomy: {config.anatomy}')
    logging.info(f'Batch Size: {config.batch_size}')
    logging.info(f'Image Size: {config.image_size}')
    logging.info(f'Orientation: {config.orientation}')
    logging.info(f'Pattern: {config.pattern}')
    config_dict = copy.copy(vars(config.model))
    logging.info(f'Model Info:\n{yaml.dump(config_dict, default_flow_style=False)}')

    # run the experiment
    if args.sampler == 'ddim':
        guided_DDIM(args, config, logger).sample()
    if args.sampler == 'LD':
        guided_LD(args, config).sample()

    return 0


if __name__ == '__main__':
    sys.exit(main())
