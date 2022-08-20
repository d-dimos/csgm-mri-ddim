import os
import sys
import argparse
import yaml
import shutil
import logging

import torch
import numpy as np

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
    # logging
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(20)  # info

    # input arguments and configuration file
    args = get_args()
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    config = dict2namespace(config)
    config.device = args.device
    logging.info(f'Using device: {config.device}')

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

    # run the experiment
    if args.sampler == 'ddim':
        guided_DDIM(args, config, logger).sample()
    if args.sampler == 'LD':
        guided_LD(args, config).sample()

    return 0


if __name__ == '__main__':
    sys.exit(main())
