import argparse
import json
import os
import sys
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def get_args():
    parser = argparse.ArgumentParser(description='Template')
    parser.add_argument('--exp', type=str, required=True, help='Path to experiment data')
    parser.add_argument('--orientation', type=str, required=True, help='Sampling orientation',
                        choices=['horizontal', 'vertical'])
    return parser.parse_args()


def main(args):
    R_values = [2, 3, 6, 8, 12]
    corr_color = {0: '#ffd700',
                  3: '#fa8775'}
    step_color = {25: '#ea5f94',
                  37: '#cd34b5',
                  128: '#2e3e5b'}
    LD_color = {3990: '#9d02d7',
                275: '#0000ff',
                135: '#b44b06'}

    ## SSIM METRIC
    logging.info("--- SSIM ---")
    plt.figure(figsize=(7, 5), dpi=300)
    plt.xlabel("R")
    plt.ylabel("Masked SSIM")
    plt.xticks(R_values)

    # ddim 32 steps
    for corr in [0, 3]:
        means = []
        variances = []
        for R in R_values:
            try:
                f = open(os.path.join(args.exp, f'ddim_32_R={R}_{args.orientation}_corr={corr}/stats.json'))
                stats = json.load(f)
                mean = np.array(stats['ssim']).mean()
                conf_interval = mean_confidence_interval(stats['ssim'])[1]
                var = np.array(conf_interval)  # np.array(stats['ssim']).var()
                means.append(mean)
                variances.append(var)
            except:
                print(f'ddim_32_R={R}_{args.orientation}_corr={corr}: Not found!')

        if not means:
            continue
        means = np.array(means)
        variances = np.array(variances)

        plt.ylim((0.81, 1.01))
        plt.yticks([0.85, 0.9, 0.95, 1.00])
        plt.plot(R_values, means, '--o', color=corr_color[corr], linewidth=1.2,
                 label='32 steps (SBIM)' if corr == 0 else '128 steps (SBIM \w corr=3)')
        plt.fill_between(R_values, means - variances, means + variances, color='b', alpha=0.05)
        logging.info(f"ddim {32 * (1 + corr)} steps \w corr={corr}: {means}")

    # ddim 25 and 37 steps
    for steps in [25, 37, 128]:
        means = []
        variances = []
        for R in R_values:
            try:
                f = open(os.path.join(args.exp, f'ddim_{steps}_R={R}_{args.orientation}_corr=0/stats.json'))
                stats = json.load(f)
                mean = np.array(stats['ssim']).mean()
                conf_interval = mean_confidence_interval(stats['ssim'])[1]
                var = np.array(conf_interval)  # np.array(stats['ssim']).var()
                means.append(mean)
                variances.append(var)
            except:
                print(f'ddim_{steps}_R={R}_{args.orientation}_corr=0: Not found!')

        if not means:
            continue
        means = np.array(means)
        variances = np.array(variances)

        plt.ylim((0.81, 1.01))
        plt.yticks([0.85, 0.9, 0.95, 1.00])
        plt.plot(R_values, means, '--o', color=step_color[steps], linewidth=1.2,
                 label=f'{steps} steps (SBIM)')
        plt.fill_between(R_values, means - variances, means + variances, color='b', alpha=0.05)
        logging.info(f'ddim {steps} steps: {means}')

    # LD 3990 and 275 steps
    for steps in [3990, 135]:
        means = []
        variances = []
        for R in R_values:
            try:
                f = open(os.path.join(args.exp, f'LD_{steps}_R={R}_{args.orientation}_corr=0/stats.json'))
                stats = json.load(f)
                mean = np.array(stats['ssim']).mean()
                conf_interval = mean_confidence_interval(stats['ssim'])[1]
                var = np.array(conf_interval)  # np.array(stats['ssim']).var()
                means.append(mean)
                variances.append(var)
            except:
                print(f'LD_{steps}_R={R}_{args.orientation}_corr=0: Not found!')

        if not means:
            continue
        means = np.array(means)
        variances = np.array(variances)

        plt.ylim((0.81, 1.01))
        plt.yticks([0.85, 0.9, 0.95, 1.00])
        plt.plot(R_values, means, '--o', color=LD_color[steps], linewidth=1.2,
                 label=f'{steps} steps (LD)')
        plt.fill_between(R_values, means - variances, means + variances, color='b', alpha=0.05)
        logging.info(f'LD {steps} steps: {means}')

    plt.legend()
    plt.grid(linestyle='--', linewidth=0.4)

    output_file = os.path.join(args.exp, f'SSIM_{args.orientation}')
    plt.savefig(output_file, dpi=300)

    ## PSNR metric
    logging.info("--- PSNR ---")
    plt.figure(figsize=(7, 5), dpi=300)
    plt.xlabel("R")
    plt.ylabel("Masked PSNR")
    plt.xticks(R_values)

    # ddim 32 steps
    for corr in [0, 3]:
        means = []
        variances = []
        for R in R_values:
            try:
                f = open(os.path.join(args.exp, f'ddim_32_R={R}_{args.orientation}_corr={corr}/stats.json'))
                stats = json.load(f)
                mean = np.array(stats['psnr']).mean()
                conf_interval = mean_confidence_interval(stats['psnr'])[1]
                var = np.array(conf_interval)  # np.array(stats['ssim']).var()
                means.append(mean)
                variances.append(var)
            except:
                print(f'ddim_32_R={R}_{args.orientation}_corr={corr}: Not found!')

        if not means:
            continue
        means = np.array(means)
        variances = np.array(variances)

        plt.ylim((24, 50))
        plt.yticks([25, 30, 35, 40, 45, 50])
        plt.plot(R_values, means, '--o', color=corr_color[corr], linewidth=1.2,
                 label='32 steps (SBIM)' if corr == 0 else '128 steps (SBIM \w corr=3)')
        plt.fill_between(R_values, means - variances, means + variances, color='b', alpha=0.05)
        logging.info(f'ddim 32 steps corr={corr}: {means}')

    # ddim 25 and 37 steps
    for steps in [25, 37]:
        means = []
        variances = []
        for R in R_values:
            try:
                f = open(os.path.join(args.exp, f'ddim_{steps}_R={R}_{args.orientation}_corr=0/stats.json'))
                stats = json.load(f)
                mean = np.array(stats['psnr']).mean()
                conf_interval = mean_confidence_interval(stats['psnr'])[1]
                var = np.array(conf_interval)  # np.array(stats['ssim']).var()
                means.append(mean)
                variances.append(var)
            except:
                print(f'ddim_{steps}_R={R}_{args.orientation}_corr=0: Not found!')

        if not means:
            continue
        means = np.array(means)
        variances = np.array(variances)

        plt.ylim((24, 50))
        plt.yticks([25, 30, 35, 40, 45, 50])
        plt.plot(R_values, means, '--o', color=step_color[steps], linewidth=1.2,
                 label=f'{steps} steps (SBIM)')
        plt.fill_between(R_values, means - variances, means + variances, color='b', alpha=0.05)
        logging.info(f'ddim {steps} steps: {means}')

    # LD 3990 and 275 steps
    for steps in [3990, 135]:
        means = []
        variances = []
        for R in R_values:
            try:
                f = open(os.path.join(args.exp, f'LD_{steps}_R={R}_{args.orientation}_corr=0/stats.json'))
                stats = json.load(f)
                mean = np.array(stats['psnr']).mean()
                conf_interval = mean_confidence_interval(stats['psnr'])[1]
                var = np.array(conf_interval)  # np.array(stats['ssim']).var()
                means.append(mean)
                variances.append(var)
            except:
                print(f'LD_{steps}_R={R}_{args.orientation}_corr=0: Not found!')

        if not means:
            continue
        means = np.array(means)
        variances = np.array(variances)

        plt.ylim((24, 50))
        plt.yticks([25, 30, 35, 40, 45, 50])
        plt.plot(R_values, means, '--o', color=LD_color[steps], linewidth=1.2,
                 label=f'{steps} steps (LD)')
        plt.fill_between(R_values, means - variances, means + variances, color='b', alpha=0.05)
        logging.info(f'LD {steps} steps: {means}')

    plt.legend()
    plt.grid(linestyle='--', linewidth=0.4)

    output_file = os.path.join(args.exp, f'PSNR_{args.orientation}')
    plt.savefig(output_file, dpi=300)

    return 0


if __name__ == '__main__':
    args = get_args()

    # logging
    handler = logging.FileHandler(os.path.join(args.exp, 'tables.txt'))
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(20)  # info

    sys.exit(main(args))
