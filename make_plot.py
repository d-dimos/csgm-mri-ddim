import argparse
import json
import os
import sys
import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

R_values = [2, 3, 6, 8, 12]
color_dict = {('ddim', 25, 0): '#5F4690',
              ('ddim', 32, 0): '#FF0000',
              ('ddim', 37, 0): '#38A6A5',
              ('ddim', 128, 0): '#0F8554',
              ('ddim', 32, 3): '#73AF48',
              ('ddim', 64, 3): '#000000',
              ('LD', 128, 0): '#E17C05',
              ('LD', 3990, 0): '#CC503E'}


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, h


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True, help='Path to experiment data')
    parser.add_argument('--orientation', type=str, required=True, help='Sampling orientation',
                        choices=['horizontal', 'vertical'])
    return parser.parse_args()


def configure_logging(exp_dir):
    handler = logging.FileHandler(os.path.join(exp_dir, 'tables.txt'))
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(20)


def fix(stats, args):
    return stats


def plot_metric(metric, args):
    logging.info(f"--- {metric} ---")
    plt.figure(figsize=(7, 5))
    plt.xlabel("R")
    plt.xticks(R_values)

    if metric == 'ssim':
        plt.ylabel(f"Masked SSIM")
        plt.ylim((0.81, 1.01))
        plt.yticks([0.85, 0.9, 0.95, 1.00])
    else:  # psnr
        plt.ylabel(f"Masked PSNR")
        plt.ylim((24, 50))
        plt.yticks([25, 30, 35, 40, 45, 50])

    for corr in [0, 3]:
        for sampler in ['ddim', 'LD']:
            for steps in [25, 32, 37, 64, 128, 3990]:
                means = []
                confs = []
                for R in R_values:
                    try:
                        f = open(os.path.join(args.exp,
                                              f'{sampler}_{steps}_R={R}_{args.orientation}_corr={corr}/stats.json'))
                        stats = fix(json.load(f), args)
                        mean = np.array(stats[f'{metric}']).mean()
                        conf_interval = mean_confidence_interval(stats[f'{metric}'])[1]
                        conf_interval = np.array(conf_interval)
                        means.append(mean)
                        confs.append(conf_interval)
                    except:
                        continue
                if not means:
                    continue

                means = np.array(means)
                confs = np.array(confs)

                if sampler == 'ddim':
                    label = f'{steps * (1 + corr)} steps (SBIM)' if corr == 0 else \
                            f'{steps * (1 + corr)} steps (SBIM \w corr={corr})'
                else:  # langevin
                    label = f'{steps * (1 + corr)} steps (Langevin)'

                plt.plot(R_values, means, '--o', linewidth=0.8, label=label,
                         color=color_dict[(sampler, steps, corr)])

                plt.fill_between(R_values, means - confs, means + confs, color='b', alpha=0.05)
                logging.info(f"{sampler}-steps {steps}-corr {corr}: {means}")

    plt.legend()
    plt.grid(linestyle='--', linewidth=0.4)
    output_file = os.path.join(args.exp, f'{metric}_{args.orientation}')
    plt.savefig(output_file, dpi=300)
    return


def main(args):
    configure_logging(args.exp)
    plot_metric('ssim', args)
    plot_metric('psnr', args)
    return 0


if __name__ == '__main__':
    sys.exit(main(get_args()))
