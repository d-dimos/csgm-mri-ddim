import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Template')
    parser.add_argument('--masked', action='store_true', help='masked or not')
    parser.add_argument('--exp', type=str, required=True, help='Path to experiment data')
    parser.add_argument('--orientation', type=str, required=True, help='Sampling orientation',
                        choices=['horizontal', 'vertical'])
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    R_values = [3, 6, 8, 12]

    corr_color = {0: '#804000',
                  2: '#0039E6',
                  3: '#0343DF'}

    ## SSIM METRIC
    plt.figure(figsize=(7, 5), dpi=300)
    plt.xlabel("R")
    plt.ylabel("Masked SSIM" if args.masked else "SSIM")
    plt.xticks(R_values)

    for corr in [0, 2]:
        means = []
        variances = []
        for R in R_values:
            with open(os.path.join(args.exp, f'ddim_32_R={R}_{args.orientation}_corr={corr}/stats.json')) as f:
                stats = json.load(f)
                mean = np.array(stats['ssim']).mean()
                var = np.array(stats['ssim']).var()
                means.append(mean)
                variances.append(var)

        means = np.array(means)
        variances = np.array(variances)

        plt.ylim((0.3, 1.01))
        plt.yticks([0.8, 0.85, 0.9, 0.95, 1.00])
        plt.plot(R_values, means, '--o', color=corr_color[corr], linewidth=1.2,
                 label=f'{32 * (1 + corr)} steps (SBIM \w corr={corr})')
        plt.fill_between(R_values, means - variances, means + variances, color='b', alpha=0.2)

    # means = []
    # variances = []
    # for R in R_values:
    #     with open(os.path.join(args.exp, f'LD_3990_R={R}_{args.orientation}_corr=0/stats.json')) as f:
    #         stats = json.load(f)
    #         mean = np.array(stats['ssim']).mean()
    #         var = np.array(stats['ssim']).var()
    #         means.append(mean)
    #         variances.append(var)

    # means = np.array(means)
    # variances = np.array(variances)

    # plt.ylim((0.79, 1.01))
    # plt.yticks([0.8, 0.85, 0.9, 0.95, 1.00])
    # plt.plot(R_values, means, '--o', color=corr_color[3], linewidth=1.2,
    #             label=f'3990 steps (LD)')
    # plt.fill_between(R_values, means - variances, means + variances, color='b', alpha=0.2)

    plt.legend()
    plt.grid(linestyle='--', linewidth=0.4)

    output_file = os.path.join(args.exp, f'SSIM_{args.orientation}')
    plt.savefig(output_file, dpi=300)

    # # PSNR METRIC
    # plt.figure(figsize=(7, 5), dpi=300)
    # plt.xlabel("R")
    # plt.ylabel("Masked PSNR" if args.masked else "PSNR")
    # plt.xticks(R_values)

    # for steps in steps_values:
    #     means = []
    #     variances = []
    #     for R in R_values:
    #         with open(os.path.join(args.exp, f'{args.sampler}_{steps}_R={R}_{args.orientation}/stats.json')) as f:
    #             stats = json.load(f)
    #             mean = np.array(stats['psnr']).mean()
    #             var = np.array(stats['psnr']).var()
    #             means.append(mean)
    #             variances.append(var)

    #     means = np.array(means)
    #     variances = np.array(variances)

    #     plt.ylim((19, 46))
    #     plt.yticks([20, 25, 30, 35, 40, 45])
    #     plt.plot(R_values, means, '--o', color=step_color[steps], linewidth=1.2,
    #              label=f'{steps} steps ({args.sampler})')
    #     plt.fill_between(R_values, means - variances, means + variances, color='b', alpha=0.2)

    # plt.legend()
    # plt.grid(linestyle='--', linewidth=0.4)

    # output_file = os.path.join(args.exp, f'PSNR_{args.sampler}_{args.orientation}')
    # plt.savefig(output_file, dpi=300)

    return 0


if __name__ == '__main__':
    sys.exit(main())
