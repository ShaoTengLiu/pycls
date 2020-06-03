#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train a classification model."""

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
from pycls.core.config import cfg

import argparse
from rich.console import Console
from rich.table import Column, Table

def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Test a trained classification model")
    parser.add_argument("--corruptions", default=['original', 'gaussian_noise', 'shot_noise', 'impulse_noise', 
                                             'defocus_blur', 'glass_blur', 'motion_blur', 
                                             'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 
                                             'contrast', 'elastic_transform', 'pixelate', 
                                             'jpeg_compression'], nargs='+')
    parser.add_argument("--levels", default=[5,4,3,2,1], nargs='+', type=int)
    parser.add_argument("--cfg") # just for the cfg conflict
    return parser.parse_args()

def main():

    config.load_cfg_fom_args("Train a classification model.")
    config.assert_and_infer_cfg()
    # cfg.freeze()

    args = parse_args()
    # Perform training
    results = dist.multi_proc_run(num_proc=cfg.NUM_GPUS, \
        fun=trainer.test_feedback_all_model(args.corruptions, args.levels))

    # plot the results table
    for index, level in enumerate(args.levels):
        console = Console()
        table = Table(show_header=True, header_style="cyan")
        print("[bold green]{}[/bold green]".format(str(level)))
        table.add_column('Model')
        for corruption in args.corruptions:
            table.add_column(corruption[:3])
        table.add_row(results[index])
        console.print(table)

if __name__ == "__main__":
    main()
