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

from rich.console import Console
from rich.table import Column, Table


def main():

    corruptions, levels = config.load_cfg_fom_args("Train a classification model.")
    config.assert_and_infer_cfg()

    # Perform training
    results = dist.multi_proc_run(
                num_proc=cfg.NUM_GPUS, 
                fun=lambda: trainer.test_ftta_model(corruptions, levels))

    # plot the results table
    for index, level in enumerate(levels):
        console = Console()
        table = Table(show_header=True, header_style="cyan")
        table.add_column('level')
        for corruption in corruptions:
            table.add_column(corruption)
        res = list( map(lambda x: str(x), results[index]) )
        res = [str(level)] + res
        table.add_row(*res)
        console.print(table)


if __name__ == "__main__":
    main()
