#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train a classification model."""

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.test_feedback as test_feedback
from pycls.core.config import cfg

def main():
    config.load_cfg_fom_args("Train a classification model.")
    config.assert_and_infer_cfg()
    cfg.freeze()

    # Perform training
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=test_feedback.train_model)

if __name__ == "__main__":
    main()
