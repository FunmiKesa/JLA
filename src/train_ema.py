from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import subprocess

import json
import torch
import torch.utils.data
from torchvision.transforms import transforms as T
from opts import opts
from models.model import create_model, load_model, save_model
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory

import torch.distributed as dist

import torch
import torch.backends.cudnn as cudnn

from utils.launch import launch
from trains.trainer import Trainer
from exp import get_exp

import random
import warnings
from loguru import logger

@logger.catch
def main(exp, opts):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    cudnn.benchmark = True

    trainer = Trainer(exp, opts)
    trainer.train()


if __name__ == "__main__":
    opts = opts().parse()
    exp = get_exp(opts.exp_file, "jla")
    # exp.merge(opts)

    if not opts.exp_id:
        opts.exp_id = exp.exp_name

    num_gpu = torch.cuda.device_count() if opts.devices is None else opts.devices
    assert num_gpu <= torch.cuda.device_count()

    launch(
        main,
        num_gpu,
        opts.num_machines,
        opts.machine_rank,
        backend=opts.dist_backend,
        dist_url=opts.dist_url,
        args=(exp, opts),
    )
