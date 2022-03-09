#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import json
import os
import torch
import torch.nn as nn
from datasets.dataset_factory import get_dataset
from models.model import create_model, load_model, save_model

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
from torchvision.transforms import transforms as T

import os
import random

from lib.exp.base_exp import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 1
        self.ltrb = True
        self.reid_dim = 128
        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        self.num_workers = 4
        self.input_size = (640, 640)
        self.random_size = None # (14, 26)
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"

        # --------------- transform config ----------------- #
        self.degrees = 10.0
        self.translate = 0.1
        self.scale = (0.1, 2)
        self.mscale = (0.8, 1.6)
        self.shear = 2.0
        self.perspective = 0.0
        self.enable_mixup = True

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 30
        self.warmup_lr = 0
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "cos"
        self.no_aug_epochs = 30
        self.min_lr_ratio = 0.05
        self.ema = True

        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.print_interval = 10
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[
            1].split(".")[0]

        # -----------------  testing config ------------------ #
        self.test_size = (640, 640)
        self.test_conf = 0.001
        self.nmsthre = 0.65

    def get_model(self):
        model = create_model(self.arch, self.heads, self.head_conv)

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        

        Dataset = get_dataset(self.dataset, self.task)
        val_dataset = Dataset(self.opts, dataset_root, valset_paths,
                        (1088, 608), augment=False, transforms=transforms)
        f = open(self.data_cfg)
        data_config = json.load(f)
        trainset_paths = data_config['train']
        dataset_root = data_config['root']
        f.close()
        transforms = T.Compose([T.ToTensor()])
        dataset = Dataset(self, dataset_root, trainset_paths,
                        (1088, 608), augment=True, transforms=transforms)
        batch_size = self.batch_size
        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset)
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=self.num_workers, pin_memory=True, sampler=train_sampler, drop_last=False)

        self.dataset = dataset

        # if is_distributed:

        # sampler = InfiniteSampler(
        #     len(self.dataset), seed=self.seed if self.seed else 0)

        # batch_sampler = YoloBatchSampler(
        #     sampler=sampler,
        #     batch_size=batch_size,
        #     drop_last=False,
        #     input_dimension=self.input_size,
        #     mosaic=not no_aug,
        # )

        # dataloader_kwargs = {
        #     "num_workers": self.num_workers, "pin_memory": True}
        # dataloader_kwargs["batch_sampler"] = batch_sampler
        # train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = data_loader.change_input_dim(
            multiple=(tensor[0].item(), tensor[1].item()), random_range=None
        )
        return input_size

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, is_distributed, testdev=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            data_dir=None,
            json_file=self.val_ann if not testdev else "image_info_test-dev2017.json",
            name="val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(
                rgb_means=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(
            valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False):

        val_loader = self.get_eval_loader(
            batch_size, is_distributed, testdev=testdev)
        from tracking_utils.evaluation import Evaluator
        
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            backbone = YOLOFPN()
            head = YOLOXHead(self.num_classes, self.width,
                             in_channels=[128, 256, 512], act="lrelu")
            self.model = YOLOX(backbone, head)
        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False):
        from data.datasets.cocodataset import COCODataset
        from data.datasets.mosaicdetection import MosaicDetection
        from data.datasets.data_augment import TrainTransform
        from data.datasets.dataloading import YoloBatchSampler, DataLoader, InfiniteSampler
        import torch.distributed as dist

        dataset = COCODataset(
            data_dir='data/COCO/',
            json_file=self.train_ann,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=50
            ),
        )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                rgb_means=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_labels=120
            ),
            degrees=self.degrees,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
            perspective=self.perspective,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = InfiniteSampler(
                len(self.dataset), seed=self.seed if self.seed else 0)
        else:
            sampler = torch.utils.data.RandomSampler(self.dataset)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            input_dimension=self.input_size,
            mosaic=not no_aug
        )

        dataloader_kwargs = {
            "num_workers": self.num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader
