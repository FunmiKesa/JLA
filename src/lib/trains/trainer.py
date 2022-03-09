#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import json
from loguru import logger

import torch

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision.transforms import transforms as T

from torch.utils.tensorboard import SummaryWriter
from datasets.dataset_factory import get_dataset
from opts import opts

from datasets.data_prefetcher import DataPrefetcher
from models.model import load_model, save_model, create_model
from trains.base_trainer import ModelWithLoss
from trains.mot import MotLoss
from utils import (
    MeterBuffer,
    ModelEMA,
    all_reduce_norm,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    occupy_mem,
    setup_logger,
    synchronize
)

import datetime
import os
import time


class Trainer:
    def __init__(self, exp, opts):
        self.exp = exp
        self.opts = opts

        # training related attr
        self.max_epoch = opts.num_epochs
        self.amp_training = opts.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=opts.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = opts.local_rank
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema

        # data/dataloader related attr
        self.data_type = torch.float16 if opts.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = opts.save_dir

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()
        epoch = self.epoch + 1

        batch = self.prefetcher.next()
        if batch is None:
            return
        inps = batch['input']
        pasts = batch['pasts']
        inps = inps.to(self.data_type)
        pasts = pasts.to(self.data_type)
        batch['input'] = [inps, pasts]
        # targets.requires_grad = False
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            output, loss, loss_stats = self.model(batch)
            # outputs = self.model(inps)
        # loss = outputs["total_loss"]
        loss = loss.mean()

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        # if epoch in self.opts.lr_step:
        #     lr = self.opts.lr * (0.1 ** (self.opts.lr_step.index(epoch) + 1))
        #     print('Drop LR to', lr)
        #     for param_group in self.optimizer.param_groups:
        #         param_group['lr'] = lr
                
        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **loss_stats,
        )

    def before_train(self):
        logger.info("opts: {}".format(self.opts))
        logger.info("exp value:\n{}".format(self.exp))
        torch.cuda.set_device(self.local_rank)

        # data related init
        Dataset = get_dataset(self.opts.dataset, self.opts.task)
        f = open(self.opts.data_cfg)
        data_config = json.load(f)
        trainset_paths = data_config['train']
        valset_paths = data_config['test']

        dataset_root = data_config['root']
        f.close()
        transforms = T.Compose([T.ToTensor()])
        dataset = Dataset(self.opts, dataset_root, trainset_paths,
                        (1088, 608), augment=True, transforms=transforms)
        self.opts = opts().update_dataset_info_and_set_heads(self.opts, dataset)

        val_dataset = Dataset(self.opts, dataset_root, valset_paths,
                        (1088, 608), augment=False, transforms=transforms)

        

        batch_size = self.opts.batch_size
        if self.is_distributed:
            batch_size = batch_size // dist.get_world_size()
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset)
        else:
            train_sampler = None

         # model related init
        model = create_model(self.opts.arch, self.opts.heads, self.opts.head_conv)
        self.loss_stats, self.loss = self._get_losses(self.opts)

        model = ModelWithLoss(model, self.loss)
        model.to(self.device)
        logger.info(
            "Model Summary: {}".format(
                get_model_info(model.model, self.opts.img_size))
        )
         # model related init

        self.train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=self.opts.num_workers, pin_memory=True, sampler=train_sampler, drop_last=False)
        self.val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=self.opts.num_workers, pin_memory=True, sampler=train_sampler, drop_last=False)
        
        # self.train_loader = self.exp.get_data_loader(
        #     batch_size=self.opts.batch_size,
        #     is_distributed=self.is_distributed,
        #     no_aug=self.no_aug,
        # )

        logger.info("init prefetcher, this might take one minute or less...")
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        # solver related init
        # self.optimizer = self.exp.get_optimizer(self.opts.batch_size)
        self.optimizer = torch.optim.Adam(model.parameters(), self.opts.lr)

        if self.opts.load_model != '':
            model, self.optimizer, self.start_epoch = load_model(
                model, self.opts.load_model, self.optimizer, self.opts.resume, self.opts.lr, self.opts.lr_step)

        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.opts.batch_size, self.max_iter
        )
        if self.opts.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[
                        self.local_rank], broadcast_buffers=False, find_unused_parameters=True)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        # self.evaluator = self.exp.get_evaluator(
        #     batch_size=self.opts.batch_size, is_distributed=self.is_distributed
        # )
        # Tensorboard logger
        if self.rank == 0:
            self.tblogger = SummaryWriter(self.file_name)

        logger.info("Training start...")
        # logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(
                self.best_ap * 100
            )
        )

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        self.prefetcher = DataPrefetcher(self.train_loader)


        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:

            logger.info("--->No mosaic aug now!")
            # self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            # if self.is_distributed:
            #     self.model.module.head.use_l1 = True
            # else:
            #     self.model.head.use_l1 = True

            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch.pth")

    def after_epoch(self):
        if self.use_model_ema:
            self.ema_model.update_attr(self.model)

        self.save_ckpt(ckpt_name="latest.pth")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - \
                (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(
                datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.3f}".format(k, v.latest)
                 for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )
            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )
            self.meter.clear_meters()

        # random resizing
        if self.exp.random_size is not None and (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        

        if self.opts.resume:
            logger.info("resume training")
            if self.opts.ckpt is None:
                ckpt_file = os.path.join(
                    self.file_name, "latest" + "_ckpt.pth.tar")
            else:
                ckpt_file = self.opts.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = (
                self.opts.start_epoch - 1
                if self.opts.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.opts.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.opts.load_model is not None:
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        # evalmodel = self.ema_model.ema if self.use_model_ema else self.model
        # ap50_95, ap50, summary = self.exp.eval(
        #     evalmodel, self.evaluator, self.is_distributed
        # )
        # self.model.train()
        # if self.rank == 0:
        #     self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
        #     self.tblogger.add_scalar(
        #         "val/COCOAP50_95", ap50_95, self.epoch + 1)
        #     logger.info("\n" + summary)
        # synchronize()

        #self.best_ap = max(self.best_ap, ap50_95)
        self.save_ckpt("last_epoch.pth")
        # self.save_ckpt("last_epoch.pth", ap50 > self.best_ap)
        # self.best_ap = max(self.best_ap, ap50)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))

            save_model(
                self.file_name,
                self.epoch+1,
                model.model,
                self.optimizer,
                ckpt_name,
                update_best_ckpt,
            )


    def _get_losses(self, opt):
        weights = {
            "wh_loss":opt.wh_weight,
            "off_loss": opt.off_weight,
            "id_loss" : opt.id_weight,
            "loss": 1,
            "hm_loss":1
        }

        if opt.forecast:
            weights.update({
                "f_loss": opt.futures_weight,
                "p_loss": opt.pasts_weight
            })


        # validate loss weight is greater than 0:
        loss_stats = list(weights.keys())
        for k in loss_stats:
            if weights[k] <= 0:
                del weights[k]

        loss_stats = list(weights.keys())

        loss = MotLoss(opt)

        return loss_stats, loss