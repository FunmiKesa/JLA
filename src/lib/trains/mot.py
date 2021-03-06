from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = (
            RegL1Loss()
            if opt.reg_loss == "l1"
            else RegLoss()
            if opt.reg_loss == "sl1"
            else None
        )
        self.crit_wh = (
            torch.nn.L1Loss(reduction="sum")
            if opt.dense_wh
            else NormRegL1Loss()
            if opt.norm_wh
            else RegWeightedL1Loss()
            if opt.cat_spec_wh
            else self.crit_reg
        )
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

        if opt.forecast:
            self.FCTLoss = nn.MSELoss()
            self.s_fct = nn.Parameter(3.94 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss, pasts_loss, futures_loss = 0, 0, 0, 0, 0, 0
        if self.opt.forecast:
            fct_loss = 0

        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output["hm"] = _sigmoid(output["hm"])

            hm_loss += self.crit(output["hm"], batch["hm"]) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += (
                    self.crit_reg(
                        output["wh"], batch["reg_mask"], batch["ind"], batch["wh"]
                    )
                    / opt.num_stacks
                )

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += (
                    self.crit_reg(
                        output["reg"], batch["reg_mask"], batch["ind"], batch["reg"]
                    )
                    / opt.num_stacks
                )

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output["id"], batch["ind"])
                id_head = id_head[batch["reg_mask"] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch["ids"][batch["reg_mask"] > 0]

                id_output = self.classifier(id_head).contiguous()
                id_loss += self.IDLoss(id_output, id_target)

            if opt.forecast:
                past_length = opt.forecast["past_length"]
                future_length = opt.forecast["future_length"]
                pred_pasts, pred_futures = output["fct"]

                if opt.pasts_weight > 0 and len(batch["pasts"]):
                    pasts = batch["pasts"]
                    batch_size = pasts.shape[0]
                    pasts_mask = batch["pasts_mask"]

                    pasts_mask = (
                        pasts_mask.view(batch_size * opt.K, -1).contiguous().float()
                    )

                    pasts = (
                        pasts.view(batch_size * opt.K, past_length, -1)
                        .contiguous()
                        .float()
                    )

                    pred_pasts = pred_pasts * pasts_mask[:, :, None]

                    pasts_loss += F.l1_loss(pred_pasts, pasts, reduction="sum")

                    if pasts_mask.sum() > 0:
                        pasts_loss /= pasts_mask.sum() * 8

                    pasts_loss /= opt.num_stacks

                if opt.futures_weight > 0 and len(batch["futures"]):
                    futures = batch["futures"]
                    futures_mask = batch["futures_mask"]

                    futures_mask = (
                        futures_mask.view(batch_size * opt.K, -1).contiguous().float()
                    )
                    futures = (
                        futures.view(batch_size * opt.K, future_length, -1)
                        .contiguous()
                        .float()[..., :4]
                    )

                    pred_futures = pred_futures[..., :4] * futures_mask[:, :, None]

                    futures_loss += F.l1_loss(pred_futures, futures, reduction="sum")

                    if futures_mask.sum() > 0:
                        futures_loss /= futures_mask.sum() * 4


                    futures_loss /= opt.num_stacks

        det_loss = (
            opt.hm_weight * hm_loss
            + opt.wh_weight * wh_loss
            + opt.off_weight * off_loss
        )

        loss = (
            torch.exp(-self.s_det) * det_loss
            + torch.exp(-self.s_id) * id_loss
            + (self.s_det + self.s_id)
        )

        if self.opt.forecast:
            fct_loss = opt.pasts_weight * pasts_loss + opt.futures_weight * futures_loss
            loss += torch.exp(-self.s_fct) * fct_loss + self.s_fct
        loss *= 0.5

        loss_stats = {
            "loss": loss,
            "hm_loss": hm_loss,
            "wh_loss": wh_loss,
            "off_loss": off_loss,
            "id_loss": id_loss,
        }
        if self.opt.forecast:
            if opt.futures_weight > 0:
                loss_stats["f_loss"] = futures_loss
            if opt.pasts_weight > 0:
                loss_stats["p_loss"] = pasts_loss
        return loss, loss_stats


class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        weights = {
            "wh_loss": opt.wh_weight,
            "off_loss": opt.off_weight,
            "id_loss": opt.id_weight,
            "loss": 1,
            "hm_loss": 1,
        }

        if opt.forecast:
            weights.update({"f_loss": opt.futures_weight, "p_loss": opt.pasts_weight})

        # validate loss weight is greater than 0:
        loss_stats = list(weights.keys())
        for k in loss_stats:
            if weights[k] <= 0:
                del weights[k]

        loss_stats = list(weights.keys())

        loss = MotLoss(opt)

        return loss_stats, loss

    def save_result(self, output, batch, results):
        reg = output["reg"] if self.opt.reg_offset else None
        dets = mot_decode(
            output["hm"],
            output["wh"],
            reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh,
            K=self.opt.K,
        )
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(),
            batch["meta"]["c"].cpu().numpy(),
            batch["meta"]["s"].cpu().numpy(),
            output["hm"].shape[2],
            output["hm"].shape[3],
            output["hm"].shape[1],
        )
        results[batch["meta"]["img_id"].cpu().numpy()[0]] = dets_out[0]
