from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.losses import FocalLoss, TripletLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss
from models.decode import mot_decode
from models.utils import _sigmoid, _tranpose_and_gather_feat
from utils.post_process import ctdet_post_process
from .base_trainer import BaseTrainer


class MotLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MotLoss, self).__init__()
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
            NormRegL1Loss() if opt.norm_wh else \
            RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.opt = opt
        self.emb_dim = opt.reid_dim
        self.nID = opt.nID
        self.classifier = nn.Linear(self.emb_dim, self.nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        #self.TriLoss = TripletLoss()
        self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
        self.s_det = nn.Parameter(-1.85 * torch.ones(1))
        self.s_id = nn.Parameter(-1.05 * torch.ones(1))

        if opt.forecast:
            # from models.networks.forecast_rnn import ForeCastRNN, EncoderRNN, DecoderRNN
            # input_size = self.opt.forecast['input_size']
            # hidden_size = self.opt.forecast['hidden_size']
            # output_size = self.opt.forecast['output_size']

            # self.rnn = ForeCastRNN(input_size, hidden_size)
            # self.encoder = EncoderRNN(self.opt.device, input_size, hidden_size, 1)
            # self.decoder = DecoderRNN(self.opt.device,input_size, output_size, hidden_size, 0, 1)

            # self.FCTLoss = nn.MSELoss()
            self.s_fct = nn.Parameter(-1.85 * torch.ones(1))

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, off_loss, id_loss, pasts_loss, futures_loss = 0, 0, 0, 0, 0, 0
        if self.opt.forecast:
            fct_loss = 0

        for s in range(opt.num_stacks):
            output = outputs[s]
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

            if opt.id_weight > 0:
                id_head = _tranpose_and_gather_feat(output['id'], batch['ind'])
                id_head = id_head[batch['reg_mask'] > 0].contiguous()
                id_head = self.emb_scale * F.normalize(id_head)
                id_target = batch['ids'][batch['reg_mask'] > 0]

                id_output = self.classifier(id_head).contiguous()
                id_loss += self.IDLoss(id_output, id_target)

            if opt.forecast:

                # fc_ = fc_all

                # forecast_mask = batch['forecast_mask']
                # fc_ = fc_.view(batch_size, m, 4, forecast_size).contiguous()
                # fc_ = fc_[..., :opt.past_length].contiguous()

                # target = batch['bb_hist'][..., 5:].permute(0, 2, 3, 1)

                # forecast_mask = batch['forecast_mask'].unsqueeze(1).unsqueeze(2).expand_as(target)

                # fc_ = fc_ * forecast_mask
                # target = target * forecast_mask

                # mask = batch['reg_mask'].unsqueeze(-1).unsqueeze(-1).expand_as(fc_).float()

                # mask = batch['reg_mask'].unsqueeze(2).expand_as(fc_).float()
                # fc_ = fc_ * mask
                # target = target * mask

                # fc_ = fc_[mask > 0]
                # target = target[mask > 0]
                # target = batch['bb_hist']
                # batch_size, past_length, max_objs, input_size = target.shape
                # target = target.permute(1,0,2,3).contiguous().view(past_length, -1, input_size)
                # future_length = self.opt.forecast['future_length']
                # context = self.encoder(target)

                # fc_all = fc_all.view(-1, fc_all.size(2)).contiguous()
                # decoded_input, output= self.decoder(context, fc_all, future_length=future_length, past_length=past_length)

                # target = target.view(target.size(0), batch_size, max_objs,target.size(2)).contiguous().permute(1, 0, 2, 3).contiguous()
                # decoded_input = decoded_input.view(decoded_input.size(0), batch_size, max_objs,decoded_input.size(2)).contiguous().permute(1, 0, 2, 3).contiguous()

                # forecast_mask = batch['forecast_mask'].unsqueeze(-1).unsqueeze(-1).expand_as(target)

                # decoded_input = decoded_input * forecast_mask
                # target = target * forecast_mask

                # target = target.flip([1])
                # index = math.ceil(input_size / 2)

                # target[:, :, :, index:] *= -1
                # pasts = batch['pasts']

                # pasts = pasts.permute(1, 0, 2, 3).contiguous().view(past_length, -1, input_size)

                pred_pasts, pred_futures = output['fct']

                # pasts = pasts.view(past_length, batch_size, max_objs, input_size).contiguous().permute(1, 0, 2, 3).contiguous()

                

                if opt.pasts_weight:
                    # pasts_mask = pasts_mask.unsqueeze(-1).unsqueeze(-1).expand_as(pasts)
                    pasts = batch['pasts']
                    pasts_mask = batch['pasts_mask']
                    batch_size, max_objs, past_length, input_size = pasts_mask.shape
                    
                    pred_pasts = pred_pasts.view( -1, max_objs, pred_pasts.size(1),pred_pasts.size(2)).contiguous()

                    pred_pasts = pred_pasts * pasts_mask.float()
                    pasts = pasts * pasts_mask.float()

                    # pasts = pasts.flip([1])
                    index = math.ceil(input_size / 2)

                    pasts[:, :, :, index:] = pasts[:, :, :, index:] * -1 # reverse direction

                    pasts_loss = pasts_loss  + F.l1_loss(pred_pasts, pasts, size_average=False) / (pasts_mask.sum([0,1]).max() + 1e-4) 

                if opt.futures_weight:
                    # futures_mask = futures_mask.unsqueeze(-1).unsqueeze(-1).expand_as(futures)

                    futures = batch['futures']
                    futures_mask = batch['futures_mask']
                    batch_size, max_objs, future_length, input_size = futures_mask.shape

                    futures = futures.view(-1, max_objs, futures.size(1), futures.size(2))
                    pred_futures = pred_futures.view(-1, max_objs, pred_futures.size(1), pred_futures.size(
                    2)).contiguous()

                    pred_futures = pred_futures * futures_mask.float()

                    futures = futures * futures_mask.float()

                    futures_loss += F.l1_loss(pred_futures, futures, size_average=False) / (
                        futures_mask.sum([0,1]).max() + 1e-4) / opt.num_stacks

                # fct_loss += self.FCLoss(decoded_input * forecast_mask, target * forecast_mask) / opt.num_stacks

                # fct_loss += torch.abs(decoded_input - target).sum() /(forecast_mask.sum() + 1e-4)
                # fct_loss += torch.abs(decoded_input - target).sum() /(past_length * input_size * forecast_mask.sum())

        #loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.off_weight * off_loss + opt.id_weight * id_loss

        fct_loss = opt.pasts_weight * pasts_loss + opt.futures_weight * futures_loss

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * \
            wh_loss + opt.off_weight * off_loss

        loss = torch.exp(-self.s_det) * det_loss + \
            torch.exp(-self.s_id) * id_loss + (self.s_det + self.s_id)

        if self.opt.forecast:
            loss += torch.exp(-self.s_fct) * fct_loss + self.s_fct
        loss *= 0.5
        #loss = det_loss

        #print(loss, hm_loss, wh_loss, off_loss, id_loss)

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'id_loss': id_loss}
        if self.opt.forecast:
            loss_stats['fct_loss'] = fct_loss
        return loss, loss_stats


# class MotLossFlex(MotLoss):
#     def __init__(self, opt, model):
#         super(MotLossFlex, self).__init__(opt)
#         if model:
#             if self.opt.forecast:
#                 self.rnn = model.rnn

class MotTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(MotTrainer, self).__init__(opt, model, optimizer=optimizer)
        # self.model = model
        # if self.opt.forecast:
        #     self.loss_stats += ['fct_loss']
        #     self.model_with_loss.loss =  MotLossFlex(self.opt, self.model)

        #     self.loss_stats += ['fct_loss']
        #     self.model_with_loss.loss = MotLossFlex(self.opt, self.model)

    def _get_losses(self, opt):
        loss_stats = ['loss', 'hm_loss', 'wh_loss', 'off_loss', 'id_loss']
        if opt.forecast:
            loss_stats += ['fct_loss']

        loss = MotLoss(opt)
        return loss_stats, loss

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = mot_decode(
            output['hm'], output['wh'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = ctdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]
