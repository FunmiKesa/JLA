from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths


import argparse
import pandas as pd
from .utils import *
from utils.utils import load_txt
import numpy as np
import os
import os.path as osp
import glob
import motmetrics as mm
mm.lap.default_solver = 'lap'


def eval_seq(label_root, pred_length=30, gt_folder='future', pred_folder='pred'):
    future_bboxes = []
    pred_bboxes = []
    for filename in sorted(glob.glob(label_root + "/*.txt")):
        # check if file exist in pred folder
        pred_filename = filename.replace(gt_folder, pred_folder)
        if not osp.exists(pred_filename):
            continue

        bbox, mask = load_txt(filename, column_length=pred_length*4+1)
        mask = mask[:, 1:].reshape(mask.shape[0], -1, 4)
        future_bbox = bbox[:, 1:].reshape(bbox.shape[0], pred_length, -1)

        bbox, _ = load_txt(pred_filename, column_length=pred_length*4+1)
        pred_bbox = bbox[:, 1:].reshape(bbox.shape[0], pred_length, -1)

        f, p = future_bbox[:, 0, :], pred_bbox[:, 0, :]
        
        # get distance matrix
        dists = mm.distances.iou_matrix(f, p, max_iou=0.5)

        valid_i, valid_j = np.where(np.isfinite(dists))

        future_bbox = future_bbox[valid_i]
        mask = mask[valid_i]
        pred_bbox = pred_bbox[valid_j] * mask

        future_bboxes += [future_bbox]
        pred_bboxes += [pred_bbox]

    future_bboxes = np.concatenate(future_bboxes)
    pred_bboxes = np.concatenate(pred_bboxes)

    ade = calc_ade(future_bboxes, pred_bboxes)
    fde = calc_fde(future_bboxes, pred_bboxes)
    aiou = calc_aiou(future_bboxes, pred_bboxes) * 100
    fiou = calc_fiou(future_bboxes, pred_bboxes) * 100

    return aiou, fiou, ade, fde


def eval(label_root, pred_length=30, gt_folder='future', pred_folder='pred'):
    seqs = [s for s in os.listdir(label_root)]

    aious = []
    fious = []
    ades = []
    fdes = []

    for seq in seqs:
        seq_label_root = osp.join(label_root, seq, gt_folder)

        aiou, fiou, ade, fde = eval_seq(seq_label_root, pred_length, gt_folder, pred_folder)

        aious.append(aiou)
        fious.append(fiou)
        ades.append(ade)
        fdes.append(fde)

        print()
        print(seq)
        print('AIOU: ', round(aiou, 1))
        print('FIOU: ', round(fiou, 1))
        print('ADE:  ', round(ade, 1))
        print('FDE:  ', round(fde, 1))

    print()
    aiou = round(np.mean(aious), 1)
    fiou = round(np.mean(fious), 1)
    ade = round(np.mean(ades), 1)
    fde = round(np.mean(fdes), 1)

    print('Mean')
    print('AIOU: ', aiou)
    print('FIOU: ', fiou)
    print('ADE:  ', ade)
    print('FDE:  ', fde)

    return aiou, fiou, ade, fde

def save_result(filename, result, index, columns):
    result = np.array(result).T
    df = pd.DataFrame(result, index=index, columns=columns)
    df.loc['Mean'] = df.mean()
    df.to_csv(filename)

# def save_result(filename, result, index, columns):
#     import pandas as pd
#     df = pd.DataFrame(result, index=index, columns=columns)
#     df['Mean'] = df.mean(axis=1)
#     writer = pd.ExcelWriter(filename)
#     df.to_excel(writer)
#     writer.save()