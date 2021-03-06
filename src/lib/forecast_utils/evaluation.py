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

def get_bboxes(gt_label_root, pred_length=30, gt_folder='future', pred_folder='pred', fixed_length=False):
    bboxes1 = []
    bboxes2 = []
    pred_label_root = gt_label_root.replace(gt_folder, pred_folder)
    gt_files = sorted(glob.glob(gt_label_root+"/*.txt"))
    pred_files = sorted(glob.glob(pred_label_root+"/*.txt"))

    if len(pred_files) < len(gt_files):
        larger, smaller = gt_files, pred_files
        larger_folder, smaller_folder = gt_folder, pred_folder

    else:
        smaller, larger = gt_files, pred_files
        smaller_folder, larger_folder = gt_folder, pred_folder

    intersect = [filename for filename in smaller if filename.replace(
        smaller_folder, larger_folder) in larger]
    info = []
    for filename1 in intersect:

        filename2 = filename1.replace(smaller_folder, larger_folder)

        bbox1, mask1 = load_txt(filename1, column_length=pred_length*4+1)
        mask1 = mask1[:, 1:].reshape(mask1.shape[0], -1, 4)
        bbox1 = bbox1[:, 1:].reshape(bbox1.shape[0], pred_length, -1)

        bbox2, mask2 = load_txt(filename2, column_length=pred_length*4+1)
        mask2 = mask2[:, 1:].reshape(mask2.shape[0], -1, 4)
        bbox2 = bbox2[:, 1:].reshape(bbox2.shape[0], pred_length, -1)

        b1, b2 = bbox1[:, 0, :].copy(), bbox2[:, 0, :].copy()
        b1[:, :2] -= b1[:, 2:] / 2
        b2[:, :2] -= b2[:, 2:] / 2

        # get distance matrix
        dists = mm.distances.iou_matrix(b1, b2, max_iou=0.5)

        # match_is, match_js = np.where(np.isfinite(dists))
        match_is, match_js = mm.lap.linear_sum_assignment(dists)
        match_is, match_js = map(lambda a: np.asarray(
            a, dtype=int), [match_is, match_js])
        match_ious = dists[match_is, match_js]

        match_js = np.asarray(match_js, dtype=int)
        match_js = match_js[np.logical_not(np.isnan(match_ious))]
        match_is = match_is[np.logical_not(np.isnan(match_ious))]

        bbox1 = bbox1[match_is]
        mask1 = mask1[match_is]

        bbox2 = bbox2[match_js]
        mask2 = mask2[match_js]

        mask = mask1 & mask2
            
        bbox1 *= mask
        bbox2 *= mask

        if fixed_length:
            # Quick check
            obj_filter = mask.min(axis=(1,2)) > 0
            bbox1 = bbox1[obj_filter]
            bbox2 = bbox2[obj_filter]

            # or
            # total = pred_length * 4
            # obj_filter = mask.sum(axis=(1,2)) == total
            # bbox1 = bbox1[obj_filter]
            # bbox2 = bbox2[obj_filter]

        bboxes1 += [bbox1]
        bboxes2 += [bbox2]
        if len(bbox1):
            info.extend([filename1.replace(smaller_folder, "images")] * len(bbox1))

    if len(bboxes1) > 0:
        bboxes1 = np.concatenate(bboxes1)
    else:
        bboxes1 = np.array([])

    if len(bboxes2) > 0:
        bboxes2 = np.concatenate(bboxes2)
    else:
        bboxes2 = np.array([])

    print(smaller_folder, larger_folder)
    return {smaller_folder: bboxes1, larger_folder: bboxes2, "filenames":info}

 
def eval_seq(gt_label_root, pred_length=30, gt_folder='future', pred_folder='pred', fixed_length=True, return_mean=True):
    
    bboxes = get_bboxes(gt_label_root, pred_length, gt_folder, pred_folder, fixed_length)
    bboxes1, bboxes2 = bboxes[gt_folder], bboxes[pred_folder]

    aiou, fiou, ade, fde = 0, 0, 0, 0
    if (len(bboxes1) > 0) & (len(bboxes2) > 0):
        ade = calc_ade(bboxes1, bboxes2, return_mean)
        fde = calc_fde(bboxes1, bboxes2, return_mean)
        aiou = calc_aiou(bboxes1, bboxes2, return_mean) 
        fiou = calc_fiou(bboxes1, bboxes2, return_mean) 
        if not return_mean:
            aiou *= 100
            fiou *= 100

    return aiou, fiou, ade, fde


def eval(label_root, pred_length=30, gt_folder='future', pred_folder='pred', filename=None):
    seqs = [s for s in os.listdir(label_root)]

    aious = []
    fious = []
    ades = []
    fdes = []
    i = 0
    for seq in seqs:
        seq_label_root = osp.join(label_root, seq, 'img1')

        aiou, fiou, ade, fde = eval_seq(
            seq_label_root, pred_length, gt_folder, pred_folder)

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

        if filename:
            save_result(filename, [aious, fious, ades, fdes],
                        seqs[:i+1], ["aiou", "fiou", "ade", "fde"])
        i += 1

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

    # if filename:
    #     save_result(filename, [aious, fious, ades, fdes], seqs, ["aiou", "fiou", "ade", "fde"])

    return aious, fious, ades, fdes


def save_result(filename, result, index, columns):
    result = np.array(result).T
    df = pd.DataFrame(result, index=index, columns=columns)
    df.loc['Mean'] = df.mean()
    df.to_csv(filename)

    print(df)

# def save_result(filename, result, index, columns):
#     import pandas as pd
#     df = pd.DataFrame(result, index=index, columns=columns)
#     df['Mean'] = df.mean(axis=1)
#     writer = pd.ExcelWriter(filename)
#     df.to_excel(writer)
#     writer.save()
