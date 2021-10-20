from __future__ import absolute_import, division, print_function
import _init_paths

import logging
import os
import os.path as osp
import shutil

import cv2
import datasets.dataset.jde as datasets
import motmetrics as mm
import numpy as np
import torch
import torch.nn.functional as F
from models import *
from models.model import create_model, load_model
from opts import opts
from tracking_utils.kalman_filter import KalmanFilter
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.utils import *
from tracking_utils.utils import mkdirs


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(
            opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K
        self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)

        self.use_kf = not opt.no_kf
        self.kalman_filter = KalmanFilter() if self.use_kf else None
        self.forecast = opt.forecast
        self.past_length = 0
        if self.forecast:
            self.past_length = self.forecast['past_length']
            self.future_length = self.forecast['future_length']
            self.hidden_size = self.forecast['hidden_size']
            self.input_size = self.forecast['input_size']
            self.output_size = self.forecast['output_size']

    def update(self, im_blob, img0, pasts_data, p_mask):
        self.frame_id += 1

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        if self.opt.forecast:
            im_blob = [im_blob]
            self.pasts = torch.zeros(
                (self.max_per_image, self.past_length, self.input_size), device=self.opt.device)
            pasts_mask = np.zeros(
                (self.max_per_image, self.past_length), dtype=np.float32)

            ratio = min(float(inp_height) / height, float(inp_width) / width)
            new_shape = (round(width * ratio), round(height * ratio))

            dw = (inp_width - new_shape[0]) / 2  # width padding
            dh = (inp_height - new_shape[1]) / 2  # height padding
            rw = ratio * width / inp_width
            rh = ratio * height / inp_height
            output_h = inp_height // self.opt.down_ratio
            output_w = inp_width // self.opt.down_ratio
            objs_count = 0
            bboxes = []

            pasts = np.zeros(
                (self.max_per_image, self.past_length, self.input_size), dtype=np.float32)
            track_ids = []
            if len(pasts_data) > 0:
                track_ids = pasts_data[:, 0, 1]
                objs_count = len(track_ids)
                bboxes = pasts_data[..., 2:].copy() # xywh
                pasts_mask[:objs_count] = p_mask.max(axis=-1)
                if len(bboxes) > 0:
                    bboxes = np.stack(bboxes, axis=0)
                    bbox = bboxes.copy()
                    labels = bbox.copy()
                    
                    labels[..., [0, 2]] *= output_w
                    labels[..., [1, 3]] *= output_h

                    # flip - oldest first
                    labels = np.flip(labels, 1)
                    pasts_mask = np.flip(pasts_mask, 1)

                    labels_change = np.diff(labels, axis=1)

                    pasts[:labels_change.shape[0], :, 4:] = labels_change
                    pasts[:labels_change.shape[0], :, :4] = labels[:, 1:, :]

                    pasts = pasts * pasts_mask[:, :, np.newaxis]
                    self.pasts = torch.tensor(pasts, device=self.opt.device)

            im_blob += [self.pasts]

        with torch.no_grad():
            output = self.model(im_blob)[-1]
            pred_futures = None
            if len(track_ids) > 0:
                pred_futures = output['fct'][-1]
                pred_futures = pred_futures.cpu().numpy()
                # flip back
                pasts_mask = np.flip(pasts_mask, 1)
                mask = pasts_mask.max(axis=1)

                # pp = pred_futures.clone()
                pred_futures = pred_futures * mask[:, np.newaxis, np.newaxis, ]
                pred_futures = pred_futures[:objs_count]


                pred_futures[..., [0, 2]] /= output_w
                pred_futures[..., [1, 3]] /= output_h
                pred_futures[..., [1, 3]] *= inp_height
                pred_futures[..., [0, 2]] *= inp_width
                pred_futures = xywh2xyxy(pred_futures.copy())
                pred_futures[..., [1, 3]] -= dh
                pred_futures[..., [0, 2]] -= dw
                pred_futures /= ratio
               
                # pred_futures[..., [0,2]] = np.clip(pred_futures[..., [0,2]], 0, width)
                # pred_futures[..., [1,3]] = np.clip(pred_futures[..., [1,3]], 0, height)
        return track_ids, pred_futures


def write_results_forecasts(dir, results):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    mkdirs(dir)
    for frame_id, forecasts in results:
        filename = os.path.join(dir, '{:06d}.txt'.format(frame_id))
        forecasts = np.array(forecasts)
        fmt = fmt = '%d '+'%f ' * (forecasts.shape[-1] - 1)
        np.savetxt(filename, forecasts, fmt=fmt)
    logger.info('save forecast results to {}'.format(dir))

def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    if save_dir:
        mkdirs(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    len_all = len(dataloader)
    start_frame = int(len_all / 2)
    frame_id = int(len_all / 2)
    forecast_results = []

    for i, (path, img, img0, pasts_data, pasts_mask) in enumerate(dataloader):
        if i < start_frame:
            continue
        if frame_id % 100 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))
        
        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)

        online_ids, forecasts = tracker.update(blob, img0, pasts_data, pasts_mask)

        online_forecasts = []
        if forecasts is not None:
            forecasts = xyxy2xywh(forecasts.copy())
            # print(forecasts[0][0])
            forecasts = forecasts.reshape(len(online_ids), -1)
            for i in range(len(online_ids)):
                online_forecasts.append(
                        np.array([online_ids[i]] + list(forecasts[i])))
            # online_forecasts = np.concatenate((online_ids, forecasts), axis=1)
        
        timer.toc()
        # save results
        if len(online_forecasts):
            forecast_results.append((frame_id + 1, online_forecasts))
        frame_id += 1

    if len(forecast_results):
        write_results_forecasts(opt.forecast_dir, forecast_results)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdirs(result_root)
    data_type = 'mot'
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()

    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    if opt.forecast:
        aious = []
        fious = []
        ades = []
        fdes = []
    for i, seq in enumerate(seqs):
        output_dir = os.path.join(
            data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        forecast_dir = os.path.join(
            opt.forecast_pred, seq, 'img1') if opt.forecast else None
        opt.forecast_dir = forecast_dir
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImagesAndPasts(
            osp.join(data_root, seq, 'img1'), opt.img_size, past_length=opt.past_length)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        with open(os.path.join(data_root, seq, 'seqinfo.ini'), 'r') as file:
            meta_info = file.read()
            frame_rate = int(meta_info[meta_info.find(
            'frameRate') + 10:meta_info.find('\nseqLength')])
        # delete later
        opt.data_root = data_root
        opt.seq = seq
        nf, ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        if opt.forecast:
            logger.info('Evaluate seq (forecast): {}'.format(seq))
            future_label_root = osp.join(opt.forecast_root, seq, 'img1')

            from forecast_utils import evaluation
            aiou, fiou, ade, fde = evaluation.eval_seq(
                future_label_root, pred_length=60, pred_folder=f"pred_{exp_name}", fixed_length=True)
            aious.append(aiou)
            fious.append(fiou)
            ades.append(ade)
            fdes.append(fde)

            logger.info('\n')
            logger.info(seq)
            logger.info('AIOU: ' + str(round(aiou, 1)))
            logger.info('FIOU: ' + str(round(fiou, 1)))
            logger.info('ADE:  ' + str(round(ade, 1)))
            logger.info('FDE:  ' + str(round(fde, 1)))

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(
        all_time, 1.0 / avg_time))

    if opt.forecast:
        aiou = round(np.mean(aious), 1)
        fiou = round(np.mean(fious), 1)
        ade = round(np.mean(ades), 1)
        fde = round(np.mean(fdes), 1)

        logger.info('Mean')
        logger.info('AIOU: ' + str(aiou))
        logger.info('FIOU: ' + str(fiou))
        logger.info('ADE:  ' + str(ade))
        logger.info('FDE:  ' + str(fde))

        filename = os.path.join(
            result_root, 'forecast_{}.csv'.format(exp_name))

        evaluation.save_result(filename, [aious, fious, ades, fdes], seqs, [
                               "aiou", "fiou", "ade", "fde"])


if __name__ == '__main__':
    opt = opts().init()
    print(opt)
    if not opt.val_mot16:
        seqs_str = '''KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte'''
        #seqs_str = '''TUD-Campus'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    else:
        seqs_str = '''MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13'''
        data_root = os.path.join(opt.data_dir, 'MOT16/images/train')
    if opt.test_mot16:
        seqs_str = '''MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14'''
        seqs_str = '''MOT16-06 MOT16-07 MOT16-08'''
        data_root = os.path.join(opt.data_dir, 'MOT16/images/test')
    if opt.test_mot15:
        seqs_str = '''ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/test')
    if opt.test_mot17:
        seqs_str = '''MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP'''
        # seqs_str = '''MOT17-01-SDP
        # MOT17-06-SDP
        # MOT17-07-SDP
        # MOT17-12-SDP
        # '''
        #seqs_str = '''MOT17-07-SDP MOT17-08-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/test')
    if opt.val_mot17:
        seqs_str = '''MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP'''
        #seqs_str = '''MOT17-02-SDP'''
        data_root = os.path.join(opt.data_dir, 'MOT17/images/train')
    if opt.val_mot15:
        seqs_str = '''Venice-2
                      KITTI-13
                      KITTI-17
                      ETH-Bahnhof
                      ETH-Sunnyday
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte
                      ADL-Rundle-6
                      ADL-Rundle-8
                      ETH-Pedcross2
                      TUD-Stadtmitte'''
        #seqs_str = '''Venice-2'''
        data_root = os.path.join(opt.data_dir, 'MOT15/images/train')
    if opt.val_mot20:
        seqs_str = '''MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/train')
    if opt.test_mot20:
        seqs_str = '''MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      '''
        data_root = os.path.join(opt.data_dir, 'MOT20/images/test')
    seqs = [seq.strip() for seq in seqs_str.split()]

    if opt.forecast:
        opt.forecast_root = data_root.replace('images', 'future')
        opt.forecast_pred = data_root.replace('images', f'pred_{opt.exp_id}')
        mkdirs(opt.forecast_pred, del_existing=True)
    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.exp_id,
         show_image=False,
         save_images=False,
         save_videos=False)


