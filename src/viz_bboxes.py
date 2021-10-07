from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy.lib.ufunclike import fix

import _init_paths
import os
import os.path as osp
import cv2
import logging
import motmetrics as mm
import numpy as np
import torch
import sys

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from tracking_utils.log import logger
from tracking_utils.timer import Timer
from tracking_utils.evaluation import Evaluator
import datasets.dataset.jde as datasets
from tracking_utils.io import unzip_objs

from tracking_utils.utils import mkdirs
from opts import opts
import shutil


pred_folder = ""
baseline_cv_folder = ""
baseline_kf_folder = ""
GOOD = True
LENGTH = 10

def draw_box(frame,start_x,width,start_y,height,line_weight,color):
    #frame[int(start_y):int(start_y+height),int(start_x):int(start_x+width),:] = 255
    # Top line
    frame[int(start_y):int(start_y+line_weight),int(start_x):int(start_x+width),0] = color[0]
    frame[int(start_y):int(start_y+line_weight),int(start_x):int(start_x+width),1] = color[1]
    frame[int(start_y):int(start_y+line_weight),int(start_x):int(start_x+width),2] = color[2]

    # Bottom line
    frame[int(start_y+height-line_weight):int(start_y+height),int(start_x):int(start_x+width),0] = color[0]
    frame[int(start_y+height-line_weight):int(start_y+height),int(start_x):int(start_x+width),1] = color[1]
    frame[int(start_y+height-line_weight):int(start_y+height),int(start_x):int(start_x+width),2] = color[2]


    # Left line
    frame[int(start_y):int(start_y+height),int(start_x):int(start_x+line_weight),0] = color[0]
    frame[int(start_y):int(start_y+height),int(start_x):int(start_x+line_weight),1] = color[1]
    frame[int(start_y):int(start_y+height),int(start_x):int(start_x+line_weight),2] = color[2]

    # Right line
    frame[int(start_y):int(start_y+height),int(start_x+width-line_weight):int(start_x+width),0] = color[0]
    frame[int(start_y):int(start_y+height),int(start_x+width-line_weight):int(start_x+width),1] = color[1]
    frame[int(start_y):int(start_y+height),int(start_x+width-line_weight):int(start_x+width),2] = color[2]


    return frame

def drawline(img,pt1,pt2,color,thickness=3,style='dotted',gap=7):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='line'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])]
    drawpoly(img,pts,color,thickness,style)

def draw_shaded_dotted_box(frame,start_x,width,start_y,height,line_weight,color,opacity,cross_size=10):
    #frame[int(start_y):int(start_y+height),int(start_x):int(start_x+width),:] = 255
    overlay = frame.copy()
    end_x = int(start_x + width)
    end_y = int(start_y + height)
    start_x = int(start_x)
    start_y = int(start_y)
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y),color, -1)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    drawrect(frame,(start_x,start_y),(end_x,end_y),color,line_weight)
    mid_x = int((start_x + end_x) / 2)
    mid_y = int((start_y + end_y) / 2)

    frame = cv2.line(frame, (mid_x+cross_size,mid_y-cross_size),(mid_x-cross_size,mid_y+cross_size), color,3)
    frame = cv2.line(frame, (mid_x+cross_size,mid_y+cross_size),(mid_x-cross_size,mid_y-cross_size), color,3)
    return frame

def draw_shaded_box(frame,start_x,width,start_y,height,line_weight,color,opacity,cross_size=10):
    #frame[int(start_y):int(start_y+height),int(start_x):int(start_x+width),:] = 255
    overlay = frame.copy()
    end_x = int(start_x + width)
    end_y = int(start_y + height)
    start_x = int(start_x)
    start_y = int(start_y)
    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y),color, -1)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
    frame = draw_box(frame,start_x,width,start_y,height,line_weight,color)

    mid_x = int((start_x + end_x) / 2)
    mid_y = int((start_y + end_y) / 2)

    frame = cv2.line(frame, (mid_x+cross_size,mid_y-cross_size),(mid_x-cross_size,mid_y+cross_size), color,line_weight)
    frame = cv2.line(frame, (mid_x+cross_size,mid_y+cross_size),(mid_x-cross_size,mid_y-cross_size), color,line_weight)

    return frame


def draw_trajectory(im,xs,ys,color,weight):
    prev_x = xs[0]
    prev_y = ys[0]
    for x,y in zip(xs,ys):
        im = cv2.line(im, (int(x), int(y)), (int(prev_x), int(prev_y)), color, weight)
        prev_x = x
        prev_y = y
    return im

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


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_score(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, s=score)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=True, frame_rate=30):
    
    if save_dir:
        mkdirs(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    forecast_results = []
    evaluator = Evaluator(opt.data_root, opt.seq, data_type)

    # for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        # if i % 8 != 0:
        # continue
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1. / max(1e-5, timer.average_time)))

        # run tracking
        timer.tic()
        blob = torch.from_numpy(img).cuda().unsqueeze(0)
        online_targets = tracker.update(blob, img0)
        online_tlwhs = []
        online_ids = []
        # online_scores = []
        online_forecasts = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                # online_scores.append(t.score)
                if len(t.forecasts):
                    online_forecasts.append(
                        np.array([tid] + list(t.forecasts_xywh.reshape(-1))))
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        if len(online_forecasts):
            forecast_results.append((frame_id + 1, online_forecasts))
        #results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if show_image or save_dir is not None:
            # online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
            #                               fps=1. / timer.average_time, forecasts=online_forecasts)
            gt_objs = evaluator.gt_frame_dict.get(frame_id+1, [])
            gt_tlwhs, gt_ids = unzip_objs(gt_objs)[:2]
            online_im = vis.plot_tracking(img0, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time, gt_tlwhs=gt_tlwhs, gt_ids=gt_ids, forecasts=online_forecasts)
        if show_image:
            cv2.imshow('online_im', online_im)
        if save_dir is not None:
            cv2.imwrite(os.path.join(
                save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
        frame_id += 1
    # save results
    write_results(result_filename, results, data_type)
    if len(forecast_results):
        write_results_forecasts(opt.forecast_dir, forecast_results)
    #write_results_score(result_filename, results, data_type)
    return frame_id, timer.average_time, timer.calls


def main(opt, data_root='/data/MOT16/train', det_root=None, seqs=('MOT16-05',), exp_name='demo',
         save_images=False, save_videos=False, show_image=True):
    
    logger.setLevel(logging.INFO)
    logger.info((str(sys.argv)))
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    data_type = 'mot'
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    aious = []
    fious = []
    ades = []
    fdes = []
    gt_folder = "future"
    pred_folder  = f"pred_{exp_name}"
    for i, seq in enumerate(seqs):
        future_label_root = osp.join(opt.forecast_root, seq, 'img1')

        from forecast_utils import evaluation
        from forecast_utils.utils import calc_fiou

        logger.info('Evaluate seq (forecast): {}'.format(future_label_root))
        bboxes = evaluation.get_bboxes(future_label_root, pred_folder=pred_folder, fixed_length=opt.fixed_length, pred_length=opt.future_length)
        bboxes1, bboxes2, filenames= bboxes[gt_folder], bboxes[pred_folder], bboxes['filenames']

        pred_fiou = np.array(calc_fiou(bboxes1, bboxes2, False))

        if GOOD:
            ordered = np.argsort(pred_fiou)[::-1]
            perf_type = "good"
        else:
            ordered = np.argsort(pred_fiou)
            perf_type = "bad"

        gt_bboxes = bboxes1[ordered[:LENGTH]].copy()
        pred_bboxes = bboxes2[ordered[:LENGTH]].copy()

        print(pred_fiou[ordered[:10]])
        for i in range(0, len(gt_bboxes)):
            gt_bbox = gt_bboxes[i]
            pred_bbox = pred_bboxes[i]

            img_folder = future_label_root.replace(gt_folder, "images")
            img_file = filenames[ordered[i]].replace("txt", "jpg")
            # print(len(filenames))
            # print(future_label_root, img_file)
            # print(img_file)
            im = cv2.imread(img_file)

            
            w, h = gt_bbox[-1, 2:] 
            x,y = gt_bbox[-1, :2] - gt_bbox[-1, 2:] /2
            mid_x, mid_y = gt_bbox[-1, :2]

            x,y,w,h,mid_x,mid_y = int(x), int(y), int(w), int(h), int(mid_x), int(mid_y)

            im = draw_shaded_box(im,x,w,y,h,2,(120,240,120),0.8,cross_size=8)
            im = draw_trajectory(im,gt_bbox[:, 0],gt_bbox[:, 1],(120,240,120),2)


            w, h = pred_bbox[-1, 2:] 
            x,y = pred_bbox[-1, :2] - pred_bbox[-1, 2:] /2
            mid_x, mid_y = pred_bbox[-1, :2]

            x,y,w,h,mid_x,mid_y = int(x), int(y), int(w), int(h), int(mid_x), int(mid_y)

            im = draw_shaded_box(im,x,w,y,h,2,(244, 149, 66),0.8,cross_size=8)
            im = draw_trajectory(im,pred_bbox[:, 0],pred_bbox[:, 1], (244, 149, 66),2)

            output_folder = img_folder.replace("train", f"outputs/{exp_name}/{perf_type}").replace("img1", "")
            mkdirs(output_folder)
            frame_num = img_file.split("/")[-1]
            # print(output_folder, (x, y, w, h))
            cv2.imwrite(output_folder + f"{i}.jpg", im)






        

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    opt = opts().init()

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
        #seqs_str = '''MOT16-01 MOT16-07 MOT16-12 MOT16-14'''
        #seqs_str = '''MOT16-06 MOT16-08'''
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
        #seqs_str = '''MOT17-01-SDP MOT17-07-SDP MOT17-12-SDP MOT17-14-SDP'''
        #seqs_str = '''MOT17-03-SDP'''
        #seqs_str = '''MOT17-06-SDP MOT17-08-SDP'''
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

    opt.forecast_root = data_root.replace('images', 'future')
    opt.forecast_pred = data_root.replace('images', f'pred_{opt.exp_id}')
    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.exp_id,
         show_image=False,
         save_images=True,
        #  save_videos=True
         )