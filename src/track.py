from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

from tracking_utils.utils import mkdirs
from opts import opts
import shutil


def write_results_forecasts(dir, results):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    mkdirs(dir)
    for frame_id, forecasts in results:
        filename = os.path.join(dir, "{:06d}.txt".format(frame_id))
        forecasts = np.array(forecasts)
        fmt = fmt = "%d " + "%f " * (forecasts.shape[-1] - 1)
        np.savetxt(filename, forecasts, fmt=fmt)
    logger.info("save forecast results to {}".format(dir))


def write_results(filename, results, data_type):
    if data_type == "mot":
        save_format = "{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n"
    elif data_type == "kitti":
        save_format = "{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n"
    else:
        raise ValueError(data_type)

    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == "kitti":
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h
                )
                f.write(line)
    logger.info("save results to {}".format(filename))


def write_results_score(filename, results, data_type):
    if data_type == "mot":
        save_format = "{frame},{id},{x1},{y1},{w},{h},{s},1,-1,-1,-1\n"
    elif data_type == "kitti":
        save_format = "{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n"
    else:
        raise ValueError(data_type)

    with open(filename, "w") as f:
        for frame_id, tlwhs, track_ids, scores in results:
            if data_type == "kitti":
                frame_id -= 1
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    w=w,
                    h=h,
                    s=score,
                )
                f.write(line)
    logger.info("save results to {}".format(filename))


def eval_seq(
    opt,
    dataloader,
    data_type,
    result_filename,
    save_dir=None,
    show_image=True,
    frame_rate=30,
):
    if save_dir:
        mkdirs(save_dir)
    tracker = JDETracker(opt, frame_rate=frame_rate)
    timer = Timer()
    results = []
    frame_id = 0
    forecast_results = []

    # for path, img, img0 in dataloader:
    for i, (path, img, img0) in enumerate(dataloader):
        # if i % 8 != 0:
        # continue
        if frame_id % 20 == 0:
            logger.info(
                "Processing frame {} ({:.2f} fps)".format(
                    frame_id, 1.0 / max(1e-5, timer.average_time)
                )
            )

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
                        np.array([tid] + list(t.forecasts_xywh.reshape(-1)))
                    )
        timer.toc()
        # save results
        results.append((frame_id + 1, online_tlwhs, online_ids))
        # results.append((frame_id + 1, online_tlwhs, online_ids, online_scores))
        if len(online_forecasts):
            forecast_results.append((frame_id + 1, online_forecasts))
        if show_image or save_dir is not None:
            online_im = vis.plot_tracking(
                img0,
                online_tlwhs,
                online_ids,
                frame_id=frame_id,
                fps=1.0 / timer.average_time,
                forecasts=online_forecasts,
            )
        if show_image:
            cv2.imshow("online_im", online_im)
        if save_dir is not None:
            cv2.imwrite(
                os.path.join(save_dir, "{:05d}.jpg".format(frame_id)), online_im
            )
        frame_id += 1

    # save results
    write_results(result_filename, results, data_type)
    # write_results_score(result_filename, results, data_type)
    if len(forecast_results):
        write_results_forecasts(opt.forecast_dir, forecast_results)
    return frame_id, timer.average_time, timer.calls


def main(
    opt,
    data_root="/data/MOT16/train",
    det_root=None,
    seqs=("MOT16-05",),
    exp_name="demo",
    save_images=False,
    save_videos=False,
    show_image=True,
):
    logger.setLevel(logging.INFO)
    logger.info((str(sys.argv)))
    result_root = os.path.join(data_root, "..", "results", exp_name)
    mkdirs(result_root)
    data_type = "mot"
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
        output_dir = (
            os.path.join(data_root, "..", "outputs", exp_name, seq)
            if save_images or save_videos
            else None
        )
        forecast_dir = (
            os.path.join(opt.forecast_pred, seq, "img1") if opt.forecast else None
        )
        opt.forecast_dir = forecast_dir
        logger.info("start seq: {}".format(seq))
        logger.info(data_root)
        dataloader = datasets.LoadImages(osp.join(data_root, seq, "img1"), opt.img_size)
        result_filename = os.path.join(result_root, "{}.txt".format(seq))
        meta_info = open(os.path.join(data_root, seq, "seqinfo.ini")).read()
        frame_rate = int(
            meta_info[meta_info.find("frameRate") + 10 : meta_info.find("\nseqLength")]
        )
        nf, ta, tc = eval_seq(
            opt,
            dataloader,
            data_type,
            result_filename,
            save_dir=output_dir,
            show_image=show_image,
            frame_rate=frame_rate,
        )
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        logger.info("Evaluate seq: {}".format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
        summary = Evaluator.get_summary(accs, seqs[: i + 1], metrics)
        strsummary = mm.io.render_summary(
            summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names
        )
        logger.info("\n" + strsummary)
        summary.to_csv(os.path.join(result_root, "summary_{}.csv".format(exp_name)))

        # evaluate forecast results
        if opt.forecast:
            logger.info("Evaluate seq (forecast): {}".format(seq))
            future_label_root = osp.join(opt.forecast_root, seq, "img1")

            from forecast_utils import evaluation

            aiou, fiou, ade, fde = evaluation.eval_seq(
                future_label_root,
                pred_length=opt.future_length,
                pred_folder=f"pred_{exp_name}",
                fixed_length=opt.fixed_length,
            )
            aious.append(aiou)
            fious.append(fiou)
            ades.append(ade)
            fdes.append(fde)

            logger.info("\n")
            logger.info(seq)
            logger.info("AIOU: " + str(round(aiou, 1)))
            logger.info("FIOU: " + str(round(fiou, 1)))
            logger.info("ADE:  " + str(round(ade, 1)))
            logger.info("FDE:  " + str(round(fde, 1)))

            filename = os.path.join(result_root, "forecast_{}.csv".format(exp_name))

            evaluation.save_result(
                filename,
                [aious, fious, ades, fdes],
                seqs[: i + 1],
                ["aiou", "fiou", "ade", "fde"],
            )

        if save_videos:
            output_video_path = osp.join(output_dir, "{}.mp4".format(seq))
            cmd_str = "ffmpeg -f image2 -i {}/%05d.jpg -c:v copy {}".format(
                output_dir, output_video_path
            )
            os.system(cmd_str)

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info(
        "Time elapsed: {:.2f} seconds, FPS: {:.2f}".format(all_time, 1.0 / avg_time)
    )

    # get summary

    # summary = Evaluator.get_summary(accs, seqs, metrics)
    # strsummary = mm.io.render_summary(
    #     summary,
    #     formatters=mh.formatters,
    #     namemap=mm.io.motchallenge_metric_names
    # )
    # logger.info(strsummary)
    # Evaluator.save_summary(summary, os.path.join(
    # result_root, 'summary_{}.xlsx'.format(exp_name)))
    # summary.to_csv(os.path.join(
    #     result_root, 'summary_{}.csv'.format(exp_name)))

    if opt.forecast:
        aiou = round(np.mean(aious), 1)
        fiou = round(np.mean(fious), 1)
        ade = round(np.mean(ades), 1)
        fde = round(np.mean(fdes), 1)

        logger.info("Mean")
        logger.info("AIOU: " + str(aiou))
        logger.info("FIOU: " + str(fiou))
        logger.info("ADE:  " + str(ade))
        logger.info("FDE:  " + str(fde))

        # filename = os.path.join(
        # result_root, 'forecast_{}.csv'.format(exp_name))

        # evaluation.save_result(filename, [aious, fious, ades, fdes], seqs, ["aiou", "fiou", "ade", "fde"])


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    opt = opts().init()

    if not opt.val_mot16:
        seqs_str = """KITTI-13
                      KITTI-17
                      ADL-Rundle-6
                      PETS09-S2L1
                      TUD-Campus
                      TUD-Stadtmitte"""
        data_root = os.path.join(opt.data_dir, "MOT15/images/train")
    else:
        seqs_str = """MOT16-02
                      MOT16-04
                      MOT16-05
                      MOT16-09
                      MOT16-10
                      MOT16-11
                      MOT16-13"""
        data_root = os.path.join(opt.data_dir, "MOT16/images/train")
    if opt.test_mot16:
        seqs_str = """MOT16-01
                      MOT16-03
                      MOT16-06
                      MOT16-07
                      MOT16-08
                      MOT16-12
                      MOT16-14"""
        data_root = os.path.join(opt.data_dir, "MOT16/images/test")
    if opt.test_mot15:
        seqs_str = """ADL-Rundle-1
                      ADL-Rundle-3
                      AVG-TownCentre
                      ETH-Crossing
                      ETH-Jelmoli
                      ETH-Linthescher
                      KITTI-16
                      KITTI-19
                      PETS09-S2L2
                      TUD-Crossing
                      Venice-1"""
        data_root = os.path.join(opt.data_dir, "MOT15/images/test")
    if opt.test_mot17:
        seqs_str = """MOT17-01-SDP
                      MOT17-03-SDP
                      MOT17-06-SDP
                      MOT17-07-SDP
                      MOT17-08-SDP
                      MOT17-12-SDP
                      MOT17-14-SDP"""
        data_root = os.path.join(opt.data_dir, "MOT17/images/test")
    if opt.val_mot17:
        seqs_str = """MOT17-02-SDP
                      MOT17-04-SDP
                      MOT17-05-SDP
                      MOT17-09-SDP
                      MOT17-10-SDP
                      MOT17-11-SDP
                      MOT17-13-SDP"""
        data_root = os.path.join(opt.data_dir, "MOT17/images/train")
    if opt.val_mot15:
        seqs_str = """Venice-2
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
                      TUD-Stadtmitte"""
        data_root = os.path.join(opt.data_dir, "MOT15/images/train")
    if opt.val_mot20:
        seqs_str = """MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                      """
        data_root = os.path.join(opt.data_dir, "MOT20/images/train")
    if opt.test_mot20:
        seqs_str = """MOT20-04
                      MOT20-06
                      MOT20-07
                      MOT20-08
                      """
        data_root = os.path.join(opt.data_dir, "MOT20/images/test")
    seqs = [seq.strip() for seq in seqs_str.split()]

    if opt.forecast:
        opt.forecast_root = data_root.replace("images", "future")
        opt.forecast_pred = data_root.replace("images", f"pred_{opt.exp_id}")
        mkdirs(opt.forecast_pred, del_existing=True)
    main(
        opt,
        data_root=data_root,
        seqs=seqs,
        exp_name=opt.exp_id,
        show_image=False,
        save_images=False,
        save_videos=False,
    )
