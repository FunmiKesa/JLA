from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import logging
import os
import os.path as osp
from opts import opts
from tracking_utils.utils import mkdirs
from tracking_utils.log import logger
import datasets.dataset.jde as datasets
from track import eval_seq
import motmetrics as mm
from tracking_utils.evaluation import Evaluator
import sys
import numpy as np

logger.setLevel(logging.INFO)


def process_video(opt):
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdirs(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate

    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    forecast_dir = os.path.join('./pred') if opt.forecast else None
    opt.forecast_dir = forecast_dir
    eval_seq(opt, dataloader, 'mot', result_filename,
             save_dir=frame_dir, show_image=False, frame_rate=frame_rate)

    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'MOT16-03-results.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(osp.join(result_root, 'frame'), output_video_path)
        os.system(cmd_str)


def main(opt):
    logger.setLevel(logging.INFO)
    logger.info((str(sys.argv)))
    exp_name =  opt.exp_id
    data_root = os.path.join(opt.data_dir, 'CityWalks/clips')
    result_root = os.path.join(data_root, '..', 'results', exp_name
    )
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

    seqs = sorted(os.listdir(data_root))
    vids = []
    opt.forecast_root = data_root.replace('clips', 'future')
    for i, seq in enumerate(seqs):
        data_root_seq = osp.join(data_root, seq)
        for vid in os.listdir(data_root_seq):
            opt.filename = f'{seq}_{vid}'
            logger.info('Starting tracking...')
            dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
            result_filename = os.path.join(result_root, '{opt_filename}.txt')
            frame_rate = dataloader.frame_rate

            frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame', opt.filename)
            forecast_dir = os.path.join('./pred') if opt.forecast else None
            opt.forecast_dir = forecast_dir
            nf, ta, tc =eval_seq(opt, dataloader, 'mot', result_filename,
                    save_dir=frame_dir, show_image=False, frame_rate=frame_rate)

           

            n_frame += nf
            timer_avgs.append(ta)
            timer_calls.append(tc)

            # eval
            logger.info('Evaluate seq: {}'.format(seq))
            evaluator = Evaluator(data_root, opt.filename, data_type)
            accs.append(evaluator.eval_file(result_filename))
            vids += [vid]
            summary = Evaluator.get_summary(accs, vids, metrics)
            strsummary = mm.io.render_summary(
                summary,
                formatters=mh.formatters,
                namemap=mm.io.motchallenge_metric_names
            )
            logger.info("\n"+strsummary)
            summary.to_csv(os.path.join(
            result_root, 'summary_{}.csv'.format(exp_name)))

            # evaluate forecast results
            if opt.forecast:
                logger.info('Evaluate seq (forecast): {}'.format(seq))
                future_label_root = osp.join(opt.forecast_root, seq, 'img1')

                from forecast_utils import evaluation
                aiou, fiou, ade, fde = evaluation.eval_seq(future_label_root, pred_folder= f"pred_{exp_name}")
                aious.append(aiou)
                fious.append(fiou)
                ades.append(ade)
                fdes.append(fde)

                logger.info('\n')
                logger.info(seq)
                logger.info('AIOU: ' + round(aiou, 1))
                logger.info('FIOU: ' + round(fiou, 1))
                logger.info('ADE:  ' + round(ade, 1))
                logger.info('FDE:  ' + round(fde, 1))

                filename = os.path.join(
                result_root, 'forecast_{}.csv'.format(exp_name))

                evaluation.save_result(filename, [aious, fious, ades, fdes], seqs[:i+1], ["aiou", "fiou", "ade", "fde"])
            
            
            if opt.output_format == 'video':
                output_video_path = osp.join(result_root, opt.filename)
                cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}'.format(frame_dir, output_video_path)
                os.system(cmd_str)


        timer_avgs = np.asarray(timer_avgs)
        timer_calls = np.asarray(timer_calls)
        all_time = np.dot(timer_avgs, timer_calls)
        avg_time = all_time / np.sum(timer_calls)
        logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(
            all_time, 1.0 / avg_time))

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

            logger.info('Mean')
            logger.info('AIOU: ' + aiou)
            logger.info('FIOU: ' + fiou)
            logger.info('ADE:  ' + ade)
            logger.info('FDE:  ' + fde)

            # filename = os.path.join(
                # result_root, 'forecast_{}.csv'.format(exp_name))

            # evaluation.save_result(filename, [aious, fious, ades, fdes], seqs, ["aiou", "fiou", "ade", "fde"])


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()
    main(opt)
