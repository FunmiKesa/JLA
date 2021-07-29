import numpy as np
from numba import jit
from collections import deque
import itertools
import os
import os.path as osp
import time
import torch
import cv2
import torch.nn.functional as F

from models.model import create_model, load_model
from models.decode import mot_decode
from tracking_utils.utils import *
from tracking_utils.log import logger
from tracking_utils.kalman_filter import KalmanFilter
from models import *
from tracker import matching
from .basetrack import BaseTrack, TrackState
from utils.post_process import ctdet_post_process
from utils.image import get_affine_transform
from models.utils import _tranpose_and_gather_feat
import copy

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, temp_feat, buffer_size=30, past_length=15, use_kf=True):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.past_length = past_length
        self.pasts = deque([], maxlen=past_length)
        self.alpha = 0.9
        self.forecasts = []
        self.forecast_score = 0
        self.forecast_index = 0
        self.use_kf = use_kf

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * \
                self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(
            mean_state, self.covariance)
        print(self.mean, self.forecasts_xywh)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        if self.use_kf:
            self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        if self.use_kf:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
            )
        else:
            self._tlwh = new_track.tlwh

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        # store previous xywh
        self.pasts.append([self.frame_id, self.track_id] + list(self.tlbr))
        # self.forecasts = new_track.forecasts

    def update(self, new_track, frame_id, update_feature=True, use_forecast=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        # store previous xywh
        self.pasts.append([self.frame_id, self.track_id] + list(self.tlbr))
        # self.forecasts = new_track.forecasts

        new_tlwh = new_track.tlwh
        if self.use_kf:
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        else:

            # use forecast 
            self._tlwh = new_tlwh
            if len(self.forecasts):
                if use_forecast:
                    self._tlwh = STrack.tlbr_to_tlwh(np.asarray([self.forecasts[self.forecast_index], new_track.tlbr]).mean(axis=0))

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

    @property
    # @jit(nopython=True)
    def forecasts_xywh(self):
        """Get forecasts using dcx, dcy, dw, dy"""
        f = np.array(self.forecasts).reshape(-1, 4)
        f[:, 2:] -= f[:, :2]
        f[:, :2] += f[:, 2:] / 2

        return f

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    # @jit(nopython=True)
    def xywh(self):
        """Convert bounding box to format `(center x, center y, width, height)`
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


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

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

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
    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}
        selected_strack = {}
        # inspect = [14]

        if self.opt.forecast:
            im_blob = [im_blob]
            self.pasts = torch.zeros(
                (self.max_per_image, self.past_length, self.input_size), device=self.opt.device)
            pasts_mask = np.zeros(
                (self.max_per_image, self.past_length), dtype=np.float32)
            # self.pasts_inds = {} #torch.zeros((self.max_per_image), device=self.opt.device)

            ratio = min(float(inp_height) / height, float(inp_width) / width)
            new_shape = (round(width * ratio), round(height * ratio))

            dw = (inp_width - new_shape[0]) / 2  # width padding
            dh = (inp_height - new_shape[1]) / 2  # height padding
            rw = ratio * width / inp_width
            rh = ratio * height / inp_height
            output_h = new_shape[1] // self.opt.down_ratio
            output_w = new_shape[0] // self.opt.down_ratio

            objs_count = 0
            bboxes = []

            strack_pool = list(self.tracked_stracks) #joint_stracks(self.tracked_stracks, self.lost_stracks)
            strack_pool = [t for t in strack_pool if t.time_since_update == 0]
            pasts = np.zeros(
                (self.max_per_image, self.past_length, self.input_size), dtype=np.float32)
            if len(strack_pool) > 0:
                for t in strack_pool:
                    bbox = np.zeros((self.past_length+1, 4), dtype=np.float32)
                    if len(t.pasts) > 1:
                        t_pasts = np.array(t.pasts)
                        t_pasts = t_pasts[::-1] # reverse order
                        bbox[:t_pasts.shape[0], :] = t_pasts[:, 2:]
                        bboxes.append(bbox)
                        pasts_mask[objs_count, :t_pasts.shape[0] - 1] = 1
                        # self.pasts_inds.append(t.track_id)
                        selected_strack[t.track_id] = t
                        objs_count += 1

                if len(bboxes) > 0:
                    bboxes = np.stack(bboxes, axis=0)
                    bbox = bboxes.copy()
                    # bbox[..., [0,2]] /= width
                    # bbox[..., [1,3]] /= height
                    labels = bbox.copy()
                    # labels[..., 0] = ratio * width * (bbox[..., 0] - bbox[..., 2] / 2) + dw
                    # labels[..., 1] = ratio * height * (bbox[..., 1] - bbox[..., 3] / 2) + dh
                    # labels[..., 2] = ratio * width * (bbox[..., 0] + bbox[..., 2] / 2) + dw
                    # labels[..., 3] = ratio * height * (bbox[..., 1] + bbox[..., 3] / 2) + dh

                    labels[..., 0] = ratio * bbox[..., 0] + dw
                    labels[..., 1] = ratio * bbox[..., 1] + dh
                    labels[..., 2] = ratio * bbox[..., 2] + dw
                    labels[..., 3] = ratio * bbox[..., 3] + dh

                    labels = xyxy2xywh(labels.copy())
                    labels[..., [0, 2]] /= inp_width
                    labels[..., [1, 3]] /= inp_height

                    labels[..., [0, 2]] *= output_w
                    labels[..., [1, 3]] *= output_h

                    # flip - oldest first
                    labels = np.flip(labels, 1)
                    # labels_change = np.flip(labels_change, 1)
                    pasts_mask = np.flip(pasts_mask, 1)

                    labels_change = np.diff(labels, axis=1)

                    # bbox_xyxy = bbox.copy()
                    # bbox_xyxy = np.diff(bbox_xyxy, axis=1)
                    # bbox_xyxy[..., [0,2]] *= rw
                    # bbox_xyxy[..., [1,3]] *= rh


                    pasts[:labels_change.shape[0], :, 4:] = labels_change
                    pasts[:labels_change.shape[0], :, :4] = labels[:, 1:, :]
                    # mask = pasts_mask.unsqueeze(-1).expand_as(self.pasts).float()
                    # self.pasts *= mask

                    pasts = pasts * pasts_mask[:, :, np.newaxis]
                    self.pasts = torch.tensor(pasts, device=self.opt.device)

            im_blob += [self.pasts]

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            output = self.model(im_blob)[-1]
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(
                hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()
            pred_futures = None
            if len(selected_strack) > 0:
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

                # pred_futures_xywh = xyxy2xywh(pred_futures)


                # self.post_process(xywh2xyxy(pred_futures), meta)

                # pred_pasts = output['fct'][0]
                # pred_pasts =  pred_pasts.cpu().numpy()
                # pred_pasts = np.flip(pred_pasts, 1)[:objs_count]

                # # get  scores
                # pred_pasts = pred_pasts * pasts_mask[:objs_count, :, np.newaxis]
                # # pasts = pasts * pasts_mask
                # last_bbox = pred_pasts[:,:1, :4]
                # actual = labels.copy()
                # actual = np.flip(actual, 1)

                # actual[:, 1:, :] *= pasts_mask[:objs_count, :, np.newaxis]
                # pred_bboxes = pred_pasts[:,:,4:] + pred_pasts[:,:,:4]

                # pred_bboxes = np.concatenate((last_bbox, pred_bboxes), axis=1)
                # pred_bboxes[..., [0, 2]] /= output_w
                # pred_bboxes[..., [1, 3]] /= output_h
                # pred_bboxes[..., [1, 3]] *= inp_height
                # pred_bboxes[..., [0, 2]] *= inp_width
                # pred_bboxes = xywh2xyxy(pred_bboxes.copy())
                # pred_bboxes[..., [1, 3]] -= dh
                # pred_bboxes[..., [0, 2]] -= dw
                # pred_bboxes /= ratio

                # # actual = xywh2xyxy(actual.copy())
                # actual = bboxes.copy()


                # scores = []

                # for i in range(objs_count):
                #     score = 1 - np.mean([matching.iou_distance(pred_bboxes[i, j:j+1], actual[i, j:j+1]) for j in range(int(pasts_mask[i].sum())+1)])
                #     scores.append(score) 

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]
        # dets_xywh = xyxy2xywh(dets[...,:4])
        # vis
        '''
        os.environ['DISPLAY'] = 'localhost:13.0'
        img = img0.copy()
        for i in range(0, dets.shape[0]):
            bbox = dets[i][0:4]
            cv2.rectangle(img, (bbox[0], bbox[1]),
                          (bbox[2], bbox[3]),
                          (0, 255, 0), 2)
            for j in range(10, forecasts.shape[1] // 4):
                bbox_pred = forecasts[i][j*4: j*4+4]
                cv2.rectangle(img, (bbox_pred[0], bbox_pred[1]),
                          (bbox_pred[2], bbox_pred[3]),
                          (255, 255, j+100), 2)
                cv2.imshow('dets', img)
                cv2.waitKey(1)
        # id0 = id0-1
        # '''
        keys = list(selected_strack.keys())
        if len(selected_strack) > 0:
            for i, tid in enumerate(keys):
                t = selected_strack[tid]
                forecasts = pred_futures[i]
                # if scores[i] < 0.2:
                #     print(t, scores[i], t.forecast_score, t.score, t.forecast_index)
                #     t.forecast_index += 1
                #     continue

                t.forecasts = forecasts
                # t.forecast_score = scores[i]

        if len(dets) > 0:
            '''Detections'''
            # if self.opt.forecast:
            #     detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30, ft) for (tlbrs, f, ft) in zip(dets[:, :5], id_feature, pred_futures)]
            # else:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbrs[:4]), tlbrs[4], f, 30, past_length=self.past_length, use_kf=self.use_kf) for (
                tlbrs, f) in zip(dets[:, :5], id_feature)]
        else:
            detections = []
        detections_copy = list(detections)
        
        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        # for strack in strack_pool:
        # strack.predict()
        u_track, u_detection = range(len(strack_pool)), range(len(detections))
        dists = matching.embedding_distance(strack_pool, detections)
        # dists = matching.iou_distance(strack_pool, detections)
        if self.use_kf:
            STrack.multi_predict(strack_pool)
            dists = matching.fuse_motion(
                self.kalman_filter, dists, strack_pool, detections)
        elif self.forecast:
            #fuse short term motion
            r_tracked_stracks = list(strack_pool)
            dists, forecasts_inds = matching.fuse_motion2(dists, r_tracked_stracks, detections)

        matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=0.4)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if not self.use_kf and self.forecast:
                track.forecast_index = int(forecasts_inds[itracked, idet])
            track.time_since_update = 0

            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        r_tracked_stracks = list(strack_pool)

        ''' Step 3: Second association, with IOU'''

        detections = [detections[i] for i in u_detection]
        r_tracked_stracks = [strack_pool[i] for i in u_track if r_tracked_stracks[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections[idet]
            # if len(track.forecasts):
                # det._tlwh = STrack.tlbr_to_tlwh(np.asarray([track.forecasts[track.time_since_update+1], det.tlbr]).mean(axis=0))
            track.time_since_update = 0
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Use forecast predictions'''
        if self.forecast:
            r_tracked_stracks = [r_tracked_stracks[i] for i in u_track if r_tracked_stracks[i].state == TrackState.Tracked]
            forecasts, inds = get_forecast_distance(r_tracked_stracks, (width, height))
            # forecasts, inds = get_forecast(r_tracked_stracks)
            r_tracks =  [t for i, t in enumerate(r_tracked_stracks) if i not in inds]
            r_tracked_stracks = [r_tracked_stracks[i] for i in inds]
            for track in r_tracks:
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

            dists = matching.iou_distance(r_tracked_stracks, forecasts)
            matches, u_track, _ = matching.linear_assignment(
                dists, thresh=0.5)
            
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = forecasts[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
    
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        # r_tracked_stracks = [r_tracked_stracks[i] for i in u_track ]
        

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [
            t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [
            track for track in self.tracked_stracks if track.is_activated]

        logger.debug('===========Frame {}=========='.format(self.frame_id))
        logger.debug('Activated: {}'.format(
            [track.track_id for track in activated_starcks]))
        logger.debug('Refind: {}'.format(
            [track.track_id for track in refind_stracks]))
        logger.debug('Lost: {}'.format(
            [track.track_id for track in lost_stracks]))
        logger.debug('Removed: {}'.format(
            [track.track_id for track in removed_stracks]))

        viz = False
        # viz = True
        focus = [0, 2, 5, 7, 1, 29]
        if viz and (self.frame_id % 1 == 0) :
            os.environ['DISPLAY'] = 'localhost:11.0'
            cv2.namedWindow("forecasts",cv2.WINDOW_NORMAL)
            img = img0.copy()
            tl = round(0.0004 * max(img.shape[0:2])) + 1
            tf = max(tl - 1, 1)  # font thickness
            colors  = [(0, 255, 251), (0, 255, 0), (0, 0, 255)]

            tracks = output_stracks+ detections_copy  + lost_stracks
            for t in tracks:
                if (t.track_id not in focus):
                    continue

                bbox = t.tlbr
                bbox = [int(v) for v in bbox]
                color = colors[t.state]
                cv2.rectangle(img, (bbox[0], bbox[1]),
                                (bbox[2], bbox[3]),color , 2)
                # color = [random.randint(0, 255) for _ in range(3)]
                
                label = f"{t.track_id}-{ int(t.score * 100)}"
                t_size = cv2.getTextSize(
                        label, 0, fontScale=tl / 3, thickness=tf)[0]
                up = bbox[3] if t.state == 0 else bbox[1]
                c1, c2 = (bbox[0], up
                                ), (bbox[2], bbox[3])
                    
                # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                # cv2.rectangle(img, c1, c2, color, -1)
                cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                            color, thickness=tf, lineType=cv2.LINE_AA)
                color = (255, 255, 0)
                if t.state == 2 and len(t.forecasts):
                    ind =(track.forecast_index + 1) * track.time_since_update
                    bbox_pred = t.forecasts[ind]
                    bbox_pred = [int(v) for v in bbox_pred]
                    cv2.rectangle(
                        img, (bbox_pred[0], bbox_pred[1]), (bbox_pred[2], bbox_pred[3]), color, 2)
                    up = bbox_pred[3] if t.state == 0 else bbox[1]
                    c1, c2 = (bbox_pred[0], up
                                ), (bbox_pred[2], bbox_pred[3])
                    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                            color, thickness=tf, lineType=cv2.LINE_AA)

                    for j in range(0, len(t.forecasts), 5):
                        bbox_pred = t.forecasts[j]
                        bbox_pred = [int(v) for v in bbox_pred]
                        cx, cy = (
                            bbox_pred[0] + bbox_pred[2]) // 2, (bbox_pred[1] + bbox_pred[3]) // 2
                        # cv2.rectangle(img, (bbox_pred[0], bbox_pred[1]), (bbox_pred[2], bbox_pred[3]), (255, 0, j+100), 2)

                        cv2.rectangle(img, (cx, cy), (cx+j, cy+j), color, 2)
            
            
            cv2.imshow('forecasts', img)
            cv2.waitKey(1)
       
        return output_stracks
    
def frame_distance(xywh, img_size):

    width, height = img_size
    center = [width / 2.0, height / 2.0]
    # convert to xywh
    centerx = np.linalg.norm(center[:2])

    xywh = np.array(xywh)
    # calculate the distance of the object with respect to the center of the frame
    dist = np.linalg.norm(xywh[:,:2] - center, axis=1) / centerx

    return  dist
 

# def forecast_track2(track):
#     # print(track, track.time_since_update, frame_id, track.score )
#     pred = copy.deepcopy(track)
#     # futures = track.forecasts
#     # if len(futures) == 0: 
#     #     return pred

#     # if (track.end_frame - track.start_frame) < 10: 
#     #     return pred
#     if track.tracklet_len < 10:
#         return pred

#     # if (track.time_since_update >= 20) or track.score < 0.4: 
#     #     return pred

#     # track.time_since_update += 1
    
#     # tlwh = STrack.tlbr_to_tlwh(futures[track.time_since_update])
#     # pred._tlwh = tlwh
#     return pred

def forecast_track_in_frame(track, img_size=()):
    # print(track, track.time_since_update, frame_id, track.score )

    pred = None

    futures = track.forecasts
    # if len(futures) < 10: 
    #     return pred
    track.time_since_update += 1

    max_threshold = 20
    if (track.tracklet_len + len(track.pasts)) < 15:
        return pred

    forecast_index = (track.forecast_index + 1) * track.time_since_update
    if forecast_index >= len(futures):
        return pred

    tlbr = futures[forecast_index]
    tlwh = STrack.tlbr_to_tlwh(tlbr)
    xywh = tlwh.copy()
    xywh[0] += xywh[2] / 2
    xywh[1] += xywh[3] / 2

    # f = (1-track.forecast_score) * (track.time_since_update)/ max_threshold
    tsu = track.time_since_update/max_threshold 

    lambda_ = 0.5
    if len(img_size):
        f_dist = frame_distance([xywh], img_size)[0]
        f = lambda_  * f_dist + (1-lambda_)  * tsu

    if f > 0.55:
        return pred
    
    
    
    # track.forecast_index += track.time_since_update
    # track.forecast_index   =   forecast_index
    pred = STrack(tlwh, track.score, track.smooth_feat, 30, past_length=track.past_length)
    return pred


def get_forecast_distance(tracks, img_size):
    selected = []
    forecasts = []
    for i, t in enumerate(tracks):
        forecast = forecast_track_in_frame(t, img_size)
        if forecast != None:
            forecasts.append(forecast)
            selected.append(i)

    return forecasts,selected

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb

  