import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracking_utils import kalman_filter
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix

def fuse_motion2(cost_matrix, tracks, detections, lambda_=0.75, max_length=10):
    forecasts_inds = np.zeros((len(tracks), len(detections)), dtype=np.int)

    if cost_matrix.size == 0:
        return cost_matrix, forecasts_inds
    dists = iou_distance(tracks, detections)
    dets = np.array([d.tlbr for d in detections])
    for row, track in enumerate(tracks):
        d = dists[row]
        if (len(track.forecasts) > 0):
            c=max_length+track.time_since_update+track.forecast_index
            forecasts = track.forecasts[:c]
            f_dists = iou_distance(forecasts, dets)
            m = f_dists.argmin(axis=0)
            v = f_dists.min(axis=0)
            i = np.argmax(f_dists < 0.5, axis=0)
            d *= v
            forecasts_inds[row] = i
        # if track.forecast_score > 0.5:
        #     print(track, track.forecast_score, track.score, track.time_since_update, "\nd: ", d, "\ncost: ", cost_matrix[row])

        cost_matrix[row, d>=1] *= 2
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * d
    return cost_matrix, forecasts_inds

def fuse_motion2_(cost_matrix, tracks, detections, lambda_=0.75, max_length=20):
    forecasts_inds = np.zeros((len(tracks), len(detections)), dtype=np.int)

    if cost_matrix.size == 0:
        return cost_matrix, forecasts_inds
    dists = iou_distance(tracks, detections)
    dets = np.array([d.tlbr for d in detections])
    # dets = np.array([d.xywh[:2] for d in detections])

    # trks = np.array([t.tlbr for t in tracks])
    # print(dists)
    for row, track in enumerate(tracks):
        d = dists[row]

        if len(track.forecasts) > 0:
            forecasts = track.forecasts[:max_length]
            f_dists = iou_distance(forecasts, dets)
            i = np.argmax(f_dists < 0.5, axis=0)
            v = f_dists.T[np.arange(f_dists.shape[1]), i.squeeze()]
            d = (v * max_length / (max_length - i))  * dists[row]
            # d = 0.5 * (d + v) * (max_length/(max_length - i))
            forecasts_inds[row] = i

        cost_matrix[row, d >= 1] *= 2
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * d
    return cost_matrix, forecasts_inds


def normalized_euclidean_distance(atracks, btracks):

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        axywhs = atracks
        bxywhs = btracks
    else:
        axywhs = np.array([track.xywh for track in atracks])
        bxywhs = np.array([track.xywh for track in btracks])
    
    dists = np.ones((len(axywhs), len(bxywhs)), dtype=np.float)
    if dists.size == 0:
        return dists
    
    for i in range(len(axywhs)):
        axywh = axywhs[i][:2]
        bxywh = bxywhs[i][:2]
    
        dist = 0.5 * ((axywh - bxywh).var( keepdims=True) / (axywh.var(keepdims=True) + bxywh.var( keepdims=True) + 1e-8))
        dists[i,i] = dist ** 0.5

    return dists
