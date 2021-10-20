import numpy as np


def calc_fde(outputs, targets, return_mean=True):
    '''
    Calculates the final displacement between outputs and
    targets centroids
    Args:
        outputs: np array. 3D array format (trajectory) x (timestep) x (cxcywh)
        targets: np array. 3D array format (trajectory) x (timestep) x (cxcywh)
    Returns:
        float: Final displacement error between outputs and targets
    '''
    # out_mid_xs = outputs[:, -1, 0]
    # out_mid_ys = outputs[:, -1, 1]

    # tar_mid_xs = targets[:, -1, 0]
    # tar_mid_ys = targets[:, -1, 1]
    # diff = ((out_mid_xs - tar_mid_xs) * (out_mid_xs - tar_mid_xs)) + \
    #     ((out_mid_ys - tar_mid_ys) * (out_mid_ys - tar_mid_ys))

    outputs = outputs[:, -1, :2]
    targets = targets[:, -1, :2]

    diff = (outputs - targets) * (outputs - targets)
    if return_mean:
        return float(np.mean(np.sqrt(np.mean(diff, axis=1))))
    else:
        return np.sqrt(np.mean(diff, axis=1))


def calc_ade(outputs, targets, return_mean=True):
    '''
    Calculates the average displacement between outputs and
    targets centroids
    Args:
        outputs: np array. 3D array format (trajectory) x (timestep) x (cxcywh)
        targets: np array. 3D array format (trajectory) x (timestep) x (cxcywh)
    Returns:
        float: Average IOU between outputs and targets
    '''
    out_mid_xs = outputs[..., 0]
    out_mid_ys = outputs[..., 1]

    tar_mid_xs = targets[..., 0]
    tar_mid_ys = targets[..., 1]

    diff = ((out_mid_xs - tar_mid_xs) * (out_mid_xs - tar_mid_xs)) + \
        ((out_mid_ys - tar_mid_ys) * (out_mid_ys - tar_mid_ys))

    if return_mean:
        return np.mean(np.sqrt(np.mean(diff, axis=1)))
    else:
        return np.sqrt(np.mean(diff, axis=1))


def calc_fiou(outputs, targets, return_mean=True):
    '''
    Calculates the final IOU between outputs and
    targets
    Args:
        outputs: np array. 3D array format (trajectory) x (timestep) x (cxcywh)
        targets: np array. 3D array format (trajectory) x (timestep) x (cxcywh)
    Returns:
        float: Average final IOU between outputs and targets
    '''
    final_outputs = outputs[:, -1, :]
    final_targets = targets[:, -1, :]
    fiou = get_iou(final_outputs, final_targets)
    return float(np.mean(fiou)) if return_mean else fiou


def calc_aiou(outputs, targets, return_mean=True):
    '''
    Calculates the average IOU between outputs and
    targets
    Args:
        outputs: np array. 3D array format (trajectory) x (timestep) x (cxcywh)
        targets: np array. 3D array format (trajectory) x (timestep) x (cxcywh)
    Returns:
        float: Average IOU between outputs and targets
    '''
    ious = np.zeros((targets.shape[0], targets.shape[1]))
    for t in range(targets.shape[1]):
        t_outputs = outputs[:, t, :]
        t_targets = targets[:, t, :]
        t_iou = get_iou(t_outputs, t_targets)
        ious[:, t] = t_iou

    if return_mean:
        return float(np.mean(np.mean(ious, axis=1)))
    else:
        return np.mean(ious, axis=1)


def get_iou(bboxes1, bboxes2):
    """
    Adapted from https://gist.github.com/zacharybell/8d9b1b25749fe6494511f843361bb167
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array,  list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        np array: intersection-over-onion of bboxes1 and bboxes2
    """
    ious = []
    bboxes1 = xywh2xyxy(bboxes1.copy())
    bboxes2 = xywh2xyxy(bboxes2.copy())
    for bbox1, bbox2 in zip(bboxes1, bboxes2):
        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]
        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2
        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)
        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            ious.append(0)
            continue
        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * \
            (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection
        iou = size_intersection / size_union
        ious.append(iou)
    return ious


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = np.zeros(x.shape)
    y[..., 0] = (x[..., 0] - x[..., 2] / 2)
    y[..., 1] = (x[..., 1] - x[..., 3] / 2)
    y[..., 2] = (x[..., 0] + x[..., 2] / 2)
    y[..., 3] = (x[..., 1] + x[..., 3] / 2)
    return y
