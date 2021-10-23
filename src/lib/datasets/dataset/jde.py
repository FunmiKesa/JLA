import glob
import math
import os
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import numpy as np
import torch
import copy

from torch.utils.data import Dataset
from torchvision.transforms import transforms as T
from cython_bbox import bbox_overlaps as bbox_ious
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.utils import xyxy2xywh, load_txt


class LoadImages:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[
                              1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        img_path = self.files[self.count]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return img_path, img, img0

    def __getitem__(self, idx):
        idx = idx % self.nF
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)
        # TODO: FORECASTING PREVIOUS LABELS
        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        return img_path, img, img0

    def __len__(self):
        return self.nF  # number of files


class LoadImagesAndPasts:  # for inference
    def __init__(self, path, img_size=(1088, 608), past_length=0):
        if os.path.isdir(path):
            image_format = ['.jpg', '.jpeg', '.png', '.tif']
            self.files = sorted(glob.glob('%s/*.*' % path))
            self.files = list(filter(lambda x: os.path.splitext(x)[
                              1].lower() in image_format, self.files))
        elif os.path.isfile(path):
            self.files = [path]

        self.past_length = past_length
        if self.past_length > 0:
            self.forecast_past_files = [
                x.replace('images', 'past').replace('jpg', 'txt').replace('png', 'txt') for x in self.files]

        self.nF = len(self.files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        assert self.nF > 0, 'No images found in ' + path

    def six_dim(self, data, mask):
        inds = data[:, 0]
        data = data[:, 1:]
        n = data.shape[-1] // 4
        data = data.reshape(data.shape[0], n, 4)
        mask = mask[:, 1:].reshape(mask.shape[0], n, 4)

        temp = np.zeros((data.shape[0], n, 6))
        temp[:, :, 2:] = data
        temp[:, :, 1] = inds[:, np.newaxis]
        temp = temp.reshape(-1, 6)
        data = temp
        mask = mask.reshape(-1, 4)
        return data, mask

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration
        return self.get_data(self.count)

    def get_data(self, idx):
        height = self.height
        width = self.width
        img_path = self.files[idx]

        # Read image
        img0 = cv2.imread(img_path)  # BGR
        assert img0 is not None, 'Failed to load ' + img_path

        h, w, _ = img0.shape

        # Padded resize
        img, ratio, dw, dh = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image

        # load pasts data
        pasts_data, p_mask = np.array([]), np.array([])
        if self.past_length > 0:
            forecast_past_path = self.forecast_past_files[self.count]
            if os.path.exists(forecast_past_path):
                column_length = (self.past_length + 1) * 4 + 1
                pasts_data, p_mask = load_txt(
                    forecast_past_path, column_length, max_column=121)
                pasts_data, p_mask = self.six_dim(pasts_data, p_mask)
                
                labels = pasts_data.copy()

                labels[:, [2, 4]] /= w
                labels[:, [3, 5]] /= h

                # Normalized xywh to pixel xyxy format
                pasts_data[:, 2] = ratio * w * \
                    (labels[:, 2] - labels[:, 4] / 2) + dw
                pasts_data[:, 3] = ratio * h * \
                    (labels[:, 3] - labels[:, 5] / 2) + dh
                pasts_data[:, 4] = ratio * w * \
                    (labels[:, 2] + labels[:, 4] / 2) + dw
                pasts_data[:, 5] = ratio * h * \
                    (labels[:, 3] + labels[:, 5] / 2) + dh

                # convert xyxy to xywh
                pasts_data[:, 2:] = xyxy2xywh(
                    pasts_data[:, 2:].copy())  # / height
                pasts_data[:, 2] /= width
                pasts_data[:, 3] /= height
                pasts_data[:, 4] /= width
                pasts_data[:, 5] /= height

                pasts_data = pasts_data.reshape(-1, self.past_length + 1, 6)
                p_mask = p_mask.reshape(-1, self.past_length + 1, 4)[:, 1:, :]
        return img_path, img, img0, pasts_data, p_mask

    def __getitem__(self, idx):
        idx = idx % self.nF
        return self.get_data(idx)

    def __len__(self):
        return self.nF  # number of files


class LoadVideo:  # for inference
    def __init__(self, path, img_size=(1088, 608)):
        self.cap = cv2.VideoCapture(path)
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.width = img_size[0]
        self.height = img_size[1]
        self.count = 0

        self.w, self.h = 1920, 1080
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def get_size(self, vw, vh, dw, dh):
        wa, ha = float(dw) / vw, float(dh) / vh
        a = min(wa, ha)
        return int(vw * a), int(vh * a)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration
        # Read image
        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)
        img0 = cv2.resize(img0, (self.w, self.h))

        # Padded resize
        img, _, _, _ = letterbox(img0, height=self.height, width=self.width)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return self.count, img, img0

    def __len__(self):
        return self.vn  # number of files


class LoadImagesAndLabels:  # for training
    def __init__(self, path, img_size=(1088, 608), augment=False, transforms=None):
        with open(path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [x.replace('\n', '') for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt')
                            for x in self.img_files]

        self.nF = len(self.img_files)  # number of image files
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

    def __getitem__(self, files_index):
        img_path = self.img_files[files_index]
        label_path = self.label_files[files_index]
        return self.get_data(img_path, label_path)

    def get_data(self, img_path, label_path):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = ratio * w * \
                (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] = ratio * h * \
                (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] = ratio * w * \
                (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = ratio * h * \
                (labels0[:, 3] + labels0[:, 5] / 2) + padh
        else:
            labels = np.array([])

        # Augment image and labels
        if self.augment:
            img, labels, M = random_affine(
                img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))

        plotFlag = False
        if plotFlag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T,
                     labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w)

    def __len__(self):
        return self.nF  # number of batches


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    # new_shape = [width, height]
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    # resized, no border
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def random_affine(img, targets=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4

    border = 0  # width of added border (optional)
    height = img.shape[0]
    width = img.shape[1]

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(
        img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * \
        img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * \
        img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) +
                        shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue

    # Return warped points also
    if targets is not None:
        targets, _ = warp_points(targets, M, a)
        return imw, targets, M, a
    else:
        return imw


def warp_points(targets, M, a):
    i = np.ones(targets.shape[0]).astype(bool)

    if len(targets) > 0:
        n = targets.shape[0]
        points = targets[:, 2:6].copy()
        area0 = (points[:, 2] - points[:, 0]) * \
            (points[:, 3] - points[:, 1])

        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # apply angle-based reduction
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)),
                        abs(math.cos(radians))) ** 0.5
        x = (xy[:, 2] + xy[:, 0]) / 2
        y = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate(
            (x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

        # reject warped points outside of image
        #np.clip(xy[:, 0], 0, width, out=xy[:, 0])
        #np.clip(xy[:, 2], 0, width, out=xy[:, 2])
        #np.clip(xy[:, 1], 0, height, out=xy[:, 1])
        #np.clip(xy[:, 3], 0, height, out=xy[:, 3])
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]
        area = w * h
        ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

        targets = targets[i]
        targets[:, 2:6] = xy[i]

    return targets, i


def collate_fn(batch):
    imgs, labels, paths, sizes = zip(*batch)
    batch_size = len(labels)
    imgs = torch.stack(imgs, 0)
    max_box_len = max([l.shape[0] for l in labels])
    labels = [torch.from_numpy(l) for l in labels]
    filled_labels = torch.zeros(batch_size, max_box_len, 6)
    labels_len = torch.zeros(batch_size)

    for i in range(batch_size):
        isize = labels[i].shape[0]
        if len(labels[i]) > 0:
            filled_labels[i, :isize, :] = labels[i]
        labels_len[i] = isize

    return imgs, filled_labels, paths, sizes, labels_len.unsqueeze(1)


class JointDataset(LoadImagesAndLabels):  # for training
    default_resolution = [1088, 608]
    mean = None
    std = None
    num_classes = 1

    def __init__(self, opt, root, paths, img_size=(1088, 608), augment=False, transforms=None):
        self.opt = opt
        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        self.num_classes = 1

        self.forecast = opt.forecast
        if self.forecast:
            self.past_length = self.forecast['past_length']
            self.future_length = self.forecast['future_length']
            self.fixed_length = self.forecast['fixed_length']
            self.hidden_size = self.forecast['hidden_size']
            self.input_size = self.forecast['input_size']
            self.output_size = self.forecast['output_size']
            self.race = 0

            self.forecast_future_files = OrderedDict()
            self.forecast_past_files = OrderedDict()

        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [
                    osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(
                    filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace(
                    '.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

            if self.forecast:
                self.forecast_future_files[ds] = [
                    x.replace('labels_with_ids', 'future') for x in self.label_files[ds]]
                self.forecast_past_files[ds] = [
                    x.replace('labels_with_ids', 'past') for x in self.label_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.max_objs = opt.K
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')

        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def six_dim(self, data, mask):
        inds = data[:, 0]
        data = data[:, 1:]
        n = data.shape[-1] // 4
        data = data.reshape(data.shape[0], n, 4)
        mask = mask[:, 1:].reshape(mask.shape[0], n, 4)

        temp = np.zeros((data.shape[0], n, 6))
        temp[:, :, 2:] = data
        temp[:, :, 1] = inds[:, np.newaxis]
        temp = temp.reshape(-1, 6)
        data = temp
        mask = mask.reshape(-1, 4)
        return data, mask

    def get_file_path(self, files_index):
        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]
        return img_path, label_path, ds, start_index

    def __getitem__(self, files_index):

        img_path, label_path, ds, start_index = self.get_file_path(files_index)
        futures_data, pasts_data = [], []
        if self.forecast:
            # initialization
            forecast_future_path = self.forecast_future_files[ds][files_index - start_index]
            forecast_past_path = self.forecast_past_files[ds][files_index - start_index]

            futures = np.zeros(
                (self.max_objs, self.future_length, self.output_size), dtype=np.float32)
            pasts = np.zeros((self.max_objs, self.past_length,
                              self.input_size), dtype=np.float32)
            futures_mask = np.zeros(
                (self.max_objs, self.future_length, self.output_size), dtype=np.uint8)
            futures_inds = np.zeros((self.max_objs), dtype=np.int64)
            pasts_mask = np.zeros(
                (self.max_objs, self.past_length, self.input_size), dtype=np.uint8)
            pasts_inds = np.zeros((self.max_objs), dtype=np.int64)

            if os.path.exists(forecast_past_path) and os.path.exists(forecast_future_path):
                column_length = (self.past_length + 1) * 4 + 1
                pasts_data, p_mask = load_txt(
                    forecast_past_path, column_length, max_column=121)
                pasts_data, p_mask = self.six_dim(pasts_data, p_mask)

                # if self.fixed_length:
                #     print(forecast_past_path, p_mask.shape,p_mask.sum())

                column_length = (self.future_length) * 4 + 1
                futures_data, f_mask = load_txt(
                    forecast_future_path, column_length, max_column=361)

                futures_data, f_mask = self.six_dim(futures_data, f_mask)

        imgs, labels, img_path, (input_h, input_w), (f_data, f_data_mask), (p_data, p_data_mask) = self.get_data(
            img_path, label_path, futures_data, pasts_data)

        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        output_h = imgs.shape[1] // self.opt.down_ratio
        output_w = imgs.shape[2] // self.opt.down_ratio
        num_classes = self.num_classes
        num_objs = labels.shape[0]
        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        if self.opt.ltrb:
            wh = np.zeros((self.max_objs, 4), dtype=np.float32)
        else:
            wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs, ), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs, ), dtype=np.uint8)
        ids = np.zeros((self.max_objs, ), dtype=np.int64)
        bbox_xys = np.zeros((self.max_objs, 4), dtype=np.float32)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else draw_umich_gaussian
        for k in range(num_objs):
            label = labels[k]
            bbox = label[2:]
            cls_id = int(label[0])
            bbox[[0, 2]] = bbox[[0, 2]] * output_w
            bbox[[1, 3]] = bbox[[1, 3]] * output_h
            bbox_amodal = copy.deepcopy(bbox)
            bbox_amodal[0] = bbox_amodal[0] - bbox_amodal[2] / 2.
            bbox_amodal[1] = bbox_amodal[1] - bbox_amodal[3] / 2.
            bbox_amodal[2] = bbox_amodal[0] + bbox_amodal[2]
            bbox_amodal[3] = bbox_amodal[1] + bbox_amodal[3]
            bbox[0] = np.clip(bbox[0], 0, output_w - 1)
            bbox[1] = np.clip(bbox[1], 0, output_h - 1)
            h = bbox[3]
            w = bbox[2]

            bbox_xy = copy.deepcopy(bbox)
            bbox_xy[0] = bbox_xy[0] - bbox_xy[2] / 2
            bbox_xy[1] = bbox_xy[1] - bbox_xy[3] / 2
            bbox_xy[2] = bbox_xy[0] + bbox_xy[2]
            bbox_xy[3] = bbox_xy[1] + bbox_xy[3]

            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = 6 if self.opt.mse_loss else radius
                #radius = max(1, int(radius)) if self.opt.mse_loss else radius
                ct = np.array(
                    [bbox[0], bbox[1]], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                if self.opt.ltrb:
                    wh[k] = ct[0] - bbox_amodal[0], ct[1] - bbox_amodal[1], \
                        bbox_amodal[2] - ct[0], bbox_amodal[3] - ct[1]
                else:
                    wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                ids[k] = label[1]
                bbox_xys[k] = bbox_xy

        if self.forecast:
            if len(f_data):
                futures_data[f_data_mask] = f_data
                futures_data[f_data_mask == False] = 0

                futures_data = futures_data.reshape(-1, self.future_length, 6)
                labels = futures_data.copy()[..., 2:]
                inds = futures_data[..., 0, 1]
                labels[..., [0, 2]] *= output_w
                labels[..., [1, 3]] *= output_h

                futures[:labels.shape[0], ...] = labels
                f_mask = f_mask.reshape(-1, self.future_length, 4)
                futures_mask[:f_mask.shape[0], :] = f_mask
                futures_inds[:f_mask.shape[0]] = inds

                futures = futures * futures_mask
                futures = futures.astype(np.float32)
                futures_mask = futures_mask.astype(np.uint8)

            if len(p_data):
                pasts_data[p_data_mask] = p_data
                pasts_data[p_data_mask == False] = 0

                pasts_data = pasts_data.reshape(-1, self.past_length + 1, 6)
                p_mask = p_mask.reshape(-1, self.past_length + 1, 4)[:, 1:, :]

                labels = pasts_data.copy()[..., 2:]
                inds = pasts_data[:, 0, 1]
                labels[..., [0, 2]] *= output_w
                labels[..., [1, 3]] *= output_h

                # flip - oldest first
                labels = np.flip(labels, 1)
                mask = np.flip(p_mask, 1)

                labels_change = np.diff(labels, axis=1)

                labels = labels[:, 1:, :]

                pasts[:labels_change.shape[0], :, 4:] = labels_change
                pasts[:labels_change.shape[0], :, :4] = labels

                pasts_mask[:mask.shape[0], :, :4] = mask
                pasts_mask[:mask.shape[0], :,  4:] = mask
                pasts_inds[:mask.shape[0]] = inds

                pasts = pasts * pasts_mask
                pasts = pasts.astype(np.float32)
                pasts_mask = pasts_mask.astype(np.uint8)

            ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask,
                   'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys, 'futures': futures, 'futures_mask': futures_mask, 'futures_inds': futures_inds, 'pasts': pasts, 'pasts_mask': pasts_mask, 'pasts_inds': pasts_inds}
        else:
            ret = {'input': imgs, 'hm': hm, 'reg_mask': reg_mask,
                   'ind': ind, 'wh': wh, 'reg': reg, 'ids': ids, 'bbox': bbox_xys}

        return ret

    def get_data(self, img_path, label_path, futures_data=[], pasts_data=[]):
        height = self.height
        width = self.width
        img = cv2.imread(img_path)  # BGR
        if img is None:
            raise ValueError('File corrupt {}'.format(img_path))
        augment_hsv = True
        if self.augment and augment_hsv:
            # SV augmentation by 50%
            fraction = 0.50
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            S = img_hsv[:, :, 1].astype(np.float32)
            V = img_hsv[:, :, 2].astype(np.float32)

            a = (random.random() * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            a = (random.random() * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            img_hsv[:, :, 1] = S.astype(np.uint8)
            img_hsv[:, :, 2] = V.astype(np.uint8)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)

        h, w, _ = img.shape
        img, ratio, padw, padh = letterbox(img, height=height, width=width)

        labels = np.array([])
        labels_f = np.array(futures_data)
        labels_p = np.array(pasts_data)
        labels_f_mask = np.array([])
        labels_p_mask = np.array([])

        # Load labels
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

            # Normalized xywh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = ratio * w * \
                (labels0[:, 2] - labels0[:, 4] / 2) + padw
            labels[:, 3] = ratio * h * \
                (labels0[:, 3] - labels0[:, 5] / 2) + padh
            labels[:, 4] = ratio * w * \
                (labels0[:, 2] + labels0[:, 4] / 2) + padw
            labels[:, 5] = ratio * h * \
                (labels0[:, 3] + labels0[:, 5] / 2) + padh

        if len(futures_data):
            # Normalized xywh to pixel xyxy format
            labels_f[:, 2] = ratio * \
                (futures_data[:, 2] - futures_data[:, 4] / 2) + padw
            labels_f[:, 3] = ratio * \
                (futures_data[:, 3] - futures_data[:, 5] / 2) + padh
            labels_f[:, 4] = ratio * \
                (futures_data[:, 2] + futures_data[:, 4] / 2) + padw
            labels_f[:, 5] = ratio * \
                (futures_data[:, 3] + futures_data[:, 5] / 2) + padh

            labels_f_mask = np.ones(labels_f.shape[0]).astype(bool)

        if len(pasts_data):
            # Normalized xywh to pixel xyxy format
            labels_p[:, 2] = ratio * \
                (pasts_data[:, 2] - pasts_data[:, 4] / 2) + padw
            labels_p[:, 3] = ratio * \
                (pasts_data[:, 3] - pasts_data[:, 5] / 2) + padh
            labels_p[:, 4] = ratio * \
                (pasts_data[:, 2] + pasts_data[:, 4] / 2) + padw
            labels_p[:, 5] = ratio * \
                (pasts_data[:, 3] + pasts_data[:, 5] / 2) + padh

            labels_p_mask = np.ones(labels_p.shape[0]).astype(bool)

        # Augment image and labels
        if self.augment:
            img, labels, M, a = random_affine(
                img, labels, degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))
            
            if len(futures_data):
                labels_f, labels_f_mask = warp_points(labels_f, M, a)
            
            if len(pasts_data):
                labels_p, labels_p_mask = warp_points(labels_p, M, a)

        if len(futures_data) > 0:
            # convert xyxy to xywh
            labels_f[:, 2:6] = xyxy2xywh(
                labels_f[:, 2:6].copy())  # / height
            labels_f[:, 2] /= width
            labels_f[:, 3] /= height
            labels_f[:, 4] /= width
            labels_f[:, 5] /= height

        if len(pasts_data) > 0:
            # convert xyxy to xywh
            labels_p[:, 2:6] = xyxy2xywh(
                labels_p[:, 2:6].copy())  # / height
            labels_p[:, 2] /= width
            labels_p[:, 3] /= height
            labels_p[:, 4] /= width
            labels_p[:, 5] /= height

        plotFlag = False
        if plotFlag:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(50, 50))
            plt.imshow(img[:, :, ::-1])
            plt.plot(labels[:, [1, 3, 3, 1, 1]].T,
                     labels[:, [2, 2, 4, 4, 2]].T, '.-')
            plt.axis('off')
            plt.savefig('test.jpg')
            time.sleep(10)

        nL = len(labels)
        if nL > 0:
            # convert xyxy to xywh
            labels[:, 2:6] = xyxy2xywh(labels[:, 2:6].copy())  # / height
            labels[:, 2] /= width
            labels[:, 3] /= height
            labels[:, 4] /= width
            labels[:, 5] /= height
        if self.augment:
            # random left-right flip
            lr_flip = True
            if lr_flip & (random.random() > 0.5):
                img = np.fliplr(img)
                if nL > 0:
                    labels[:, 2] = 1 - labels[:, 2]
                    if len(futures_data):
                        labels_f[:, 2] = 1 - labels_f[:, 2]
                    if len(pasts_data):
                        labels_p[:, 2] = 1 - labels_p[:, 2]

        img = np.ascontiguousarray(img[:, :, ::-1])  # BGR to RGB

        if self.transforms is not None:
            img = self.transforms(img)

        return img, labels, img_path, (h, w), (labels_f, labels_f_mask), (labels_p, labels_p_mask)


class DetDataset(LoadImagesAndLabels):  # for training
    def __init__(self, root, paths, img_size=(1088, 608), augment=False, transforms=None):

        dataset_names = paths.keys()
        self.img_files = OrderedDict()
        self.label_files = OrderedDict()
        self.tid_num = OrderedDict()
        self.tid_start_index = OrderedDict()
        for ds, path in paths.items():
            with open(path, 'r') as file:
                self.img_files[ds] = file.readlines()
                self.img_files[ds] = [
                    osp.join(root, x.strip()) for x in self.img_files[ds]]
                self.img_files[ds] = list(
                    filter(lambda x: len(x) > 0, self.img_files[ds]))

            self.label_files[ds] = [
                x.replace('images', 'labels_with_ids').replace(
                    '.png', '.txt').replace('.jpg', '.txt')
                for x in self.img_files[ds]]

        for ds, label_paths in self.label_files.items():
            max_index = -1
            for lp in label_paths:
                lb = np.loadtxt(lp)
                if len(lb) < 1:
                    continue
                if len(lb.shape) < 2:
                    img_max = lb[1]
                else:
                    img_max = np.max(lb[:, 1])
                if img_max > max_index:
                    max_index = img_max
            self.tid_num[ds] = max_index + 1

        last_index = 0
        for i, (k, v) in enumerate(self.tid_num.items()):
            self.tid_start_index[k] = last_index
            last_index += v

        self.nID = int(last_index + 1)
        self.nds = [len(x) for x in self.img_files.values()]
        self.cds = [sum(self.nds[:i]) for i in range(len(self.nds))]
        self.nF = sum(self.nds)
        self.width = img_size[0]
        self.height = img_size[1]
        self.augment = augment
        self.transforms = transforms

        print('=' * 80)
        print('dataset summary')
        print(self.tid_num)
        print('total # identities:', self.nID)
        print('start index')
        print(self.tid_start_index)
        print('=' * 80)

    def __getitem__(self, files_index):

        for i, c in enumerate(self.cds):
            if files_index >= c:
                ds = list(self.label_files.keys())[i]
                start_index = c

        img_path = self.img_files[ds][files_index - start_index]
        label_path = self.label_files[ds][files_index - start_index]
        if os.path.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 6)

        imgs, labels, img_path, (h, w) = self.get_data(img_path, label_path)
        for i, _ in enumerate(labels):
            if labels[i, 1] > -1:
                labels[i, 1] += self.tid_start_index[ds]

        return imgs, labels0, img_path, (h, w)
