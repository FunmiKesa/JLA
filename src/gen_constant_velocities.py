import os.path as osp
import os
import numpy as np
import pandas as pd
import glob
import cv2
from utils import *


def gen_cv_files(seq_label_root, cv_label_root, future_length=60, past_length=10, img_size=None):
    if not osp.exists(seq_label_root):
        print(f"{seq_label_root} not found!")
        raise Exception(f'{seq_label_root} does not exist.')

    mkdirs(cv_label_root, True)
    bboxes = []
    label_file_paths = {}
    fid = 0
    for filepath in sorted(glob.glob(seq_label_root + "/*.txt")):
        fname = filepath.split('/')[-1]
        fid += 1
        if fid not in label_file_paths:
            label_fpath = osp.join(cv_label_root, fname)
            label_file_paths[fid] = label_fpath
        else:
            print(
                "Something is not right! Frame id should be unique. Please check the source code and data.")

        bbox = np.loadtxt(filepath, dtype=np.float64)
        if not img_size:
            # get image
            img_filepath = filepath.replace('labels_with_ids', 'images')
            if osp.exists(img_filepath.replace('.txt', '.png')):
                image_file = img_filepath.replace('.txt', '.png')
            elif osp.exists(img_filepath.replace('.txt', '.jpg')):
                image_file = img_filepath.replace('.txt', '.jpg')
            else:
                continue

            img = cv2.imread(image_file)
            img_size = img.shape[:2]
        seq_height, seq_width = img_size
        if len(bbox.shape) == 1:
            bbox = bbox.reshape(1, bbox.shape[0])
        # convert the original size
        bbox[:, [2, 4]] *= seq_width
        bbox[:, [3, 5]] *= seq_height
        bbox[:, 0] = fid
        bboxes += [bbox]

    bboxes = np.concatenate(bboxes)
    df = pd.DataFrame(bboxes, columns=['fid', 'tid', 'x', 'y', 'w', 'h'])

    # group by frame
    groups = df.groupby(['tid'])

    for tid, group in groups:
        if tid == -1:
            continue

        group = group.groupby('fid', as_index=False).mean()
        fids = group.fid.unique()

        # compute constant velocity
        group_reverse = group.iloc[::-1]
        fids_reverse = fids[::-1]
        cv = {}
        for i, c in chunker1(group_reverse, past_length, 0):
            if c.empty:
                continue
            fid = int(fids_reverse[i])
            prev = c.iloc[0][['x', 'y', 'w', 'h']]
            prev_m = c.iloc[-1][['x', 'y', 'w', 'h']]
            m = c.shape[0]
            cv = (prev - prev_m) / m
            v = [
                f"{prev.x + (cv.x * n)} {prev.y + (cv.y * n)} {prev.w + (cv.w * n)} {prev.h + (cv.h * n)}" for n in range(future_length)]

            label_str = f"{int(tid)} {' '.join(v)}\n"
            label_fpath = label_file_paths[fid]
            with open(label_fpath, 'a+') as f:
                f.write(label_str)


if __name__ == "__main__":
    datasets = ["PRW", "Caltech", "MOT15", "MOT16", "MOT17", "MOT20"]

    cv_label = 'cv_10'
    for d in datasets:
        try:
            print("\n", d)
            seq_label = ''

            if 'MOT' in d:
                seq_label = 'img1'
                seq_root = f'data/{d}/images/train'
                label_root = f'data/{d}/labels_with_ids/train'

                seqs = [s for s in os.listdir(seq_root)]
                for seq in seqs:
                    print(seq)
                    img_size = None
                    # seq_info_file = osp.join(seq_root, seq, 'seqinfo.ini')
                    # if osp.exists(seq_info_file):
                    #     seq_info = open(
                    #         osp.join(seq_root, seq, 'seqinfo.ini')).read()
                    #     seq_width = int(seq_info[seq_info.find(
                    #         'imWidth=') + 8:seq_info.find('\nimHeight')])
                    #     seq_height = int(seq_info[seq_info.find(
                    #         'imHeight=') + 9:seq_info.find('\nimExt')])

                    #     img_size = (seq_height, seq_width)

                    seq_label_root = osp.join(label_root, seq, seq_label)
                    cv_label_root = seq_label_root.replace(
                        'labels_with_ids', cv_label)

                    gen_cv_files(seq_label_root, cv_label_root,
                                 img_size=img_size)

            elif 'Caltech' in d:
                seq_root = f'data/{d}/data/images'
                label_root = f'data/{d}/data/labels_with_ids'

                cv_label_root = label_root.replace(
                    'labels_with_ids', cv_label)

                gen_cv_files(label_root, cv_label_root,
                             img_size=(480, 640))
            elif 'PRW' in d:
                seq_root = f'data/{d}/images'
                label_root = f'data/{d}/labels_with_ids'

                cv_label_root = label_root.replace(
                    'labels_with_ids', cv_label)

                gen_cv_files(label_root, cv_label_root,
                             img_size=(1080, 1920))
            else:
                print('Data format not known.')

        except Exception as ex:
            print(d, ' failed due to ', ex)
