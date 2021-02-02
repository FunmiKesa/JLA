import os.path as osp
import os
import numpy as np
import pandas as pd
import glob
import shutil
import cv2


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def chunker1(seq, size, start_index):
    return ((pos, seq.iloc[pos+start_index:pos + size + start_index]) for pos in range(0, len(seq)))


def gen_future_files(seq_label_root, future_label_root, future_length=60, img_size=None):
    if os.path.exists(future_label_root):
        shutil.rmtree(future_label_root)
    mkdirs(future_label_root)
    bboxes = []
    label_file_paths = {}
    fid = 0
    for filepath in sorted(glob.glob(seq_label_root + "/*.txt")):
        fname = filepath.split('/')[-1]
        fid += 1
        if fid not in label_file_paths:
            label_fpath = osp.join(future_label_root, fname)
            label_file_paths[fid] = label_fpath
        else:
            print(
                "Something is not right! Frame id should be unique. Please check the source code and data.")

        bbox = np.loadtxt(filepath, dtype=np.float64)
        if len(bbox) == 0:
            continue
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
            print(img_size, filepath)
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
        # print(tid)
        if tid == -1:
            continue
        group = group.groupby('fid', as_index=False).mean()
        fids = group.fid.unique()
        group['cord'] = group.apply(
            lambda row: f"{row.x} {row.y} {row.w} {row.h}", axis=1)

        # compute future
        for i, c in chunker1(group, future_length, 0):
            v = c.reset_index().pivot(
                index='tid', columns=['index'], values='cord')
            if v.empty:
                continue
            label_str = f"{int(tid)} {' '.join(v.iloc[0])}\n"

            fid = int(fids[i])
            label_fpath = label_file_paths[fid]
            with open(label_fpath, 'a+') as f:
                f.write(label_str)


def main(seq_root, label_root, future_length=60, seq_label="img1", future_label="future"):
    seqs = [s for s in os.listdir(seq_root)]
    for seq in seqs:
        print(seq)
        seq_label_root = osp.join(label_root, seq, seq_label)
        future_label_root = seq_label_root.replace(
            'labels_with_ids', future_label)

        if os.path.exists(future_label_root):
            shutil.rmtree(future_label_root)

        mkdirs(future_label_root)

        bboxes = []
        for filename in sorted(glob.glob(seq_label_root + "/*.txt")):
            fid = int(filename.split('/')[-1].replace('.txt', ''))
            bbox = np.loadtxt(filename, dtype=np.float64)
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
            fids = group.fid.unique()
            group['cord'] = group.apply(
                lambda row: f"{row.x} {row.y} {row.w} {row.h}", axis=1)

            # compute future
            for i, c in chunker1(group, future_length, 0):
                v = c.reset_index().pivot(
                    index='tid', columns=['index'], values='cord')
                if v.empty:
                    continue
                label_str = f"{int(tid)} {' '.join(v.iloc[0])}\n"

                fid = int(fids[i])
                label_fpath = osp.join(
                    future_label_root, '{:06d}.txt'.format(fid))
                with open(label_fpath, 'a+') as f:
                    f.write(label_str)


if __name__ == "__main__":
    datasets = ["CUHKSYSU", "Caltech"]
    # datasets = ["PRW", "CUHKSYSU", "Caltech", "MOT15", "MOT16", "MOT17", "MOT20"]
    future_label = 'future'
    future_length = 60
    for d in datasets:
        try:
            print("\n", d)
            seq_label = ''

            if 'MOT' in d:
                seq_label = 'img1'
                seq_root = f'data/{d}/images/train'
                label_root = f'data/{d}/labels_with_ids/train'

                if not osp.exists(seq_root):
                    print(f"{seq_root} not found!")
                    continue

                seqs = [s for s in os.listdir(seq_root)]
                for seq in seqs:
                    print(seq)
                    img_size = None
                    seq_info_file = osp.join(seq_root, seq, 'seqinfo.ini')
                    if osp.exists(seq_info_file):
                        seq_info = open(
                            osp.join(seq_root, seq, 'seqinfo.ini')).read()
                        seq_width = int(seq_info[seq_info.find(
                            'imWidth=') + 8:seq_info.find('\nimHeight')])
                        seq_height = int(seq_info[seq_info.find(
                            'imHeight=') + 9:seq_info.find('\nimExt')])

                        img_size = (seq_height, seq_width)

                    # main(seq_root, label_root, seq_label=seq_label)
                    seq_label_root = osp.join(label_root, seq, seq_label)
                    future_label_root = seq_label_root.replace(
                        'labels_with_ids', future_label)

                    # if osp.exists(future_label_root):
                    #     print(f"{future_label_root} exists!")
                    #     continue

                    gen_future_files(seq_label_root, future_label_root,
                                     future_length, img_size)

            elif 'Caltech' in d:
                seq_root = f'data/{d}/data/images'
                label_root = f'data/{d}/data/labels_with_ids'
                img_size = (480, 640)

                future_label_root = label_root.replace(
                    'labels_with_ids', future_label)

                gen_future_files(label_root, future_label_root, future_length,
                                 img_size)

            elif 'PRW' in d:
                seq_root = f'data/{d}/images'
                label_root = f'data/{d}/labels_with_ids'

                # def extract_fid(x):
                #     global values
                #     xarr = x.split('_')
                #     seq = f"{xarr[-2].replace('c','').replace('s','')}{xarr[-1]}"
                #     fid = int(seq)
                #     if fid in values:
                #         print('Duplicate!!!',fid, seq, x)
                #         raise Exception()

                #     values += [fid]

                #     return fid
                img_size = (1080, 1920)
                future_label_root = label_root.replace(
                    'labels_with_ids', future_label)

                gen_future_files(label_root, future_label_root, future_length,
                                 img_size)

            elif 'CUHKSYSU' in d:
                seq_root = f'data/{d}/images'
                label_root = f'data/{d}/labels_with_ids'

                # def extract_fid(x):
                #     return int(x.replace('s', ''))

                future_label_root = label_root.replace(
                    'labels_with_ids', future_label)

                img_size = (800, 600)

                gen_future_files(label_root, future_label_root, future_length,
                                 img_size)

            else:
                print('Data format not known.')

        except Exception as ex:
            print(d, ' failed due to ', ex)
