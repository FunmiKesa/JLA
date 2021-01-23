import os.path as osp
import os
import numpy as np
import pandas as pd
import glob
import shutil


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def chunker1(seq, size, start_index):
    return ((pos, seq.iloc[pos+start_index:pos + size + start_index]) for pos in range(0, len(seq)))


def main(seq_root, label_root, future_length=60, past_length=10, seq_label="img1", cv_label="cv_10"):
    seqs = [s for s in os.listdir(seq_root)]
    for seq in seqs:
        print(seq)
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find(
            'imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find(
            'imHeight=') + 9:seq_info.find('\nimExt')])

        seq_label_root = osp.join(label_root, seq, seq_label)
        cv_label_root = osp.join(label_root, seq, cv_label)

        if os.path.exists(cv_label_root):
            shutil.rmtree(cv_label_root)

        mkdirs(cv_label_root)

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

            # compute constant velocity
            group_reverse = group.iloc[::-1]
            fids_reverse = fids[::-1]
            cv = {}
            for i, c in chunker1(group_reverse, past_length, 1):
                if c.empty:
                    continue
                fid = int(fids_reverse[i])
                prev = c.iloc[0][['x', 'y', 'w', 'h']]
                prev_m = c.iloc[-1][['x', 'y', 'w', 'h']]
                m = c.shape[0]
                cv = (prev - prev_m) / m
                v = [
                    f"{prev.x + (cv.x * n)} {prev.y + (cv.y * n)} {prev.w + (cv.w * n)} {prev.h + (cv.h * n)}" for n in range(1, future_length+1)]

                label_str = f"{int(tid)} {' '.join(v)}\n"

                label_fpath = osp.join(cv_label_root, '{:06d}.txt'.format(fid))
                with open(label_fpath, 'a+') as f:
                    f.write(label_str)


if __name__ == "__main__":
    datasets = ["MOT15", "MOT16", "MOT17", "MOT20"]
    for d in datasets:
        print("\n", d)
        seq_root = f'data/MOT/{d}/images/train'
        label_root = f'data/MOT/{d}/labels_with_ids/train'
        if not osp.exists(seq_root):
            print(f"{seq_root} not found!")
            continue
        mkdirs(label_root)
        main(seq_root, label_root)
