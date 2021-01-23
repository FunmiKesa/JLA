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


def main(seq_root, label_root, future_length=60, seq_label="img1", future_label="past"):
    seqs = [s for s in os.listdir(seq_root)]
    for seq in seqs:
        print(seq)
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find(
            'imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find(
            'imHeight=') + 9:seq_info.find('\nimExt')])

        seq_label_root = osp.join(label_root, seq, seq_label)
        future_label_root = osp.join(label_root, seq, future_label)

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
    datasets = ["MOT15", "MOT16", "MOT20"]
    for d in datasets:
        print("\n", d)
        seq_root = f'data/MOT/{d}/images/train'
        label_root = f'data/MOT/{d}/labels_with_ids/train'
        if not osp.exists(seq_root):
            print(f"{seq_root} not found!")
            continue
        mkdirs(label_root)
        main(seq_root, label_root)
