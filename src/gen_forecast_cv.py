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
    return ((pos,seq.iloc[pos+start_index
    :pos + size + start_index]) for pos in range(0, len(seq)))

seq_root = 'data/MOT/MOT16/images/train'
label_root = 'data/MOT/MOT16/labels_with_ids/train'
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]
# seqs = ['ADL-Rundle-6', 'ETH-Bahnhof', 'KITTI-13', 'PETS09-S2L1', 'TUD-Stadtmitte', 'ADL-Rundle-8', 'KITTI-16', 'ETH-Pedcross2', 'ETH-Sunnyday', 'TUD-Campus', 'Venice-2']


tid_curr = 0
tid_last = -1
for seq in seqs:
    print(seq)
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    seq_label_root = osp.join(label_root, seq, 'img1')
    cv_label_root = osp.join(label_root, seq, 'cv_15')

    if os.path.exists(cv_label_root):
        shutil.rmtree(cv_label_root)

    mkdirs(cv_label_root)
    future_length = 60
    past_length = 10

    bboxes = []
    for filename in sorted(glob.glob(seq_label_root +"/*.txt")):
        fid = int(filename.split('/')[-1].replace('.txt', ''))
        bbox = np.loadtxt(filename, dtype=np.float64)
        if len(bbox.shape) == 1: 
            bbox = bbox.reshape(1, bbox.shape[0])
        # convert the original size
        bbox[:, [2,4]] *= seq_width
        bbox[:, [3,5]] *= seq_height
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
            v = [f"{prev.x + (cv.x * n)} {prev.y + (cv.y * n)} {prev.w + (cv.w * n)} {prev.h + (cv.h * n)}" for n in range(1, future_length+1)]

            label_str = f"{int(tid)} {' '.join(v)}\n"
            
            label_fpath = osp.join(cv_label_root, '{:06d}.txt'.format(fid))
            with open(label_fpath, 'a+') as f:
                f.write(label_str)
    
        