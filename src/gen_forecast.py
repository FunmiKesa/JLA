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

seq_root = 'data/MOT/MOT20/images/train'
label_root = 'data/MOT/MOT20/labels_with_ids/train'
mkdirs(label_root)
seqs = [s for s in os.listdir(seq_root)]
# seqs = ['ADL-Rundle-6', 'ETH-Bahnhof', 'KITTI-13', 'PETS09-S2L1', 'TUD-Stadtmitte', 'ADL-Rundle-8', 'KITTI-17', 'ETH-Pedcross2', 'ETH-Sunnyday', 'TUD-Campus', 'Venice-2']


tid_curr = 0
tid_last = -1
for seq in seqs:
    print(seq)
    seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
    seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
    seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    seq_label_root = osp.join(label_root, seq, 'img1')
    future_label_root = osp.join(label_root, seq, 'future')
    past_label_root = osp.join(label_root, seq, 'past')

    # mkdirs(seq_label_root)

    if os.path.exists(future_label_root):
        shutil.rmtree(future_label_root)

    mkdirs(future_label_root)

    if os.path.exists(past_label_root):
        shutil.rmtree(past_label_root)

    mkdirs(past_label_root)
    future_length = 90
    past_length = 30

    bboxes = []
    for filename in sorted(glob.glob(seq_label_root +"/*.txt")):
        fid = int(filename.split('/')[-1].replace('.txt', ''))
        bbox = np.loadtxt(filename)
        if len(bbox.shape) == 1: 
            bbox = bbox.reshape(1, bbox.shape[0])
        bbox[:, 0] = fid
        bboxes += [bbox]

    bboxes = np.concatenate(bboxes)
    

    # data = np.zeros((bboxes.shape[0], bboxes.shape[1])).astype(np.float32)
    # data[:, :bboxes.shape[0]] = bboxes

    # prediction labels
    # [[f'x_{i}',f'x_{i}',f'x_{i}',f'x_{i}',f'x_{i}',  for i in range(future_length)]
    
    df = pd.DataFrame(bboxes, columns=['fid', 'tid', 'x', 'y', 'w', 'h'])

    # group by frame
    groups = df.groupby(['tid'])
    
    for tid, group in groups:
        fids = group.fid.unique()
        group['cord'] = group.apply(lambda  row: f"{row.x} {row.y} {row.w} {row.h} ", axis=1)

        for i, c in chunker1(group, future_length, 0):
            
            v = c.reset_index().pivot(index='tid', columns=['index'], values='cord')
            if v.empty:
                continue
            label_str = f"{int(tid)} {' '.join(v.iloc[0])}\n"
            
            fid = int(fids[i])
            label_fpath = osp.join(future_label_root, '{:06d}.txt'.format(fid))
            with open(label_fpath, 'a+') as f:
                f.write(label_str)
    
        # compute past
        group_reverse = group.iloc[::-1]
        fids_reverse = fids[::-1]
        for i, c in chunker1(group_reverse, past_length, 1):
            v = c.reset_index().pivot(index='tid', columns=['index'], values='cord')
            if v.empty:
                continue
            label_str = f"{int(tid)} {' '.join(v.iloc[0][::-1])}\n"
            fid = int(fids_reverse[i])
            label_fpath = osp.join(past_label_root, '{:06d}.txt'.format(fid))
            with open(label_fpath, 'a+') as f:
                f.write(label_str)