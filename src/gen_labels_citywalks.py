import os.path as osp
import os
import numpy as np
import pandas as pd

def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

label_root = 'data/CityWalks/labels_with_ids'
mkdirs(label_root)

mask_rcnn = 'data/CityWalks/tracks/mask-rcnn_tracks.csv'
df = pd.read_csv(mask_rcnn)
cols =['class','track','cx','cy','w','h']
groups = df.groupby(['vid', 'filename', 'frame_num'])
fmt = fmt = '%d %d '+'%f ' * 4
width = 1280
height = 720
for (vid, filename, fid), group in groups:
    folder = osp.join(label_root, filename, vid.replace('.mp4', ''))
    mkdirs(folder)
    group['class'] = 0
    group[['cx', 'w']] /= 1280
    group[['cy', 'h']] /= 720
    values = group[cols].values

    label_fpath = osp.join(folder, '{:06d}.txt'.format(fid))
    np.savetxt(label_fpath, values, delimiter=' ', fmt=fmt)