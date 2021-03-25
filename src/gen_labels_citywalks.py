import os.path as osp
import os
import numpy as np
import pandas as pd
from data_utils import *

train = ['BARCELONA', 'BRNO', 'ERFURT', 'KAUNAS', 'LEIPZIG', 'NUREMBERG', 'PALMA', 'PRAGUE', 'TALLINN', 'TARTU', 'VILNIUS', 'WEIMAR']
val = ['DRESDEN', 'HELSINKI', 'PADUA', 'POZNAN', 'VERONA', 'WARSAW']
test = ['KRAKOW', 'RIGA', 'WROCLAW']

is_gt = False
mask_rcnn = '../data/CityWalks/tracks/mask-rcnn_tracks.csv'
df = pd.read_csv(mask_rcnn)
fmt = fmt = '%d %d '+'%f ' * 4
width = 1280
height = 720
if is_gt:
    label_root = '../data/CityWalks/clips'
    mkdirs(label_root)
    cols =['frame_num','track','x','y','w','h','class']
    groups = df.groupby(['vid', 'filename'])
    fmt = fmt = '%d, %d, '+'%f, ' * 4 + ' %d'
    for (vid, filename), group in groups:
        print(filename, vid)
        folder = osp.join(label_root, filename, vid.replace('.mp4', ''), 'gt')
        mkdirs(folder)
        group['class'] =  1
        group['frame_num'] += 1
        # group[['cx', 'w']] /= 1280
        # group[['cy', 'h']] /= 720
        group['x'] = group['cx']  - group['w'] / 2
        group['y'] = group['cy']  - group['h'] / 2
        
        values = group[cols].values

        label_fpath = osp.join(folder, 'gt.txt')
        np.savetxt(label_fpath, values, delimiter=',', fmt=fmt)
else:
    label_root = '../data/CityWalks/labels_with_ids'
    mkdirs(label_root)
    groups = df.groupby(['vid', 'filename', 'frame_num'])
    cols =['class','track','cx','cy','w','h',]

    for (vid, filename, fid), group in groups:
        print(filename, vid, label_root)
        folder = osp.join(label_root, filename, vid.replace('.mp4', ''))
        mkdirs(folder)
        group['class'] = 0
        group[['cx', 'w']] /= 1280
        group[['cy', 'h']] /= 720
        values = group[cols].values

        label_fpath = osp.join(folder, '{:06d}.txt'.format(fid)) 
        print(label_fpath)
        np.savetxt(label_fpath, values, delimiter=' ', fmt=fmt)
        img_path = label_fpath.replace('labels_with_ids', 'images').replace('.txt', '.jpg').replace('../data/', '')
        if filename in train:
            with open('../data/CityWalks/citywalks.train', 'a+') as f:
                f.write(img_path +"\n")
        elif filename in val:
            with open('../data/CityWalks/citywalks.val', 'a+') as f:
                f.write(img_path +"\n")
        elif filename in val:
            with open('../data/CityWalks/citywalks.val', 'a+') as f:
                f.write(img_path +"\n")


