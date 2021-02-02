import os.path as osp
import os
import shutil


def mkdirs(d, del_existing=False):
    if del_existing and osp.exists(d):
        shutil.rmtree(d)
    if not osp.exists(d):
        os.makedirs(d)


def chunker1(seq, size, start_index):
    return ((pos, seq.iloc[pos+start_index:pos + size + start_index]) for pos in range(0, len(seq)))
