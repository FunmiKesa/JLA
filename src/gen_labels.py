import os.path as osp
import os
import numpy as np
import shutil


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)


def gen_labels_15(seq_root, label_root, seq_label="img1", gt_label="gt"):
    seqs = [s for s in os.listdir(seq_root)]
    seqs.sort()

    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        print(seq)
        with open(osp.join(seq_root, seq, 'seqinfo.ini'), 'r') as file:
            seq_info = file.read()
            seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
            seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        gt_txt = osp.join(seq_root, seq, gt_label, f'{gt_label}.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
        idx = np.lexsort(gt.T[:2, :])
        gt = gt[idx, :]

        seq_label_root = osp.join(label_root, seq, seq_label)
        mkdirs(seq_label_root)

        for fid, tid, x, y, w, h, mark, _, _, _ in gt:
            if mark == 0:
                continue
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)


def gen_labels(seq_root, label_root, seq_label="img1", gt_label="gt"):
    seqs = [s for s in os.listdir(seq_root)]
    seqs.sort()

    tid_curr = 0
    tid_last = -1
    for seq in seqs:
        print(seq)
        seq_info = open(osp.join(seq_root, seq, 'seqinfo.ini')).read()
        seq_width = int(seq_info[seq_info.find(
            'imWidth=') + 8:seq_info.find('\nimHeight')])
        seq_height = int(seq_info[seq_info.find(
            'imHeight=') + 9:seq_info.find('\nimExt')])

        gt_txt = osp.join(seq_root, seq, gt_label, f'{gt_label}.txt')
        gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')

        seq_label_root = osp.join(label_root, seq, seq_label)
        mkdirs(seq_label_root)

        for fid, tid, x, y, w, h, mark, label, _ in gt:
            if mark == 0 or not label == 1:
                continue
            fid = int(fid)
            tid = int(tid)
            if not tid == tid_last:
                tid_curr += 1
                tid_last = tid
            x += w / 2
            y += h / 2
            label_fpath = osp.join(seq_label_root, '{:06d}.txt'.format(fid))
            label_str = '0 {:d} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                tid_curr, x / seq_width, y / seq_height, w / seq_width, h / seq_height)
            with open(label_fpath, 'a') as f:
                f.write(label_str)


if __name__ == "__main__":
    datasets = ["MOT15", "MOT16", "MOT17", "MOT20"]
    for d in datasets:
        print("\n", d)
        seq_root = f'data/{d}/images/train'
        label_root = f'data/{d}/labels_with_ids/train'
        if not osp.exists(seq_root) | osp.exists(label_root):
            print(f"{seq_root} not found or {label_root} exists!")
            continue
        if os.path.exists(label_root):
            shutil.rmtree(label_root)
        mkdirs(label_root)
        
        if d == "MOT15":
            gen_labels_15(seq_root, label_root)
        else:
            gen_labels(seq_root, label_root)
