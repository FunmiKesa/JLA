# import pstats
# p = pstats.Stats('src/output.txt')
# p.strip_dirs().sort_stats('time').print_stats(100)
import _init_paths
import os.path as osp
from forecast_utils.evaluation import *
if __name__ == "__main__":
    datasets = ["MOT15", "MOT16", "MOT20"]
    for d in datasets:
        label_root = f'data/MOT/{d}/labels_with_ids/train'
        filename = f'data/MOT/{d}/images/results/forecasts_cv_10.csv'
        if not osp.exists(label_root) | osp.exists(filename):
            continue
        eval(label_root, pred_folder='cv_10', pred_length=30, filename=filename)
