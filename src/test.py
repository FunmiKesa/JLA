# import pstats
# p = pstats.Stats('src/output.txt')
# p.strip_dirs().sort_stats('time').print_stats(100)
import _init_paths

from forecast_utils.evaluation import *
if __name__ == "__main__":
    label_root = '/media2/funmi/MOT/MOT16/labels_with_ids/train'

    filename = '/media2/funmi/MOT/MOT16/images/results/forecasts_cv_15.csv'
    eval(label_root, pred_folder='cv_15', pred_length=30, filename=filename)
