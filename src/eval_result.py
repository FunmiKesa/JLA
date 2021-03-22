import _init_paths
import os
from tracking_utils.evaluation import Evaluator
import motmetrics as mm

data_root = "../data/MOT17/images/train"
seqs = os.listdir(data_root)
data_type = "mot"
# seqs_str = '''Venice-2
#                       KITTI-13
#                       KITTI-17
#                       ETH-Bahnhof
#                       ETH-Sunnyday
#                       PETS09-S2L1
#                       TUD-Campus
#                       TUD-Stadtmitte
#                       ADL-Rundle-6
#                       ADL-Rundle-8
#                       ETH-Pedcross2
#                       TUD-Stadtmitte'''
# seqs = [seq.strip() for seq in seqs_str.split()]

accs = []

for seq in seqs:
    result_filename = f"../Archive/{seq}.txt"

    evaluator = Evaluator(data_root, seq, data_type)
    accs.append(evaluator.eval_file(result_filename))

metrics = mm.metrics.motchallenge_metrics
mh = mm.metrics.create()
summary = Evaluator.get_summary(accs, seqs, metrics)
strsummary = mm.io.render_summary(
    summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names
)
print(strsummary)