import _init_paths
import os.path as osp
from forecast_utils.evaluation import *


def mkdirs(d):
    if not osp.exists(d):
        os.makedirs(d)

if __name__ == "__main__":
    datasets = ["MOT15", "MOT16", "MOT17", "MOT20"]
    cv_label = 'cv_10'
    pred_length = 60
    gt_folder = 'future'

    for d in datasets:
        filename = f'data/{d}/results/{cv_label}_{pred_length}.csv'
        mkdirs(f'data/{d}/results')
        try:
            print("\n", d)
            seq_label = ''
            aious = []
            fious = []
            ades = []
            fdes = []
            seqs = []

            if 'MOT' in d:
                seq_label = 'img1'
                label_root = f'data/{d}/future/train'

                if not osp.exists(label_root) | osp.exists(filename):
                    continue

                seqs = [s for s in os.listdir(label_root)]
                
                i = 0
                for seq in seqs:
                    print(seq)
                    
                    seq_label_root = osp.join(label_root, seq, seq_label)

                    aiou, fiou, ade, fde = eval_seq(
                                seq_label_root, pred_length, gt_folder, cv_label)
                    aious.append(aiou)
                    fious.append(fiou)
                    ades.append(ade)
                    fdes.append(fde)

                    print()
                    print(seq)
                    print('AIOU: ', round(aiou, 1))
                    print('FIOU: ', round(fiou, 1))
                    print('ADE:  ', round(ade, 1))
                    print('FDE:  ', round(fde, 1))


                    if filename:
                        save_result(filename, [aious, fious, ades, fdes],
                        seqs[:i+1], ["aiou", "fiou", "ade", "fde"])
                    i += 1
                               
            else:
                if 'Caltech' in d:
                    label_root = f'data/{d}/data/future'
                    filename = f'data/{d}/data/images/results/forecasts_cv_10.csv'
                else:
                    label_root = f'data/{d}/future'
                    filename = f'data/{d}/images/results/forecasts_cv_10.csv'
                
                if not osp.exists(label_root) | osp.exists(filename):
                    continue

                aiou, fiou, ade, fde = eval_seq(
                            label_root, pred_length, gt_folder, cv_label)
                aious.append(aiou)
                fious.append(fiou)
                ades.append(ade)
                fdes.append(fde)

                print()
                print('AIOU: ', round(aiou, 1))
                print('FIOU: ', round(fiou, 1))
                print('ADE:  ', round(ade, 1))
                print('FDE:  ', round(fde, 1))
                seqs = ['caltech']

            

                if filename:
                    save_result(filename, [aious, fious, ades, fdes],
                    seqs, ["aiou", "fiou", "ade", "fde"])

        except Exception as ex:
            print(d, ' failed due to ', ex)