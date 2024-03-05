import sys
sys.path.append("/home/scai/phd/aiz218323/Projects/XC_NLG/code")

import argparse

import pandas as pd
from utils.data_utils import *
from utils.eval_utils import _compute_precision_and_recall

from IPython.display import display

def load_predictions(pred_file):
    output = np.load(pred_file)
    return csr_matrix((output['data'], output['indices'], output['indptr']),
                      dtype=float, shape=output['shape'])

def _read_filter(filename, shape=None):
    return read_filter(filename, shape=shape) if os.path.exists(filename) else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='directory containing the data.')
    parser.add_argument('--pred_file', type=str, default=None,
                        help='file containing XC predictions.')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples.')
    parser.add_argument('--top_k', type=int, default=5, help="metrics k")
    args = parser.parse_args()


    mats, maps = read_data(args.data_dir)

    filters = []
    filters.append(_read_filter(f"{args.data_dir}/filter_labels_train.txt",
                                shape=mats[0].shape))
    filters.append(_read_filter(f"{args.data_dir}/filter_labels_test.txt",
                                shape=mats[1].shape))

    preds = load_predictions(args.pred_file)


    if args.num_samples is not None:
        tst_mat, tst_map, tst_filter = get_sample(mats[1], maps[1], filters[1],
                                                  num=args.num_samples, seed=50)
        tst_preds, _, _ = get_sample(preds, num=args.num_samples, seed=50)


        metrics = _compute_precision_and_recall(tst_preds, tst_mat, mats[0],
                                                k=args.top_k, filter_idx=tst_filter)
    else:
        metrics = _compute_precision_and_recall(preds, mats[1], mats[0], k=args.top_k,
                                                filter_idx=filters[1])

    print(metrics)

    with pd.option_context("display.precision", 2):
        display(pd.DataFrame([metrics])*100)

