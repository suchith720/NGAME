from xclib.data.data_utils import read_corpus

import sys
import os
import pandas as pd
from tqdm import tqdm


def main(in_fname, op_fname, fields):
    with open(op_fname, 'w', encoding='latin') as fp:
        for line in read_corpus(in_fname):
            t = ""
            for f in fields:
                t += f"{line[f]} "
            fp.write(t.strip() + "\n")


def main_2(in_fname, op_fname):
    with open(op_fname, 'w', encoding='latin') as fp:
        with open(in_fname, 'r', encoding='latin') as fin:
            for line in tqdm(fin):
                title = line.strip().split('->')[1]
                fp.write(title + "\n")

def main_3(in_fname, op_fname):
    df = pd.read_csv(in_fname, header=None)
    df[1] = df[1].apply(lambda x: x.replace('\n', ''))
    with open(op_fname, 'w', encoding='utf-8') as fp:
        for i in tqdm(range(df.shape[0])):
            fp.write(df.iloc[i, 1] + "\n")


if __name__ == "__main__":
    in_fname = sys.argv[1]
    op_fname = sys.argv[2]

    dataset = os.path.split(
        os.path.dirname(in_fname))[1]

    if 'title' in dataset.lower():
        fields = ["title"]
    else:
        fields = ["title", "content"]

    #main(in_fname, op_fname, fields)
    main_3(in_fname, op_fname)
