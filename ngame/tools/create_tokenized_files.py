"""
For a dataset, create tokenized files in the folder {tokenizer-type}-{maxlen} folder inside the database folder
Sample usage: python -W ignore -u create_tokenized_files.py --data-dir /scratch/Workspace/data/LF-AmazonTitles-131K --tokenizer-type bert-base-uncased --max-length 32 --out_dir .
"""
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import os
import numpy as np
import time
import functools
import argparse
import pandas as pd


def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer


def _tokenize(batch_input):
    tokenizer, max_len, batch_corpus = batch_input[0], batch_input[1], batch_input[2]
    temp = tokenizer.batch_encode_plus(
                    batch_corpus,                           # Sentence to encode.
                    add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                    max_length = max_len,                   # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,           # Construct attn. masks.
                    return_tensors = 'np',                  # Return numpy tensors.
                    truncation=True
            )

    return (temp['input_ids'], temp['attention_mask'])


def convert(corpus, tokenizer, max_len, num_threads, bsz=100000):
    batches = [(tokenizer, max_len, corpus[batch_start: batch_start + bsz]) for batch_start in range(0, len(corpus), bsz)]

    pool = mp.Pool(num_threads)
    batch_tokenized = pool.map(_tokenize, batches)
    pool.close()

    input_ids = np.vstack([x[0] for x in batch_tokenized])
    attention_mask = np.vstack([x[1] for x in batch_tokenized])

    del batch_tokenized

    return input_ids, attention_mask

@timeit
def tokenize_dump(corpus, tokenization_dir,
                  tokenizer, max_len, prefix,
                  num_threads, batch_size=10000000):
    ind = np.zeros(shape=(len(corpus), max_len), dtype='int64')
    mask = np.zeros(shape=(len(corpus), max_len), dtype='int64')

    for i in range(0, len(corpus), batch_size):
        _ids, _mask = convert(
            corpus[i: i + batch_size], tokenizer, max_len, num_threads)
        ind[i: i + _ids.shape[0], :] = _ids
        mask[i: i + _ids.shape[0], :] = _mask

    input_ids_file = f"{tokenization_dir}/{prefix}_input_ids.npy"
    print(f"Saving -- {input_ids_file}")
    np.save(input_ids_file, ind)

    attn_mask_file = f"{tokenization_dir}/{prefix}_attention_mask.npy"
    print(f"Saving -- {attn_mask_file}")
    np.save(attn_mask_file, mask)

def read_map_file(map_file):
    obj_text = []
    with open(map_file, encoding='latin-1') as file:
        for line in file:
            text = line[:-1].split("->", maxsplit=1)[1]
            obj_text.append(text)
    return obj_text


def main(args):
    data_dir = args.data_dir
    max_len = args.max_length
    out_dir = args.out_dir
    meta_tag = args.meta_tag

    if args.is_csv:
        Y = pd.read_csv(f'{data_dir}/lbl.map.txt', header=None)[2].tolist()
        trnX = pd.read_csv(f'{data_dir}/trn.map.txt', header=None)[2].tolist()
        tstX = pd.read_csv(f'{data_dir}/tst.map.txt', header=None)[2].tolist()
    else:
        if args.legacy:
            Y = [x.strip() for x in open(
                f'{data_dir}/raw_data/label.raw.txt', "r", encoding="latin").readlines()]
            trnX = [x.strip() for x in open(
                f'{data_dir}/raw_data/train.raw.txt', "r", encoding="latin").readlines()]
            tstX = [x.strip() for x in open(
                f'{data_dir}/raw_data/test.raw.txt', "r", encoding="latin").readlines()]
        else:
            mm = "raw" if meta_tag is None else f"{meta_tag}"

            lbl_file = f'{data_dir}/raw_data/label.{mm}.txt'
            Y = read_map_file(f'{data_dir}/raw_data/label.{mm}.txt') if os.path.exists(lbl_file) else None

            trnX = read_map_file(f'{data_dir}/raw_data/train.{mm}.txt')
            tstX = read_map_file(f'{data_dir}/raw_data/test.{mm}.txt')

    if Y is None:
        print(f"#train: {len(trnX)}; #test: {len(tstX)}")
    else:
        print(f"#labels: {len(Y)}; #train: {len(trnX)}; #test: {len(tstX)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_type, do_lower_case=True)

    tokenization_dir = f"{data_dir}/{out_dir}"
    os.makedirs(tokenization_dir, exist_ok=True)

    print(f"Dumping files in {tokenization_dir}...")
    print("Dumping for trnX...")
    trn_tag = "trn_doc" if meta_tag is None else f"trn_{meta_tag}_doc"
    tokenize_dump(
        trnX, tokenization_dir, tokenizer, max_len, trn_tag, args.num_threads)

    print("Dumping for tstX...")
    tst_tag = "tst_doc" if meta_tag is None else f"tst_{meta_tag}_doc"
    tokenize_dump(
        tstX, tokenization_dir, tokenizer, max_len, tst_tag, args.num_threads)

    print("Dumping for Y...")
    if Y is not None:
        lbl_tag = "lbl" if meta_tag is None else f"lbl_{meta_tag}"
        tokenize_dump(Y, tokenization_dir, tokenizer, max_len, lbl_tag, args.num_threads)


def tokenize_test_file(args):
    data_dir = args.data_dir
    max_len = args.max_length
    out_dir = args.out_dir

    test_file = args.test_file

    if args.is_csv:
        tstX = pd.read_csv(test_file)
    else:
        tstX = read_map_file(test_file)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_type, do_lower_case=True)

    tokenization_dir = f"{data_dir}/{out_dir}"
    os.makedirs(tokenization_dir, exist_ok=True)

    print(f"Dumping files in {tokenization_dir}...")
    print("Dumping for tstX...")

    if args.is_csv:
        for i in range(10):
            tokenize_dump(tstX["input_text_{i:02d}"].tolist(), tokenization_dir, tokenizer,
                          max_len, f"sampled_tst_doc_{i:02d}", args.num_threads)
    else:
        tokenize_dump(tstX, tokenization_dir, tokenizer, max_len, args.prefix,
                      args.num_threads)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Data directory path - with {trn,tst,lbl}.raw.txt")
    parser.add_argument(
        "--max-length",
        type=int,
        help="Max length for tokenizer",
        default=32)
    parser.add_argument(
        "--tokenizer-type",
        type=str,
        help="Tokenizer to use",
        default="bert-base-uncased")
    parser.add_argument(
        "--num-threads",
        type=int,
        help="Number of threads to use",
        default=24)
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Dump folder inside dataset folder",
        default="")

    # to read from CSV file
    parser.add_argument(
        "--is_csv",
        action="store_true",
        help="the text files stored in csv format.")
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="map file without the identifier.")

    # tokenizes a single file
    parser.add_argument(
        "--test_file",
        type=str,
        help="Path to the test file.",
        default=None)
    parser.add_argument(
        "--prefix",
        type=str,
        help="prefix of the output tokenized file.",
        default="lbl")

    # metadata
    parser.add_argument(
        "--meta_tag",
        type=str,
        help="tag for metadata.",
        default=None)

    args = parser.parse_args()

    if args.test_file is None:
        main(args)
    else:
        tokenize_test_file(args)

