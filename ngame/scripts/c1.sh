python3 tools/create_tokenized_files.py --data-dir ../../../data/\(mapped\)LF-WikiSeeAlsoTitles-320K/ --max-length 256 --tokenizer-type sentence-transformers/msmarco-bert-base-dot-v5 --meta_tag meta-hyper_link
python3 tools/create_tokenized_files.py --data-dir ../../../data/\(mapped\)LF-WikiSeeAlsoTitles-320K/ --max-length 32 --tokenizer-type sentence-transformers/msmarco-bert-base-dot-v5
python3 tools/create_tokenized_files.py --data-dir ../../../data/\(mapped\)LF-WikiTitles-500K/ --max-length 256 --tokenizer-type sentence-transformers/msmarco-bert-base-dot-v5 --meta_tag meta-hyper_link

