python3 tools/create_tokenized_files.py --data-dir ../../../data/\(mapped\)LF-WikiTitles-500K/ --max-length 32 --tokenizer-type sentence-transformers/msmarco-bert-base-dot-v5
python3 tools/create_tokenized_files.py --data-dir ../../../data/\(mapped\)LF-AmazonTitles-1.3M/ --max-length 256 --tokenizer-type sentence-transformers/msmarco-bert-base-dot-v5 --meta_tag meta-category
python3 tools/create_tokenized_files.py --data-dir ../../../data/\(mapped\)LF-AmazonTitles-1.3M/ --max-length 32 --tokenizer-type sentence-transformers/msmarco-bert-base-dot-v5

