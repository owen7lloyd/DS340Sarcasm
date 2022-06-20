#!/bin/sh

python extract_features.py \
  --input_file=../data/bert-input.txt \
  --output_file=../data/bert-output.jsonl \
  --vocab_file=../data/bert/vocab.txt \
  --bert_config_file=../data/bert/bert_config.json \
  --init_checkpoint=../data/bert/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8