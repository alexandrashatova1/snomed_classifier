#!/bin/bash

#output directory
#mkdir ./output_dir
cd ./output_dir

# load biobert weights
curl https://github.com/naver/biobert-pretrained/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.tar.gz

# unzip
tar -xzf biobert_v1.1_pubmed.tar.gz

# convert biobert weigth from tensorflow to transformers and save to 'biobert_v1.1_pubmed' file
transformers-cli convert --model_type bert \
--tf_checkpoint biobert_v1.1_pubmed/model.ckpt-1000000 \
--config biobert_v1.1_pubmed/bert_config.json \
--pytorch_dump_output biobert_v1.1_pubmed/pytorch_model.bin

