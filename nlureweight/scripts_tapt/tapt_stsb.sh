#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python train.py \
    --model_name_or_path roberta-base \
    --train_file /data1/ruiyi/taskweight/tapt/stsb_train_sentences.txt \
    --output_dir /data1/ruiyi/taskweight/tapt/result/roberta-base-tapt-stsb \
    --num_train_epochs 10 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 64 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
    "$@"
