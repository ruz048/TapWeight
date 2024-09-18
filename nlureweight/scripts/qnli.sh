#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

python main_fullft.py \
    --model_name_or_path roberta-base \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-simcse-roberta-base-uncased-mlm-debug \
    --iters 50000 \
    --batch_size 8 \
    --weight_decay 0.001 \
    --learning_rate 3e-5 \
    --max_seq_length 512 \
    --evaluation_strategy steps \
    --metric_for_best_model accuracy \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_eval \
    --do_train \
    --do_mlm \
    --do_sop \
    --wandb \
    --task qnli \
    --same_dataset \
    "$@"
    #  --load_save \
