python reweight.py \
    --dataroot_pt imagemol/ \
    --dataset_pt data \
    --log_dir logs/AA1R \
    --dataroot imagemol/regression \
    --dataset AA1R \
    --task_type regression \
    --image_aug \
    --pretrain_lr 0.05\
    --finetune_lr 0.001 \
    --reweight_lr 10 \
    --batch_pt 1024  \
    --split scaffold \
    --iters 10000 \
    --resume imagemol/ImageMol.pth.tar \
    --val_freq 100 \
    --wandb \
    
    