python reweight.py \
    --dataroot_pt imagemol/ \
    --dataset_pt data \
    --log_dir logs \
    --dataroot imagemol/benchmarks/MPP/classification \
    --dataset bace \
    --task_type classification \
    --image_aug \
    --pretrain_lr 0.05\
    --finetune_lr 0.01 \
    --reweight_lr 200 \
    --batch_pt 1536 \
    --lam 1e-3 \
    --split scaffold \
    --iters 10000 \
    --resume ImageMol.pth.tar \
    --wandb \
    #--same_ft_dataset
    
    