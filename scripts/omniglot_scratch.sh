#!/usr/bin/env bash
# This run reproduces the second result on the first line of Table 1 in the paper.
# Leverages MAML meta-test code to train from scratch.
num_classes=5
inner_update_batch_size_val=5
update_lr=0.1
python3 main.py \
    --dataset=omniglot --encoder=acai --num_encoding_dims=256 \
    --save_checkpoints=False \
    --num_classes_train=${num_classes} --num_classes_val=${num_classes} --mv_mode=gtgt --meta_batch_size=1 --metatrain_iterations=0 \
    --inner_update_batch_size_train=1 --inner_update_batch_size_val=${inner_update_batch_size_val} --outer_update_batch_size=5 \
    --num_filters=32 --norm=batch_norm --max_pool=False --update_lr=${update_lr} \
    --logdir=./log/omniglot/from-scratch \
    --test_set=True --train=False --from_scratch=True --num_eval_tasks=1000
