#!/usr/bin/env bash
margin=1.2  # you can try 0
num_partitions=2400 # calculated for 30000 iterations * 8 tasks per iteration / 100 tasks per partition
python3 main.py \
    --dataset=omniglot --partition_algorithm=hyperplanes \
    --save_checkpoints=False \
    --num_classes_train=20 --num_classes_val=5 \
    --inner_update_batch_size_train=1 --inner_update_batch_size_val=5 --outer_update_batch_size=5 \
    --num_filters=32 --norm=batch_norm --max_pool=False --update_lr=0.05 --num_updates=5 \
    --metatrain_iterations=30000 --meta_batch_size=8 --mt_mode=encenc --mv_mode=gtgt \
    --num_encoding_dims=256 --encoder=acai \
    --margin=${margin} --num_partitions=${num_partitions} \
    --logdir=./log/omniglot/20180917/acai-hyperplanes \
    --test_set=False --train=True
for inner_update_batch_size_val in 1 5
do
    python3 main.py \
        --dataset=omniglot --partition_algorithm=hyperplanes \
        --save_checkpoints=False \
        --num_classes_train=20 --num_classes_val=5 \
        --inner_update_batch_size_train=1 --inner_update_batch_size_val=${inner_update_batch_size_val} --outer_update_batch_size=5  \
        --num_filters=32 --norm=batch_norm --max_pool=False --update_lr=0.05 --num_updates=5 \
        --metatrain_iterations=30000 --meta_batch_size=8 --mt_mode=encenc --mv_mode=gtgt \
        --num_encoding_dims=256 --encoder=acai \
        --margin=${margin} --num_partitions=${num_partitions} \
        --logdir=./log/omniglot/20180917/acai-hyperplanes \
        --test_set=True --train=False --log_inner_update_batch_size_val=5
done