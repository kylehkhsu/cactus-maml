#!/usr/bin/env bash
# This run reproduces the first two results on the first line of Table 3 in the paper.
num_classes=5
num_clusters=500
num_partitions=2400
python3 main.py \
    --dataset=omniglot --partition_algorithm=kmeans --num_encoding_dims=256 --encoder=acai \
    --save_checkpoints=False \
    --num_classes_train=${num_classes} --num_classes_val=${num_classes} \
    --inner_update_batch_size_train=1 --inner_update_batch_size_val=5 --outer_update_batch_size=5 \
    --num_filters=32 --norm=batch_norm --max_pool=False --update_lr=0.05 --num_updates=5 \
    --metatrain_iterations=30000 --meta_batch_size=8 --mt_mode=randrand --mv_mode=gtgt \
    --num_encoding_dims=256 --encoder=acai \
    --num_clusters=${num_clusters} --num_partitions=${num_partitions} \
    --logdir=./log/omniglot/random \
    --test_set=False --train=True \
&& \
for inner_update_batch_size_val in 1 5
do
    python3 main.py \
        --dataset=omniglot --partition_algorithm=kmeans --num_encoding_dims=256 --encoder=acai \
        --save_checkpoints=False \
        --num_classes_train=${num_classes} --num_classes_val=${num_classes} \
        --inner_update_batch_size_train=1 --inner_update_batch_size_val=${inner_update_batch_size_val} --outer_update_batch_size=5  \
        --num_filters=32 --norm=batch_norm --max_pool=False --update_lr=0.05 --num_updates=5 \
        --metatrain_iterations=30000 --meta_batch_size=8 --mt_mode=randrand --mv_mode=gtgt \
        --num_clusters=${num_clusters} --num_partitions=${num_partitions} \
        --logdir=./log/omniglot/random \
        --test_set=True --train=False --log_inner_update_batch_size_val=5
