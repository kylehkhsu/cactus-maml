#!/usr/bin/env bash
# This run reproduces the second-to-last line of Table 4 in the paper.
num_clusters=500
num_partitions=1    # more (50) partitions takes a while
num_encoding_dims=256
encoder=deepcluster

# training
python3 main.py \
    --dataset=miniimagenet --partition_algorithm=kmeans \
    --save_checkpoints=False \
    --num_classes_train=5 --num_classes_val=5 \
    --inner_update_batch_size_train=1 --inner_update_batch_size_val=5 --outer_update_batch_size=5 \
    --num_filters=32 --norm=batch_norm --max_pool=True --update_lr=0.05 --num_updates=5 \
    --metatrain_iterations=60000 --meta_batch_size=8 --mt_mode=encenc --mv_mode=gtgt \
    --num_encoding_dims=${num_encoding_dims} --encoder=${encoder} \
    --num_clusters=${num_clusters} --num_partitions=${num_partitions} \
    --logdir=./log/miniimagenet/kmeans-deepcluster \
    --test_set=False --train=True \

# testing
for inner_update_batch_size_val in 1 5 20 50
do
    python3 main.py \
        --dataset=miniimagenet --partition_algorithm=kmeans \
        --save_checkpoints=False \
        --num_classes_train=5 --num_classes_val=5 \
        --inner_update_batch_size_train=1 --inner_update_batch_size_val=${inner_update_batch_size_val} --outer_update_batch_size=5 \
        --num_filters=32 --norm=batch_norm --max_pool=True --update_lr=0.05 --num_updates=5 \
        --metatrain_iterations=60000 --meta_batch_size=8 --mt_mode=encenc --mv_mode=gtgt \
        --num_encoding_dims=${num_encoding_dims} --encoder=${encoder} \
        --num_clusters=${num_clusters} --num_partitions=${num_partitions} \
        --logdir=./log/miniimagenet/kmeans-deepcluster \
        --test_set=True --train=False --log_inner_update_batch_size_val=5
done

