#!/usr/bin/env bash
num_clusters=10
num_partitions=100
num_encoding_dims=256
encoder=acai

python3 main.py \
    --dataset=mnist --partition_algorithm=kmeans \
    --save_checkpoints=False \
    --num_classes_train=10 --num_classes_val=10 \
    --inner_update_batch_size_train=1 --inner_update_batch_size_val=5 --outer_update_batch_size=10 \
    --num_filters=32 --norm=batch_norm --max_pool=False --update_lr=0.05 --num_updates=5 \
    --metatrain_iterations=20000 --meta_batch_size=16 --mt_mode=encenc --mv_mode=gtgt \
    --num_encoding_dims=${num_encoding_dims} --encoder=${encoder} \
    --num_clusters=${num_clusters} --num_partitions=${num_partitions} \
    --logdir=./log/mnist/kmeans-acai \
    --test_set=False --train=True
for inner_update_batch_size_val in 1 5 10
do
    python3 main.py \
        --dataset=mnist --partition_algorithm=kmeans \
        --save_checkpoints=False \
        --num_classes_train=10 --num_classes_val=10 \
        --inner_update_batch_size_train=1 --inner_update_batch_size_val=${inner_update_batch_size_val} --outer_update_batch_size=10  \
        --num_filters=32 --norm=batch_norm --max_pool=False --update_lr=0.05 --num_updates=5 \
        --metatrain_iterations=20000 --meta_batch_size=16 --mt_mode=encenc --mv_mode=gtgt \
        --num_encoding_dims=${num_encoding_dims} --encoder=${encoder} \
        --num_clusters=${num_clusters} --num_partitions=${num_partitions} \
        --logdir=./log/mnist/kmeans-acai \
        --test_set=True --train=False --log_inner_update_batch_size_val=5
done
