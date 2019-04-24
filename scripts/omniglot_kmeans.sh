#!/usr/bin/env bash
# To run evaluation, use the '--train=False' flag. Use the '--test_set=True' flag to use the test split.
# Note that for Omniglot, the MAML model does not use max pooling.
# This run reproduces the first two results on the 6th line of Table 1 in the paper.
num_clusters=500
num_partitions=1  # 100 partitions takes a while
num_encoding_dims=256   # for bigan, change to num_encoding_dims=200; remember to change the logdir flag!
encoder=acai    # for bigan, change to encoder=bigan
python3 main.py \
    --dataset=omniglot --partition_algorithm=kmeans \
    --save_checkpoints=False \
    --num_classes_train=20 --num_classes_val=5 \
    --inner_update_batch_size_train=1 --inner_update_batch_size_val=5 --outer_update_batch_size=5 \
    --num_filters=32 --norm=batch_norm --max_pool=False --update_lr=0.05 --num_updates=5 \
    --metatrain_iterations=30000 --meta_batch_size=8 --mt_mode=encenc --mv_mode=gtgt \
    --num_encoding_dims=${num_encoding_dims} --encoder=${encoder} \
    --num_clusters=${num_clusters} --num_partitions=${num_partitions} \
    --logdir=./log/omniglot/kmeans-acai \
    --test_set=False --train=True
for inner_update_batch_size_val in 1 5
do
    python3 main.py \
        --dataset=omniglot --partition_algorithm=kmeans \
        --save_checkpoints=False \
        --num_classes_train=20 --num_classes_val=5 \
        --inner_update_batch_size_train=1 --inner_update_batch_size_val=${inner_update_batch_size_val} --outer_update_batch_size=5  \
        --num_filters=32 --norm=batch_norm --max_pool=False --update_lr=0.05 --num_updates=5 \
        --metatrain_iterations=30000 --meta_batch_size=8 --mt_mode=encenc --mv_mode=gtgt \
        --num_encoding_dims=${num_encoding_dims} --encoder=${encoder} \
        --num_clusters=${num_clusters} --num_partitions=${num_partitions} \
        --logdir=./log/omniglot/kmeans-acai \
        --test_set=True --train=False --log_inner_update_batch_size_val=5
done
