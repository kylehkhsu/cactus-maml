#!/usr/bin/env bash
python3 main.py \
    --dataset=imagenet --input_type=images_224x224 --encoder=deepcluster --partition_algorithm=kmeans \
    --save_checkpoints=True \
    --num_classes_train=5 --num_classes_val=5 --num_clusters=10000 --num_partitions=100 \
    --inner_update_batch_size_train=5 --inner_update_batch_size_val=1 --outer_update_batch_size=5 \
    --num_filters=64 --norm=batch_norm --max_pool=True --update_lr=0.001 --num_updates=5 --meta_lr=0.0001 \
    --metatrain_iterations=240000 --meta_batch_size=2 --mt_mode=encenc --mv_mode=gtgt \
    --logdir=./log/imagenet/20181128/resnet224 \
    --test_set=False --train=True \
    --resnet=True --num_res_blocks=4 --num_parts_per_res_block=2

for inner_update_batch_size_val in 1 5 20 50
do
    python3 main.py \
        --dataset=imagenet --input_type=images_224x224 --encoder=deepcluster --partition_algorithm=kmeans \
        --num_classes_train=5 --num_classes_val=5 --num_clusters=10000 --num_partitions=100 \
        --inner_update_batch_size_train=5 --inner_update_batch_size_val=${inner_update_batch_size_val} --outer_update_batch_size=5 \
        --num_filters=64 --norm=batch_norm --max_pool=True --update_lr=0.001 --num_updates=5 --meta_lr=0.0001 \
        --metatrain_iterations=240000 --meta_batch_size=2 --mt_mode=encenc --mv_mode=gtgt \
        --logdir=./log/imagenet/20181128/resnet224 \
        --test_set=True --train=False --log_inner_update_batch_size_val=1 \
        --resnet=True --num_res_blocks=4 --num_parts_per_res_block=2
done