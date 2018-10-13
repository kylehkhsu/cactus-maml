#!/usr/bin/env bash
algorithm=embedding_cluster_matching
dataset=omniglot
for way in 5 20
do
    for shot in 1 5
    do
        python3 baselines.py \
            --encoder acai --num_encoding_dims 256 \
            --dataset ${dataset} --way ${way} --shot ${shot} --num_clusters 500 \
            --algorithm ${algorithm} --num_tasks 1000 --test_set True
        python3 baselines.py \
            --encoder bigan --num_encoding_dims 200 \
            --dataset ${dataset} --way ${way} --shot ${shot} --num_clusters 500 \
            --algorithm ${algorithm} --num_tasks 1000 --test_set True
    done
done

algorithm=embedding_nearest_neighbour
dataset=omniglot
for way in 5 20
do
    for shot in 1 5
    do
        python3 baselines.py \
            --encoder acai --num_encoding_dims 256 \
            --dataset ${dataset} --way ${way} --shot ${shot} --n_neighbours 1 \
            --algorithm ${algorithm} --num_tasks 1000 --test_set True
        python3 baselines.py \
            --encoder bigan --num_encoding_dims 200 \
            --dataset ${dataset} --way ${way} --shot ${shot} --n_neighbours 1 \
            --algorithm ${algorithm} --num_tasks 1000 --test_set True
    done
done

algorithm=embedding_logistic_regression
dataset=omniglot
for way in 5 20
do
    for shot in 1 5
    do
        python3 baselines.py \
            --encoder acai --num_encoding_dims 256 \
            --dataset ${dataset} --way ${way} --shot ${shot} --inverse_reg 1 \
            --algorithm ${algorithm} --num_tasks 1000 --test_set True
        python3 baselines.py \
            --encoder bigan --num_encoding_dims 200 \
            --dataset ${dataset} --way ${way} --shot ${shot} --inverse_reg 1 \
            --algorithm ${algorithm} --num_tasks 1000 --test_set True
    done
done

#dataset=miniimagenet
#for way in 5
#do
#    for shot in 1 5 20 50
#    do
#        python3 baselines.py \
#            --encoder bigan --num_encoding_dims 200 \
#            --dataset ${dataset} --way ${way} --shot ${shot} --num_clusters 500 \
#            --algorithm ${algorithm} --num_tasks 1000 --test_set True
#        python3 baselines.py \
#            --encoder deepcluster --num_encoding_dims 256 \
#            --dataset ${dataset} --way ${way} --shot ${shot} --num_clusters 500 \
#            --algorithm ${algorithm} --num_tasks 1000 --test_set True
#    done
#done
#
#dataset=mnist
#way=10
#for shot in 1 10 50
#do
#    python3 baselines.py \
#            --encoder bigan --num_encoding_dims 50 \
#            --dataset ${dataset} --way ${way} --shot ${shot} --num_clusters 10 \
#            --algorithm ${algorithm} --num_tasks 1000 --test_set True
#done
#
#dataset=celeba
#way=2
#shot=5
#python3 baselines.py \
#        --encoder deepcluster --num_encoding_dims 256 \
#        --dataset ${dataset} --way ${way} --shot ${shot} --num_clusters 500 \
#        --algorithm ${algorithm} --num_tasks 1000 --test_set True
#python3 baselines.py \
#        --encoder bigan --num_encoding_dims 200 \
#        --dataset ${dataset} --way ${way} --shot ${shot} --num_clusters 500 \
#        --algorithm ${algorithm} --num_tasks 1000 --test_set True