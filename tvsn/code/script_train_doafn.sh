#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 th train_doafn.lua --category chair --modelString DOAFN_SYM --imgscale 256 --batchSize 25 --lr 0.00005 --saveFreq 20 --maxEpoch 100 --gpu 0 --nThreads 8 --resume 0
CUDA_VISIBLE_DEVICES=0 th train_doafn.lua --category chair --modelString DOAFN_SYM --imgscale 256 --batchSize 25 --lr 0.00001 --saveFreq 20 --maxEpoch 300 --gpu 0 --nThreads 8 --resume 1
