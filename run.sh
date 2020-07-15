#!/bin/bash

export CUDA_VISIBLE_DEVICES="3"
echo $CUDA_VISIBLE_DEVICES

python SCRIPTS/train_merged_model.py --exp-path EXPERIMENT_01 --exp-name exp3 --epochs 2