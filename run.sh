#!/bin/bash

export CUDA_VISIBLE_DEVICES="3"
echo $CUDA_VISIBLE_DEVICES

python Dietnet/train.py --exp-path /home/rochefortc/shared_disk_wd4tb/rochefortc/Dietnetwork/EXP_DIETNET2/DEBUG/TEMP --exp-name matt_exp --epochs 2 --save_attributions
