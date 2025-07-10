#!/bin/bash

source /env/bin/activate
python ./train.py --model resnet18 --init he_normal --epochs 90  --ckpt_dir ckpts  --batch 32