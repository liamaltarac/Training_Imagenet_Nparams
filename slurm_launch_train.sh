#!/bin/bash

source $SLURM_TMPDIR/env/bin/activate
python ./train_hvd_data_pipeline.py --model resnet18 --init he_normal --epochs 90  --ckpt_dir ckpts  --batch 32