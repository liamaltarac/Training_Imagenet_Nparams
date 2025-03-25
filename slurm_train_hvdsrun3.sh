#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --gpus-per-node=v100l:1

#SBATCH --cpus-per-gpu=8  # 8*4 :There are 32 CPU cores on V100 Cedar GPU nodes
#SBATCH --mem=48G   # Request the full memory of the node use 0 for full
#SBATCH --time=20:00:00
#SBATCH --wait-all-nodes=1

#SBATCH --output=%N-%j.out 
#SBATCH --mail-user=liam.frija-altarac.1@ens.etsmtl.ca
#SBATCH --mail-type=ALL

#echo $SLURM_NNODES
slurm_prep_env.sh
#srun  -N $SLURM_NNODES  --ntasks-per-node=1 slurm_prep_env.sh

#module load gcc/9.3.0 cuda/11.8

#module load cuda cudnn 
export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

#srun slurm_launch_train.sh
source env/bin/activate

python3 ./train_no_hvd.py --model vgg16 --epochs 8  --batch 64 -n 1 --lr 0.0000025 --seed 99 --start_from 6
python3 ./train_no_hvd.py --model vgg16 --epochs 8  --batch 64 -n 3 --lr 0.0000025 --seed 99
python3 ./train_no_hvd.py --model vgg16 --epochs 8  --batch 64 -n 4 --lr 0.0000025 --seed 99
python3 ./train_no_hvd.py --model vgg16 --epochs 8  --batch 64 -n 5 --lr 0.0000025 --seed 99