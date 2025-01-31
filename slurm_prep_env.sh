#!/bin/bash

module load StdEnv/2020
module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate



pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt --verbose

#pip install --no-index horovod==0.28.0  --verbose 
#pip install numpy==1.22.0


pip install --no-index horovod==0.28.1


#cp -R ~/scratch/$USERS/ILSVRC2012_img_train $SLURM_TMPDIR
#cp -R ~/scratch/$USERS/ILSVRC2012_img_val $SLURM_TMPDIR


# mkdir $SLURM_TMPDIR/ILSVRC2012_img_train
# mkdir $SLURM_TMPDIR/ILSVRC2012_img_val



# cp ~/scratch/$USERS/ILSVRC2012_img_train.tar $SLURM_TMPDIR
# cp ~/scratch/$USERS/ILSVRC2012_img_val.tar $SLURM_TMPDIR


# tar  -xf  $SLURM_TMPDIR/ILSVRC2012_img_train.tar -C $SLURM_TMPDIR/ILSVRC2012_img_train
# tar  -xf  $SLURM_TMPDIR/ILSVRC2012_img_val.tar -C $SLURM_TMPDIR/ILSVRC2012_img_val
# sleep 10s


# for f in $SLURM_TMPDIR/ILSVRC2012_img_train/*.tar; do
#     d=`basename $f .tar`
#     mkdir $SLURM_TMPDIR/ILSVRC2012_img_train/$d
#     tar xf $f -C $SLURM_TMPDIR/ILSVRC2012_img_train/$d
# done
# sleep 10s

# for file in $SLURM_TMPDIR/ILSVRC2012_img_val/*.JPEG; do 
#     read line;
#     d=`basename $file`
#     mkdir -p $SLURM_TMPDIR/ILSVRC2012_img_val/${line}
#     mv -v "${file}" "${SLURM_TMPDIR}/ILSVRC2012_img_val/${line}"; 
#  done < imagenet_utils/ILSVRC2012_validation_ground_truth_synset.txt


# rm -r $SLURM_TMPDIR/ILSVRC2012_img_train.tar
# rm -r $SLURM_TMPDIR/ILSVRC2012_img_val.tar

wandb online 
