#http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-hvd-tf-multi-eng.html
#https://github.com/horovod/horovod/blob/master/examples/keras/keras_imagenet_resnet50.py


import os
import argparse
import keras
from keras import backend as K
from keras.preprocessing import image
import tensorflow as tf
import horovod.keras as hvd
from horovod.tensorflow import Average, Sum, Adasum

from glob import glob


import imagenet_utils.helper.wordnet_functions as wf


from tensorflow.keras.initializers import RandomNormal, Constant, HeNormal, GlorotNormal
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications import ResNet50, VGG16

from tensorflow.keras.preprocessing.image import ImageDataGenerator


import wandb
from wandb.integration.keras import WandbCallback

import sys
import numpy as np
import random

#from imagenet_utils.imagenet_clsloc import clsloc
from imagenet_utils.load_images import load_images
from imagenet_utils.preprocess import Preprocessing

from convert_to_n_params import convert_model

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['resnet18', 'resnet50', 'vgg16'], default='vgg16', type=str.lower)


parser.add_argument('-n', default=3, type=int)


parser.add_argument('--load', help='load model')
parser.add_argument('--batch', type=int, help='global batch size', default=32)
'''parser.add_argument('--data_format', help='specify NCHW or NHWC',
                    type=str, default='NCHW')'''

parser.add_argument('--lr', type=float, default=0.0125,  help='lr on one GPU' )

parser.add_argument('--decay', type=float, default=0.0001,  help='lr on one GPU' )

parser.add_argument('--drop_on', help='drop learning rate on which epochs', nargs='*', type=int)
parser.add_argument('--drop_by', help='drop learning rate by how much')

parser.add_argument('--seed', help='random seed', type=int, default=1)

parser.add_argument('--epochs', help='number of epochs', type=int)
parser.add_argument('--start_from', help='number of epochs')
parser.add_argument('--ckpt_dir', help='Checkpoint directory', default="checkpoints")

parser.add_argument('--train-dir')
parser.add_argument('--val-dir')

args = parser.parse_args()

hvd.init()

checkpoint_dir = args.ckpt_dir

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


if hvd.rank() == 0:
    print("GPUS", len(gpus))
    print(gpus)

os.environ['PYTHONHASHSEED']=str(args.seed)
#os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


SEED = args.seed
GLOBAL_BATCH_SIZE  = args.batch *  hvd.size()
batch = args.batch
epochs = args.epochs

LR_INIT = args.lr * hvd.size() #  tf.cast( args.lr * hvd.size(), dtype=tf.float32)
#LR_DECREASE_FACTOR = args.drop_by #5
#LR_DECREASE_EPOCHS = args.drop_on #[10, 20, 30]  

#init = INIT = GeometricInit3x3(rho=0.9, beta=0.9)

models = {'resnet50': ResNet50,
          'vgg16':   VGG16 }
          
model = models[args.model](
    include_top=True,
    weights='imagenet',
    classes=1000,
    classifier_activation='softmax'
)


model = convert_model(model, n=args.n)


opt = keras.optimizers.SGD(learning_rate=LR_INIT, momentum=0.9)

# Horovod: add Horovod Distributed Optimizer.
d_opt = hvd.DistributedOptimizer(opt)#, op=Sum)

model.compile(
        optimizer=d_opt,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        experimental_run_tf_function=False,
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ])
verbose=1 if hvd.rank() == 0 else 0

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),

    # Horovod: average metrics among workers at the end of every epoch.
    #
    # Note: This callback must be in the list before the ReduceLROnPlateau,
    # TensorBoard, or other metrics-based callbacks.
    hvd.callbacks.MetricAverageCallback(),

    # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
    # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
    # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
    #hvd.callbacks.LearningRateWarmupCallback(initial_lr=LR_INIT,
    #                                            warmup_epochs=5,
    #                                            verbose=verbose),'''

    # Horovod: after the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
    #hvd.callbacks.LearningRateScheduleCallback(initial_lr=LR_INIT,
    #                                            multiplier=1.,
    #                                            start_epoch=0,
    #                                            end_epoch=epochs)

    #hvd.callbacks.LearningRateScheduleCallback(initial_lr=LR_INIT*1e-1, multiplier=1., start_epoch=30, end_epoch=60),
    #hvd.callbacks.LearningRateScheduleCallback(initial_lr=LR_INIT*1e-2, multiplier=1., start_epoch=60, end_epoch=80),
    #hvd.callbacks.LearningRateScheduleCallback(initial_lr=LR_INIT*1e-3, multiplier=1., start_epoch=80)

]


'''def lr_schedule(epoch):
    if epoch < 30:
        return 0.1
    if epoch < 60:
        return 0.01
    if epoch < 80:
        return 0.001
    return 0.0001'''

#callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0))




run_name = f"{args.model}_{args.n}_seed={SEED}"

if hvd.rank() == 0:
    checkpoint_dir = checkpoint_dir +  f"/{run_name}"
    callbacks.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"))


    run = wandb.init(project="IMAGENET_Nparams_Experiments", entity="geometric_init",
                            # track hyperparameters and run metadata
                        config={
                        "Batch_size": batch,
                        "Global_batch": GLOBAL_BATCH_SIZE,
                        "epochs": epochs,
                        "nparams" : str(args.n),
                        "optimizer": opt.__class__.__name__,
                        "LR": LR_INIT,
                        "model": args.model,
                        "Seed": SEED,
                        "SIZE": hvd.size()
                        })

    wandb.run.name = run_name 

    '''if args.model == "vgg16":
        layout_callback = FLL(wandb=wandb, 
                            model=model,
                            layer_filter_dict={"block1_conv2":  [0,  3,  7, 15,  31,   63],
                                                "block3_conv2": [0, 15, 31, 63, 127,  255],
                                                "block5_conv3": [0, 31, 63, 127, 255, 511]})
    else:
        layout_callback = FLL(wandb=wandb, 
                            model=model,
                            layer_filter_dict={"conv2_block1_2_conv3x3": [0, 31, 63],
                                                "conv3_block1_2_conv3x3": [0, 31, 63],
                                                "conv5_block1_2_conv3x3": [0, 31, 63]})'''
        
    #callbacks.append(layout_callback)
    callbacks.append(WandbCallback(save_model= False))

# Configuration for creating new images
train_list = glob(os.path.join(os.getenv('SCRATCH'), "ILSVRC2012_img_train")+'/*/*.JPEG')
val_list = glob(os.path.join(os.getenv('SCRATCH'), "ILSVRC2012_img_val") +'/*/*.JPEG')

ilsvrc2012_categories = wf.get_ilsvrc2012_categories()
#print(ilsvrc2012_categories)
train_labels = [ilsvrc2012_categories.index(os.path.normpath(str(path)).split(os.path.sep)[-2]) for path in train_list]
val_labels = [ilsvrc2012_categories.index(os.path.normpath(str(path)).split(os.path.sep)[-2]) for path in val_list]

if hvd.rank() == 0:

    print("TRAIN list : ", len(train_list))
    print("TRAIN LABELS : ", len(train_labels))
    #print("VAL LABELS : ", val_labels)

train_preprocess = Preprocessing(val=False).preprocess
trainDS = tf.data.Dataset.from_tensor_slices((train_list, train_labels))
trainDS = (trainDS
    .cache()
    .shuffle(trainDS.cardinality(), seed=SEED, reshuffle_each_iteration=True)
    .shard(num_shards=hvd.size(), index=hvd.rank())
	.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
	.map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .repeat() #
	.batch(args.batch, drop_remainder=True)
	.prefetch(1) #tf.data.AUTOTUNE)
)

val_preprocess = Preprocessing(val=True).preprocess
valDS =tf.data.Dataset.from_tensor_slices((val_list, val_labels))
valDS = (valDS
    .shard(num_shards=hvd.size(), index=hvd.rank())
    .cache()
	.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
	.map(val_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .repeat() #
	.batch(args.batch, drop_remainder=True)
	.prefetch(1) #tf.data.AUTOTUNE)
)

'''if hvd.rank() == 0:


    print("\n\n\n\n*********** STARTIN TRAINING ***********\n\n\n\n")
    print(run_name)
    #print(datagen.mean, datagen.std)
    print("LR init= ", args.lr * hvd.size())
    print("BATCH =", batch)
    print("GLOBAL BATCH =", GLOBAL_BATCH_SIZE)
    print("NUM GPUS = " ,hvd.size() )
    print("\n\n\n\n*****************************************\n\n\n\n")'''

initial_results = model.evaluate(valDS, verbose=verbose, steps= 50000 // GLOBAL_BATCH_SIZE)
wandb.log({"initial_val_loss": initial_results[0], "initial_val_accuracy": initial_results[1]})


history = model.fit(trainDS, 
                    steps_per_epoch=int(np.round(1281167 / GLOBAL_BATCH_SIZE)),
                    epochs=epochs, 
                    workers=4,
                    validation_data=valDS,
                    verbose=verbose,
                    validation_steps=2 * 50000 // GLOBAL_BATCH_SIZE,
                    callbacks=callbacks ) 

if hvd.rank() == 0: 
    model.save(run_name + ".h5")
    wandb.finish()
