#http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-hvd-tf-multi-eng.html
#https://github.com/horovod/horovod/blob/master/examples/keras/keras_imagenet_resnet50.py


import os
import argparse
import keras
from keras import backend as K
from keras.preprocessing import image
import tensorflow as tf

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
parser.add_argument('--start_from', help='number of epochs',  default=0, type=int)
parser.add_argument('--ckpt_dir', help='Checkpoint directory', default="checkpoints")

parser.add_argument('--train-dir')
parser.add_argument('--val-dir')

args = parser.parse_args()


checkpoint_dir = args.ckpt_dir

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')


print("GPUS", len(gpus))
print(gpus)

os.environ['PYTHONHASHSEED']=str(args.seed)
os.environ['CUDA_VISIBLE_DEVICES']='1'

#os.environ['TF_DETERMINISTIC_OPS'] = '1'
tf.random.set_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


SEED = args.seed
GLOBAL_BATCH_SIZE  = args.batch 
batch = args.batch
epochs = args.epochs

LR_INIT = args.lr  #  tf.cast( args.lr * hvd.size(), dtype=tf.float32)
#LR_DECREASE_FACTOR = args.drop_by #5
#LR_DECREASE_EPOCHS = args.drop_on #[10, 20, 30]  

#init = INIT = GeometricInit3x3(rho=0.9, beta=0.9)

run_name = f"{args.model}_{args.n}_seed={SEED}"
callbacks = []

checkpoint_dir = checkpoint_dir +  f"/{run_name}"
callbacks.append(keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir + "/ckpt-{epoch}", save_freq="epoch"))
if args.start_from !=0 :
    latest_checkpoint =     os.path.join(checkpoint_dir,f"ckpt-{args.start_from}")
    print("latest_checkpoint", latest_checkpoint)
    model = keras.models.load_model(latest_checkpoint)
    print(f"Restoring from {latest_checkpoint}")
    
else:

    models = {'resnet50': ResNet50,
            'vgg16':   VGG16 }
            
    model = models[args.model](
        include_top=True,
        weights='imagenet',
        classes=1000,
        classifier_activation='softmax'
    )


    model = convert_model(model, n=args.n)


    d_opt = keras.optimizers.SGD(learning_rate=LR_INIT, momentum=0.9)


    model.compile(
            optimizer=d_opt,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            experimental_run_tf_function=False,
            metrics=[
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
            ])

verbose=1 #if hvd.rank() == 0 else 0



'''def lr_schedule(epoch):
    if epoch < 30:
        return 0.1
    if epoch < 60:
        return 0.01
    if epoch < 80:
        return 0.001
    return 0.0001'''

#callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0))






run = wandb.init(project="IMAGENET_Nparams_Experiments_VGG16", entity="geometric_init",
                        # track hyperparameters and run metadata
                    config={
                    "Batch_size": batch,
                    "Global_batch": GLOBAL_BATCH_SIZE,
                    "epochs": epochs,
                    "nparams" : str(args.n),
                    "LR": LR_INIT,
                    "model": args.model,
                    "Seed": SEED,
                    })

wandb.run.name = run_name 

    
#callbacks.append(layout_callback)
callbacks.append(WandbCallback(save_model= False))

# Configuration for creating new images
train_list = glob("/home/Common_datasets/imagenet/images/train" +'/*/*.jpg')
val_list = glob("/home/Common_datasets/imagenet/images/val" + '/*/*.jpg')


ilsvrc2012_categories = wf.get_ilsvrc2012_categories()
#print(ilsvrc2012_categories)
train_labels = [int(os.path.normpath(str(path)).split(os.path.sep)[-2]) for path in train_list]
val_labels = [int(os.path.normpath(str(path)).split(os.path.sep)[-2]) for path in val_list]


print("TRAIN list : ", len(train_list))
print("TRAIN LABELS : ", len(train_labels))
 
train_preprocess = Preprocessing(val=False).preprocess
trainDS = tf.data.Dataset.from_tensor_slices((train_list, train_labels))
trainDS = (trainDS
    .cache()
    .shuffle(trainDS.cardinality(), seed=SEED, reshuffle_each_iteration=True)
	.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
	.map(train_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
	.batch(args.batch, drop_remainder=True)
	.prefetch(tf.data.AUTOTUNE)
    .repeat() #

)

val_preprocess = Preprocessing(val=True).preprocess
valDS =tf.data.Dataset.from_tensor_slices((val_list, val_labels))
valDS = (valDS
    .cache()
	.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
	.map(val_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
	.batch(args.batch, drop_remainder=True)
	.prefetch(tf.data.AUTOTUNE)
    .repeat() #
)


print(model.summary())
initial_results = model.evaluate(valDS, verbose=verbose, steps= 50000 // GLOBAL_BATCH_SIZE)
wandb.log({"initial_val_loss": initial_results[0], "initial_val_accuracy": initial_results[1]})


history = model.fit(trainDS, 
                    steps_per_epoch=int(np.round(1281167 / GLOBAL_BATCH_SIZE)),
                    epochs=epochs, 
                    workers=8,
                    initial_epoch = args.start_from,
                    validation_data=valDS,
                    verbose=verbose,
                    validation_steps=2 * 50000 // GLOBAL_BATCH_SIZE,
                    callbacks=callbacks ) 

model.save(run_name + ".h5")
wandb.finish()
