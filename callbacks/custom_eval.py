from skimage.filters import sobel_h
from skimage.filters import sobel_v
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import time
import io
from PIL import Image
from ..imagenet_utils import probabilities_to_decision

class Custom_Eval(tf.keras.callbacks.Callback):

    def __init__(self, wandb, model, tf_dataset, metric):
        self.wandb = wandb
        self.model = model
        self.m = model
        self.epochs = int(wandb.config['epochs'])
        self.cur_epoch = 0
        self.tf_dataset = tf_dataset
        self.metric = metric
    
    
    def on_test_end(self, logs=None):
        mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
        correct = 0
        count = 0
        for x,y in self.tf_dataset:
            res = mapping.probabilities_to_decision(self.model.predict(x).numpy())
            if res == y:
                correct+=1
            count+=1

        #
        #mappings = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()

        log_dict = {
            self.metric: correct/count
        }

        self.wandb.log(log_dict)