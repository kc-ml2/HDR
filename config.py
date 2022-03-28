import argparse
import numpy as np
import os
import math
import cv2
import sys
import random
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras import Model
from src.metrics import mu_tonemap_tf

# results and model name
op_phase = 'train'

# image mini-batch size
img_mini_b = 8

#########################################################################
# READ & WRITE DATA PATHS									            #
#########################################################################

# path to save model
path_best_model = 'model_legacy/att_39.14.h5'
path_save_model = 'weights/att.h5'
path_save_model_finetune = 'weights/att_finetune.h5'
path_save_model_mu = 'weights/att_mu.h5'
path_save_model_all_data = 'weights/att_all.h5'


# paths to read data
path_read_train = 'train_jpg_2/'
path_read_val_test = 'valid_jpg_2/'

#########################################################################
# NUMBER OF IMAGES IN THE TRAINING, VALIDATION, AND TESTING SETS	    #
#########################################################################
if op_phase == 'train':
    total_nb_train = len([path_read_train + f for f
                          in os.listdir(path_read_train)
                          if f.endswith(('_short.jpg', '_short.JPG', '_short.png', '_short.PNG', '_short.TIF'))])

    total_nb_val = len([path_read_val_test + f for f
                        in os.listdir(path_read_val_test)
                        if f.endswith(('_short.jpg', '_short.JPG', '_short.png', '_short.PNG', '_short.TIF'))])

    # number of training image batches
    nb_train = int(math.ceil(total_nb_train/img_mini_b))
    # number of validation image batches
    nb_val = int(math.ceil(total_nb_val/img_mini_b))


elif op_phase == 'validation':
    total_nb_test = len([path_read_val_valid + f for f
                         in os.listdir(path_read_val_valid)
                         if (f.endswith(('_short.jpg', '_short.JPG', '_short.png', '_short.PNG', '_short.TIF')))])

#########################################################################
# MODEL PARAMETERS & TRAINING SETTINGS									#
#########################################################################

# input image size
img_w = 1900
img_h = 1060

# input patch size
patch_w = 320
patch_h = 320


# number of epochs
nb_epoch = 300

# number of input channels
nb_ch_all = 6
# number of output channels
nb_ch = 3  # change conv9 in the model and the folowing variable

# after how many epochs you change learning rate
scheduling_rate = 30

dropout_rate = 0.4

# generate learning rate array
lr_ = []
lr_.append(1e-4)  # initial learning rate
for i in range(int(nb_epoch/scheduling_rate)):
    lr_.append(lr_[i]*0.5)

train_set, val_set, test_set, comp_set = [], [], [], []

size_set, portrait_orientation_set = [], []

mse_list, psnr_list, ssim_list, mae_list = [], [], [], []


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch -
                                                LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)
    return lr


def step_decay_schedule(epoch):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    initial_lr = 1e-5
    decay_factor = 0.5
    step_size = 2
    return initial_lr * (decay_factor ** np.floor(epoch/step_size))


def loss_function(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    mse = tf.reduce_mean(squared_difference, axis=-1)
    ssim = SSIMLoss(y_true, y_pred)

    total_loss = 100*mse + 5*ssim
    return total_loss


def MAE(y_true, y_pred):
    squared_difference = tf.abs(y_true - y_pred)
    mae = tf.reduce_mean(squared_difference, axis=-1)
    return mae

def MAE_mu(y_true, y_pred):
    y_true, y_pred = mu_tonemap_tf(y_true), mu_tonemap_tf(y_pred)
    squared_difference = tf.abs(y_true - y_pred)
    mae = tf.reduce_mean(squared_difference, axis=-1)
    return mae


def loss_function_2(y_true, y_pred):
    absolute_difference = tf.abs(y_true - y_pred)
    mse = tf.reduce_mean(absolute_difference, axis=-1)
    ssim = SSIMLossMS(y_true, y_pred)

    total_loss = 5*mse + 1*ssim
    return total_loss


def SSIMLoss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))


def SSIMLossMS(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, 1.0))


lr__ = []
lr__.append(1e-5)
