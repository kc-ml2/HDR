import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
import tensorflow as tf


def run(file_list, train=True):
    list_gt = file_list
    for i in tqdm(range(len(list_gt))):
        gt_name = list_gt[i]
        name = gt_name.split('/')[-1][:4]
        exposures = np.load(gt_name.replace("gt.png", "exposures.npy"))
        exposures = exposures - exposures[1]
        alignratio = np.load(gt_name.replace("gt.png", "alignratio.npy"))
        sh_name = gt_name.replace("gt", "short")
        me_name = gt_name.replace("gt", "medium")
        lo_name = gt_name.replace("gt", "long")

        image_short = cv2.cvtColor(cv2.imread(
            sh_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
        image_medium = cv2.cvtColor(cv2.imread(
            me_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
        image_long = cv2.cvtColor(cv2.imread(
            lo_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0

        gamma = 2.24
        image_short_corrected = (
            ((image_short**gamma)*2.0**(-1*exposures[0]))**(1/gamma))
        image_medium_corrected = (
            ((image_medium**gamma)*2.0**(-1*exposures[1]))**(1/gamma))
        image_long_corrected = (
            ((image_long**gamma)*2.0**(-1*exposures[2]))**(1/gamma))
        output = cv2.cvtColor(cv2.imread(
            gt_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / alignratio

        size = 320
        for m in range(3):
            for n in range(5):
                a = size*m
                b = size*n
                # batch_np_rt = np.zeros([4, size, size, 6])
                sh = np.expand_dims(
                    image_short_corrected[a:a+size, b:b+size, :], axis=0)
                me = np.expand_dims(
                    image_medium_corrected[a:a+size, b:b+size, :], axis=0)
                lo = np.expand_dims(
                    image_long_corrected[a:a+size, b:b+size, :], axis=0)

                sh_l = np.expand_dims(image_short[a:a+size, b:b+size, :], axis=0)
                me_l = np.expand_dims(
                    image_medium[a:a+size, b:b+size, :], axis=0)
                lo_l = np.expand_dims(image_long[a:a+size, b:b+size, :], axis=0)

                ou = np.expand_dims(output[a:a+size, b:b+size, :], axis=0)

                shc = np.concatenate([sh_l, sh], axis=-1)
                mec = np.concatenate([me_l, me], axis=-1)
                loc = np.concatenate([lo_l, lo], axis=-1)
                ouc = np.concatenate([ou, ou], axis=-1)

                # X = np.concatenate([shc, mec, loc, ouc], axis=0)
                X = np.concatenate([sh, me, lo, ou], axis=-1)
                X_rt = distortion(X)
                X_rt = np.expand_dims(X_rt, axis=0)
                j = m*5+n
                if train:
                    np.save(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/train_npy/{}_{}.npy'.format(name, j), X)
                    np.save(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/train_npy/{}_{}_dt.npy'.format(name, j), X_rt)
                else:
                    np.save(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/valid_npy/{}_{}.npy'.format(name, j), X)
                    np.save(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/valid_npy/{}_{}_dt.npy'.format(name, j), X_rt)

def distortion(imgs):
    distortions = tf.random.uniform([2], 0, 1.0, dtype=tf.float32)

    # flip horizontally
    imgs = tf.cond(tf.less(distortions[0],0.5), lambda: tf.image.flip_left_right(imgs), lambda: imgs)

    # rotate
    k = tf.cast(distortions[1]*4+0.5, tf.int32)
    imgs = tf.image.rot90(imgs, k)

    return imgs


def create_dataset(raw_path, split=0.8):
    list_gt = sorted(glob.glob(raw_path + "*_gt.png"))
    np.random.shuffle(list_gt)
    num_train_files = round(len(list_gt) * split)
    train_list = list_gt[:num_train_files]
    valid_list = list_gt[num_train_files:]

    print("Preparing training dataset...")
    run(train_list, True)
    print("Preparing validation dataset...")
    run(valid_list, False)


create_dataset('train/')
