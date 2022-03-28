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
        sh_name = gt_name.replace("gt", "short")
        me_name = gt_name.replace("gt", "medium")
        lo_name = gt_name.replace("gt", "long")

        image_short = cv2.imread(
            sh_name, cv2.IMREAD_UNCHANGED)
        image_medium = cv2.imread(
            me_name, cv2.IMREAD_UNCHANGED)
        image_long = cv2.imread(
            lo_name, cv2.IMREAD_UNCHANGED)
        output = cv2.imread(
            gt_name, cv2.IMREAD_UNCHANGED)

        size = 320
        step = 160
        count = 0
        for m in range(5):
            for n in range(10):
                a = step*m
                b = step*n
                # batch_np_rt = np.zeros([4, size, size, 6])
                sh = image_short[a:a+size, b:b+size, :]
                me = image_medium[a:a+size, b:b+size, :]
                lo = image_long[a:a+size, b:b+size, :]
                ou = output[a:a+size, b:b+size, :]
                if train:
                    cv2.imwrite(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/train_jpg_2/{}_{}_short.png'.format(name, count), sh)
                    cv2.imwrite(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/train_jpg_2/{}_{}_medium.png'.format(name, count), me)
                    cv2.imwrite(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/train_jpg_2/{}_{}_long.png'.format(name, count), lo)
                    cv2.imwrite(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/train_jpg_2/{}_{}_groundtruth.png'.format(name, count), ou)
                else:
                    cv2.imwrite(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/valid_jpg_2/{}_{}_short.png'.format(name, count), sh)
                    cv2.imwrite(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/valid_jpg_2/{}_{}_medium.png'.format(name, count), me)
                    cv2.imwrite(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/valid_jpg_2/{}_{}_long.png'.format(name, count), lo)
                    cv2.imwrite(
                        '/home/tuvv/Tu/HDR/ANL-HDRI/valid_jpg_2/{}_{}_groundtruth.png'.format(name, count), ou)
                count += 1


def distortion(imgs):
    distortions = tf.random.uniform([2], 0, 1.0, dtype=tf.float32)

    # flip horizontally
    imgs = tf.cond(tf.less(distortions[0], 0.5), lambda: tf.image.flip_left_right(
        imgs), lambda: imgs)

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
