import os
import cv2
import argparse
import numpy as np

from config import *
from model import *
from tqdm import tqdm
from src import data_io as io
from tensorflow.keras import Model, Input


def get_test_data(images_path, number):
    images_path = os.path.join(images_path, "{}_gt.png".format(number))
    exposures = np.load(images_path.replace("gt.png", "exposures.npy"))
    exposures = exposures - exposures[1]
    sh_name = images_path.replace("gt", "short")
    me_name = images_path.replace("gt", "medium")
    lo_name = images_path.replace("gt", "long")

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
    sh = np.expand_dims(image_short_corrected, axis=0)
    me = np.expand_dims(image_medium_corrected, axis=0)
    lo = np.expand_dims(image_long_corrected, axis=0)

    sh_l = np.expand_dims(image_short, axis=0)
    me_l = np.expand_dims(image_medium, axis=0)
    lo_l = np.expand_dims(image_long, axis=0)

    shc = np.concatenate([sh_l, sh], axis=-1)
    mec = np.concatenate([me_l, me], axis=-1)
    loc = np.concatenate([lo_l, lo], axis=-1)

    imgs_np = np.concatenate([shc, mec, loc], axis=0)
    imgs_np = np.expand_dims(imgs_np, axis=0)

    return imgs_np


def tonemap(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)


def run(config, model):
    # namelist = ["%04d" % i for i in range(100)]
    namelist = ["%04d" % i for i in range(0,200,3)]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu)
    for name in tqdm(namelist):
        SDR = get_test_data(config.valid_path, name)
        rs = model.predict(SDR)
        out = rs[0]
        io.imwrite_uint16_png(config.save_path+"{}.png".format(
            name), out, config.save_path+"{}_alignratio.npy".format(name))
        # out = tonemap(out**2.24)
        # out[out > 1.0] = 1.0
        # cv2.imwrite(os.path.join(config.save_path, 'hdr_{}.jpg'.format(name)),
        #             cv2.cvtColor(np.uint8(out*255), cv2.COLOR_BGR2RGB))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--valid_path', type=str, default="testdata/")
    parser.add_argument('--save_path', type=str, default="submit_test/")
    parser.add_argument('--filter', type=int, default=8)
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--attention_filter', type=int, default=16)
    parser.add_argument('--kernel', type=int, default=3)
    parser.add_argument('--encoder_kernel', type=int, default=3)
    parser.add_argument('--decoder_kernel', type=int, default=3)
    parser.add_argument('--triple_pass_filter', type=int, default=16)
    parser.add_argument('--zip_name', type=str, default='submit')

    config = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    model_x = Net(config)
    in_data = Input(batch_shape=(None, 3, 1060, 1900, nb_ch_all))
    model = Model(inputs=in_data, outputs=model_x.main_model(in_data))
    model.load_weights('weights/atnlc.h5')

    run(config, model)
    os.system("cd submit_test/ && zip -r ..//submission_{}.zip *".format(config.zip_name))
