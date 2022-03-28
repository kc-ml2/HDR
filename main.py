"""
This is the main module for linking different components of the CNN-based model
proposed for the task of image defocus deblurring based on dual-pixel data. 

Copyright (c) 2020-present, Abdullah Abuolaim
This source code is licensed under the license found in the LICENSE file in
the root directory of this source tree.

This code imports the modules and starts the implementation based on the
configurations in config.py module.

Note: this code is the implementation of the "Defocus Deblurring Using Dual-
Pixel Data" paper accepted to ECCV 2020. Link to GitHub repository:
https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel

Email: abuolaim@eecs.yorku.ca
"""
from model import *
from config import *
from data_generator import *

def train(configure):
    if op_phase == 'train':
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        data_random_shuffling('train')
        data_random_shuffling('val')
        model_x = Net(configure)

        in_data = Input(batch_shape=(None, 3, patch_h, patch_w, nb_ch_all))

        model = Model(inputs=in_data, outputs=model_x.main_model(in_data))
        model.load_weights(path_save_model)
        model.summary()
        model.compile(optimizer=Adam(lr=lr__[0]), loss=MAE)

        # training callbacks
        model_checkpoint = ModelCheckpoint(path_save_model, monitor='val_loss',
                                           verbose=1, save_best_only=True)

        l_r_scheduler_callback = LearningRateScheduler(
            schedule=schedule_learning_rate, verbose=True)

        history = model.fit_generator(generator('train'), nb_train, nb_epoch,
                                      validation_data=generator('val'),
                                      validation_steps=nb_val, callbacks=[model_checkpoint,
                                                                          l_r_scheduler_callback])

        np.save(path_write+'train_loss_arr', history.history['loss'])
        np.save(path_write+'val_loss_arr', history.history['val_loss'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters

    parser.add_argument('--filter', type=int, default=8)
    parser.add_argument('--attention_filter', type=int, default=16)
    parser.add_argument('--kernel', type=int, default=3)
    parser.add_argument('--encoder_kernel', type=int, default=3)
    parser.add_argument('--decoder_kernel', type=int, default=3)
    parser.add_argument('--triple_pass_filter', type=int, default=16)

    configure = parser.parse_args()

    train(configure)
