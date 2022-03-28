from config import *
# from metrics import *
import random


def check_dir(_path):
    if not os.path.exists(_path):
        try:
            os.makedirs(_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def schedule_learning_rate(epoch):
    lr = lr_[int(epoch/scheduling_rate)]
    return lr


def data_random_shuffling(temp_type):
    global train_set, val_set, test_set, comp_set, path_read_train, path_read_val_test
    if temp_type != 'validation':
        if temp_type == 'train':
            path_read = path_read_train
        else:
            path_read = path_read_val_test

        images_s_src = [path_read + f for f
                        in os.listdir(path_read)
                        if f.endswith(('_short.jpg', '_short.JPG', '_short.png', '_short.PNG', '_short.TIF'))]
        images_s_src.sort()

        images_m_src = [path_read + f for f
                        in os.listdir(path_read)
                        if f.endswith(('_medium.jpg', '_medium.JPG', '_medium.png', '_medium.PNG', '_medium.TIF'))]
        images_m_src.sort()

        images_l_src = [path_read + f for f
                        in os.listdir(path_read)
                        if f.endswith(('_long.jpg', '_long.JPG', '_long.png', '_long.PNG', '_long.TIF'))]
        images_l_src.sort()

        images_g_src = [path_read + f for f
                        in os.listdir(path_read)
                        if f.endswith(('_groundtruth.jpg', '_groundtruth.JPG', '_groundtruth.png', '_groundtruth.PNG', '_groundtruth.TIF'))]
        images_g_src.sort()

        len_imgs_list = len(images_g_src)

        # generate random shuffle index list for all list
        tempInd = np.arange(len_imgs_list)
        random.shuffle(tempInd)

        images_s_src = np.asarray(images_s_src)[tempInd]
        images_m_src = np.asarray(images_m_src)[tempInd]

        images_l_src = np.asarray(images_l_src)[tempInd]
        images_g_src = np.asarray(images_g_src)[tempInd]

        for i in range(len_imgs_list):
            if temp_type == 'train':
                train_set.append([images_s_src[i], images_m_src[i], images_l_src[i],
                                  str(images_g_src[i])])
            elif temp_type == 'val':
                val_set.append([images_s_src[i], images_m_src[i], images_l_src[i],
                                str(images_g_src[i])])
            else:
                raise NotImplementedError
    else:
        path_read = 'valid/'
        images_s_src = [path_read + f for f
                        in os.listdir(path_read)
                        if f.endswith(('_short.jpg', '_short.JPG', '_short.png', '_short.PNG', '_short.TIF'))]
        images_s_src.sort()

        images_m_src = [path_read + f for f
                        in os.listdir(path_read)
                        if f.endswith(('_medium.jpg', '_medium.JPG', '_medium.png', '_medium.PNG', '_medium.TIF'))]
        images_m_src.sort()

        images_l_src = [path_read + f for f
                        in os.listdir(path_read)
                        if f.endswith(('_long.jpg', '_long.JPG', '_long.png', '_long.PNG', '_long.TIF'))]
        images_l_src.sort()

        len_imgs_list = len(images_l_src)

        # generate random shuffle index list for all list
        tempInd = np.arange(len_imgs_list)
        random.shuffle(tempInd)

        images_s_src = np.asarray(images_s_src)[tempInd]
        images_m_src = np.asarray(images_m_src)[tempInd]
        images_l_src = np.asarray(images_l_src)[tempInd]

        for i in range(len_imgs_list):
            comp_set.append(
                [images_s_src[i], images_m_src[i], images_l_src[i]])


# def test_generator(num_image):
#     in_img_tst = np.zeros((num_image, 3, img_h, img_w, nb_ch_all))

#     for i in range(num_image):
#         print('Read image: ', i, num_image)
#         in_img_tst[i, :, :, 0:3] = ((cv2.imread(test_set[i][1], color_flag)-src_mean)
#                                     / norm_val).reshape((img_h, img_w, nb_ch))
#         in_img_tst[i, :, :, 3:6] = ((cv2.imread(test_set[i][2], color_flag)-src_mean)
#     return in_img_tst


# def validation_generator(num_image):
#     in_img_tst=np.zeros((num_image, img_h, img_w, nb_ch_all))

#     for i in range(num_image):
#         print('Read image: ', i, num_image)
#         if resize_flag:
#             temp_img_l=cv2.imread(comp_set[i][0], color_flag)
#             size_set.append([temp_img_l.shape[1], temp_img_l.shape[0]])
#             if temp_img_l.shape[0] > temp_img_l.shape[1]:
#                 portrait_orientation_set.append(True)
#                 temp_img_l=cv2.rotate(
#                     temp_img_l, cv2.ROTATE_90_COUNTERCLOCKWISE)
#                 in_img_tst[i, :, :, 0:3]=(cv2.resize((temp_img_l-src_mean)/norm_val,
#                                                        (img_w, img_h))).reshape((img_h, img_w, nb_ch))
#                 temp_img_r=cv2.rotate(cv2.imread(
#                     comp_set[i][1], color_flag), cv2.ROTATE_90_COUNTERCLOCKWISE)
#                 in_img_tst[i, :, :, 3:6]=(cv2.resize((temp_img_r-src_mean)
#                                                        / norm_val, (img_w, img_h))).reshape((img_h, img_w, nb_ch))
#             else:
#                 portrait_orientation_set.append(False)
#                 in_img_tst[i, :, :, 0:3]=(cv2.resize((temp_img_l-src_mean)/norm_val,
#                                                        (img_w, img_h))).reshape((img_h, img_w, nb_ch))
#                 in_img_tst[i, :, :, 3:6]=(cv2.resize((cv2.imread(comp_set[i][1], color_flag)-src_mean)
#                                                        / norm_val, (img_w, img_h))).reshape((img_h, img_w, nb_ch))

#         else:
#             in_img_tst[i, :, :, 0:3]=((cv2.imread(comp_set[i][0], color_flag)-src_mean)
#                                         / norm_val).reshape((img_h, img_w, nb_ch))
#             in_img_tst[i, :, :, 3:6]=((cv2.imread(comp_set[i][1], color_flag)-src_mean)
#                                         / norm_val).reshape((img_h, img_w, nb_ch))
#     return in_img_tst


def generator(phase_gen='train'):
    if phase_gen == 'train':
        data_set_temp = train_set
        nb_total = total_nb_train
    elif phase_gen == 'val':
        data_set_temp = val_set
        nb_total = total_nb_val
    else:
        raise NotImplementedError

    image_counter = 0
    src_ims = np.zeros((img_mini_b, 3, patch_h, patch_w, nb_ch_all))
    trg_ims = np.zeros((img_mini_b, patch_h, patch_w, nb_ch))
    num_iter = 1
    while True:
        num_iter += 1
        if phase_gen == 'train' and num_iter == nb_train:
            np.random.shuffle(data_set_temp)
        for i in range(0, img_mini_b):
            img_data_src_s = data_set_temp[(image_counter + i) % (nb_total)][0]
            img_data_src_m = data_set_temp[(image_counter + i) % (nb_total)][1]
            img_data_src_l = data_set_temp[(image_counter + i) % (nb_total)][2]
            name = img_data_src_s.split(os.sep)[1].split('_')[0]
            align_ratio = np.load(os.path.join(
                'train', name + '_alignratio.npy'))

            exposures = np.load(os.path.join(
                'train', name + '_exposures.npy'))
            exposures = exposures - exposures[1]

            img_data_gt = data_set_temp[(image_counter + i) % (nb_total)][3]
            image_short = cv2.cvtColor(cv2.imread(
                img_data_src_s, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
            image_medium = cv2.cvtColor(cv2.imread(
                img_data_src_m, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
            image_long = cv2.cvtColor(cv2.imread(
                img_data_src_l, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0

            gamma = 2.24
            if random.random() < 0.3:
                gamma += (random.random() * 0.2 - 0.1)
            image_short_corrected = (
                ((image_short**gamma)*2.0**(-1*exposures[0]))**(1/gamma))
            image_medium_corrected = (
                ((image_medium**gamma)*2.0**(-1*exposures[1]))**(1/gamma))
            image_long_corrected = (
                ((image_long**gamma)*2.0**(-1*exposures[2]))**(1/gamma))
            output = cv2.cvtColor(cv2.imread(
                img_data_gt, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio

            src_ims[i, 0, :, :, 0:3] = image_short
            src_ims[i, 0, :, :, 3:6] = image_short_corrected
            src_ims[i, 1, :, :, 0:3] = image_medium
            src_ims[i, 1, :, :, 3:6] = image_medium_corrected
            src_ims[i, 2, :, :, 0:3] = image_long
            src_ims[i, 2, :, :, 3:6] = image_long_corrected
            trg_ims[i, :] = output
        X, y = random_flip(src_ims, trg_ims)
        X, y = random_rotate(X, y)
        yield (X, y)
        image_counter = (image_counter + img_mini_b) % (nb_total)


# def save_eval_predictions(path_to_save, test_imgaes, predictions, gt_images):
#     global mse_list, psnr_list, ssim_list, test_set
#     for i in range(len(test_imgaes)):
#         mse, psnr, ssim = MSE_PSNR_SSIM((gt_images[i]).astype(
#             np.float64), (predictions[i]).astype(np.float64))
#         mae = MAE((gt_images[i]).astype(np.float64),
#                   (predictions[i]).astype(np.float64))
#         mse_list.append(mse)
#         psnr_list.append(psnr)
#         ssim_list.append(ssim)
#         mae_list.append(mae)

#         temp_in_img = cv2.imread(test_set[i][0], color_flag)
#         if bit_depth == 8:
#             temp_out_img = (
#                 (predictions[i]*norm_val)+src_mean).astype(np.uint8)
#             temp_gt_img = ((gt_images[i]*norm_val)+src_mean).astype(np.uint8)
#         elif bit_depth == 16:
#             temp_out_img = (
#                 (predictions[i]*norm_val)+src_mean).astype(np.uint16)
#             temp_gt_img = ((gt_images[i]*norm_val)+src_mean).astype(np.uint16)
#         img_name = ((test_set[i][0]).split('/')[-1]).split('.')[0]
#         if resize_flag:
#             if portrait_orientation_set[i]:
#                 temp_out_img = cv2.resize(cv2.rotate(
#                     temp_out_img, cv2.ROTATE_90_CLOCKWISE), (size_set[i][0], size_set[i][1]))
#                 temp_gt_img = cv2.resize(cv2.rotate(
#                     temp_gt_img, cv2.ROTATE_90_CLOCKWISE), (size_set[i][0], size_set[i][1]))
#             else:
#                 temp_out_img = cv2.resize(
#                     temp_out_img, (size_set[i][0], size_set[i][1]))
#                 temp_gt_img = cv2.resize(
#                     temp_gt_img, (size_set[i][0], size_set[i][1]))
#         cv2.imwrite(path_to_save+str(img_name)+'_i.png', temp_in_img)
#         cv2.imwrite(path_to_save+str(img_name)+'_p.png', temp_out_img)
#         cv2.imwrite(path_to_save+str(img_name)+'_g.png', temp_gt_img)
#         print('Write image: ', i, len(test_imgaes))


def save_eval_comp(path_to_save, test_imgaes, predictions):
    global comp_set
    for i in range(len(test_imgaes)):
        bit_depth = 8
        norm_val = (2 ** bit_depth) - 1
        temp_out_img = ((predictions[i]*norm_val)+src_mean).astype(np.uint8)
        img_name = ((comp_set[i][0]).split('/')[-1]).split('.')[0]
        if resize_flag:
            if portrait_orientation_set[i]:
                temp_out_img = cv2.resize(cv2.rotate(
                    temp_out_img, cv2.ROTATE_90_CLOCKWISE), (size_set[i][0], size_set[i][1]))
            else:
                temp_out_img = cv2.resize(
                    temp_out_img, (size_set[i][0], size_set[i][1]))
        cv2.imwrite(path_to_save+str(img_name)[:-2]+'_g.png', temp_out_img)
        print('Write image: ', i, len(test_imgaes))


class DataGenerator():
    def __init__(self, batch_size, subset='train', shuffle=True):
        self.subset = subset
        if subset == 'train':
            self.images_dir = "train_jpg"
            self.data_ids = np.array([str(i) for i in sorted(os.listdir(
                os.path.join(self.images_dir))) if '_short.jpg' in i])
        elif subset == 'valid':
            self.images_dir = "valid_jpg"
            self.data_ids = np.array([str(i) for i in sorted(os.listdir(
                os.path.join(self.images_dir))) if '_short.jpg' in i])
        elif subset == 'test':
            self.images_dir = "valid"
            self.data_ids = np.array([str(i) for i in sorted(os.listdir(
                os.path.join(self.images_dir))) if '_short.jpg' in i])
        else:
            raise ValueError("subset must be 'train', 'valid' or 'test'")

        self.indices = np.arange(len(self.data_ids)).astype(np.uint32)
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.data_ids) / self.batch_size)

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        indexes = self.data_ids[inds]

        num_image = len(indexes)
        if self.subset == 'test':
            num_image = len(self.data_ids)
            src_ims = np.zeros((num_image, 3, img_h, img_w, nb_ch_all))
            trg_ims = np.zeros((num_image, img_h, img_w, nb_ch))
        else:
            src_ims = np.zeros((num_image, 3, patch_h, patch_w, nb_ch_all))
            trg_ims = np.zeros((num_image, patch_h, patch_w, nb_ch))
        for i in range(0, num_image):
            img_data_src_s = os.path.join(self.images_dir, self.data_ids[i])
            img_data_src_m = os.path.join(self.images_dir, self.data_ids[i]).replace(
                '_short.jpg', '_medium.jpg')
            img_data_src_l = os.path.join(self.images_dir, self.data_ids[i]).replace(
                '_short.jpg', '_long.jpg')
            img_data_gt = os.path.join(self.images_dir, self.data_ids[i]).replace(
                '_short.jpg', '_groundtruth.png')

            if phase_gen == 'train':
                align_ratio = np.load(img_data_src_s.replace(
                    'train_jpg', 'train'.replace(
                        '_short.jpg', '_alignratio.npy'
                    )))

                exposures = np.load(img_data_src_s.replace(
                    'train_jpg', 'train'.replace(
                        '_short.jpg', '_exposures.npy'
                    )))
            else:
                align_ratio = np.load(img_data_src_s.replace(
                    'valid_jpg', 'train'.replace(
                        '_short.jpg', '_alignratio.npy'
                    )))

                exposures = np.load(img_data_src_s.replace(
                    'valid_jpg', 'train'.replace(
                        '_short.jpg', '_exposures.npy'
                    )))
            exposures = exposures - exposures[1]
            if self.subset == 'valid':
                image_short = cv2.cvtColor(cv2.imread(
                    img_data_src_s, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
                image_medium = cv2.cvtColor(cv2.imread(
                    img_data_src_m, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
                image_long = cv2.cvtColor(cv2.imread(
                    img_data_src_l, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0

                gamma = 2.24
                image_short_corrected = (
                    ((image_short**gamma)*2.0**(-1*exposures[0]))**(1/gamma))
                image_medium_corrected = (
                    ((image_medium**gamma)*2.0**(-1*exposures[1]))**(1/gamma))
                image_long_corrected = (
                    ((image_long**gamma)*2.0**(-1*exposures[2]))**(1/gamma))
                output = cv2.cvtColor(cv2.imread(
                    img_data_gt, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio

                src_ims[i, 0, :, :, 0:3] = image_short
                src_ims[i, 0, :, :, 3:6] = image_short_corrected
                src_ims[i, 1, :, :, 3:6] = image_medium
                src_ims[i, 2, :, :, 3:6] = image_medium_corrected
                src_ims[i, 0, :, :, 0:3] = image_long
                src_ims[i, 0, :, :, 0:3] = image_long_corrected
                trg_ims[i, :] = output
            else:
                image_short = cv2.cvtColor(cv2.imread(
                    img_data_src_s, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
                image_medium = cv2.cvtColor(cv2.imread(
                    img_data_src_m, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0
                image_long = cv2.cvtColor(cv2.imread(
                    img_data_src_l, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0

                gamma = 2.24
                if random.random() < 0.3:
                    gamma += (random.random() * 0.2 - 0.1)
                image_short_corrected = (
                    ((image_short**gamma)*2.0**(-1*exposures[0]))**(1/gamma))
                image_medium_corrected = (
                    ((image_medium**gamma)*2.0**(-1*exposures[1]))**(1/gamma))
                image_long_corrected = (
                    ((image_long**gamma)*2.0**(-1*exposures[2]))**(1/gamma))
                output = cv2.cvtColor(cv2.imread(
                    img_data_gt, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio

                src_ims[i, 0, :, :, 0:3] = image_short
                src_ims[i, 0, :, :, 3:6] = image_short_corrected
                src_ims[i, 1, :, :, 3:6] = image_medium
                src_ims[i, 2, :, :, 3:6] = image_medium_corrected
                src_ims[i, 0, :, :, 0:3] = image_long
                src_ims[i, 0, :, :, 0:3] = image_long_corrected
                trg_ims[i, :] = output
        if self.shuffle:
            src_ims, trg_ims = random_flip(src_ims, trg_ims)
            src_ims, trg_ims = random_rotate(src_ims, trg_ims)
        # print(type(trg_ims))
        return src_ims, trg_ims

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indices)


# -----------------------------------------------------------
#  Transformations
# -----------------------------------------------------------

def random_crop(lr_img, hr_img, hr_crop_size=128):
    lr_crop_size = hr_crop_size
    lr_img_shape = tf.shape(lr_img)[:2]

    lr_w = tf.random.uniform(
        shape=(), maxval=lr_img_shape[1] - lr_crop_size + 1, dtype=tf.int32)
    lr_h = tf.random.uniform(
        shape=(), maxval=lr_img_shape[0] - lr_crop_size + 1, dtype=tf.int32)

    hr_w = lr_w
    hr_h = lr_h

    lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
    hr_img_cropped = hr_img[hr_h:hr_h + hr_crop_size, hr_w:hr_w + hr_crop_size]

    return lr_img_cropped, hr_img_cropped


def random_flip(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    for i in range(3):
        lr_img[:, i, :, :, :] = tf.cond(rn < 0.5,
                                        lambda: lr_img[:, i, :, :, :],
                                        lambda: tf.image.flip_left_right(lr_img[:, i, :, :, :]))
    hr_img = tf.cond(rn < 0.5,
                     lambda: hr_img,
                     lambda: tf.image.flip_left_right(hr_img))
    return lr_img, hr_img


def random_rotate(lr_img, hr_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    for i in range(3):
        lr_img[:, i, :, :, :] = tf.image.rot90(lr_img[:, i, :, :, :], rn)
    hr_img = tf.image.rot90(hr_img, rn)
    return lr_img, hr_img
