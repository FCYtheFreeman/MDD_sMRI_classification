# %%
import numpy as np
import os
import nibabel as nib
from math import ceil
# %%
batch_index = 0
val_batch_index = 0
test_batch_index = 0
width = 121
height = 145
depth = 121


def get_files(file_dir, ratio):
    '''
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    '''
    healthy_dir = 'health_GM/'
    MDD_dir = 'MDD_GM/'
    healthy = []
    label_healthy = []
    MDD = []
    label_MDD = []
    for file in os.listdir(file_dir + healthy_dir):
        healthy.append(file_dir + healthy_dir + file)
        label_healthy.append(0)
    for file in os.listdir(file_dir + MDD_dir):
        MDD.append(file_dir + MDD_dir + file)
        label_MDD.append(1)
    # print('**There are %d healthy\n**There are %d MDD' %(len(healthy), len(MDD)))

    image_list = np.hstack((healthy, MDD))
    label_list = np.hstack((label_healthy, label_MDD))

    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)  # 打乱顺序

    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    n_sample = len(all_label_list)
    n_val = ceil(n_sample * ratio)  # number of validation samples
    n_train = n_sample - n_val  # number of trainning samples

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(i) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(i) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


# %%

def get_batch( img_list, label_list, batch_size):

    global batch_index

    max = len(img_list)

    begin = batch_index
    end = batch_index + batch_size

    if end > max:  #剩余不够一个batch则舍弃
        batch_index = 0
        begin = batch_index
        end = batch_index + batch_size


    x_data = np.array([], np.float32)
    y_data = np.array([], np.int32) #
    #index = 0

    for i in range(begin, end):
        imagePath = img_list[i]
        FA_org = nib.load(imagePath)
        image = np.array(FA_org.get_data())  # 121x145x121; numpy.ndarray

        x_data = np.append(x_data, np.asarray(image, dtype='float32'))  # (image.data, dtype='float32')
        y_data = np.append(y_data, label_list[i])

    batch_index += batch_size  # update index for the next batch
    x_data_ = x_data.reshape(batch_size, height * width * depth)
    x_data_ = x_data_.T     # 转置方便计算
    x_mean = x_data_.mean(axis=0)
    # calculate variance
    x_std = x_data_.std(axis=0)
    # standardize
    x_data_scaled = (x_data_ - x_mean) / x_std  # standerdization:减去均值除以方差
    x_data_scaled = x_data_scaled.T

    return x_data_scaled, y_data

def get_val_batch( img_list, label_list, batch_size):

    global val_batch_index

    max = len(img_list)

    begin = val_batch_index
    end = val_batch_index + batch_size

    if end > max:  #剩余不够一个batch则舍弃
        batch_index = 0
        begin = batch_index
        end = batch_index + batch_size


    x_data = np.array([], np.float32)
    y_data = np.array([], np.int32) #
    #index = 0

    for i in range(begin, end):
        imagePath = img_list[i]
        FA_org = nib.load(imagePath)
        image = np.array(FA_org.get_data())  # 121x145x121; numpy.ndarray

        x_data = np.append(x_data, np.asarray(image, dtype='float32'))  # (image.data, dtype='float32')
        y_data = np.append(y_data, label_list[i])

    val_batch_index += batch_size  # update index for the next batch
    x_data_ = x_data.reshape(batch_size, height * width * depth)
    x_data_ = x_data_.T     # 转置方便计算
    x_mean = x_data_.mean(axis=0)
    # calculate variance
    x_std = x_data_.std(axis=0)
    # standardize
    x_data_scaled = (x_data_ - x_mean) / x_std  # standerdization:减去均值除以方差
    x_data_scaled = x_data_scaled.T

    return x_data_scaled, y_data


def get_test_batch(img_list, label_list, batch_size):
    global test_batch_index

    max = len(img_list)

    begin = test_batch_index
    end = test_batch_index + batch_size

    if end > max:  #剩余不够一个batch则舍弃
        batch_index = 0
        begin = batch_index
        end = batch_index + batch_size


    x_data = np.array([], np.float32)
    y_data = np.array([], np.int32) #
    #index = 0

    for i in range(begin, end):
        imagePath = img_list[i]
        FA_org = nib.load(imagePath)
        image = np.array(FA_org.get_data())  # 121x145x121; numpy.ndarray

        x_data = np.append(x_data, np.asarray(image, dtype='float32'))  # (image.data, dtype='float32')
        y_data = np.append(y_data, label_list[i])

    test_batch_index += batch_size  # update index for the next batch
    x_data_ = x_data.reshape(batch_size, height * width * depth)
    x_data_ = x_data_.T     # 转置方便计算
    x_mean = x_data_.mean(axis=0)
    # calculate variance
    x_std = x_data_.std(axis=0)
    # standardize
    x_data_scaled = (x_data_ - x_mean) / x_std  # standerdization:减去均值除以方差
    x_data_scaled = x_data_scaled.T

    return x_data_scaled, y_data
