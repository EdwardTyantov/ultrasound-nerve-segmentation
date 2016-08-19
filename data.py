from __future__ import print_function
import os, sys
import numpy as np
import cv2

image_rows = 420
image_cols = 580

_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), '')
data_path = os.path.join(_dir, '../')
preprocess_path = os.path.join(_dir, 'np_data')
img_train_path = os.path.join(preprocess_path, 'imgs_train.npy')
img_train_mask_path = os.path.join(preprocess_path, 'imgs_mask_train.npy')
img_train_patients = os.path.join(preprocess_path, 'imgs_patient.npy')
img_test_path = os.path.join(preprocess_path, 'imgs_test.npy') 
img_test_id_path = os.path.join(preprocess_path, 'imgs_id_test.npy') 



def load_test_data():
    print ('Loading test data from %s' % img_test_path)
    imgs_test = np.load(img_test_path)
    return imgs_test

def load_test_ids():
    print ('Loading test ids from %s' % img_test_id_path)
    imgs_id = np.load(img_test_id_path)
    return imgs_id

def load_train_data():
    print ('Loading train data from %s and %s' % (img_train_path, img_train_mask_path))
    imgs_train = np.load(img_train_path)
    imgs_mask_train = np.load(img_train_mask_path)
    return imgs_train, imgs_mask_train

def load_patient_num():
    print ('Loading patient numbers from %s' % img_train_patients)
    return np.load(img_train_patients)

def get_patient_nums(string):
    pat, photo = string.split('_')
    photo = photo.split('.')[0]
    return int(pat), int(photo)

def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = filter((lambda image: 'mask' not in image), os.listdir(train_data_path))
    total = len(images) 

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    i = 0
    print('Creating training images...')
    img_patients = np.ndarray((total,), dtype=np.uint8)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        patient_num = image_name.split('_')[0]
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        imgs[i, 0] = img
        imgs_mask[i, 0] = img_mask
        img_patients[i] = patient_num
        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')
    np.save(img_train_patients, img_patients)
    np.save(img_train_path, imgs)
    np.save(img_train_mask_path, imgs_mask)
    print('Saving to .npy files done.')


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('Creating test images...')
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)

        imgs[i, 0] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(img_test_path, imgs)
    np.save(img_test_id_path, imgs_id)
    print('Saving to .npy files done.')


def main():
    create_train_data()
    create_test_data()

if __name__ == '__main__':
    sys.exit(main())
