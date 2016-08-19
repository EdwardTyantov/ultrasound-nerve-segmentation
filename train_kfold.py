from optparse import OptionParser
import cv2, sys, os, shutil, random
import numpy as np
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import flip_axis, random_channel_shift
from keras.engine.training import slice_X
from keras_plus import LearningRateDecay
from u_model import get_unet, IMG_COLS as img_cols, IMG_ROWS as img_rows
from data import load_train_data, load_test_data, load_patient_num
from augmentation import CustomImageDataGenerator
from augmentation import random_zoom, elastic_transform, random_rotation
from utils import save_pickle, load_pickle, count_enum
from sklearn.cross_validation import KFold

_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), '')



def preprocess(imgs, to_rows=None, to_cols=None):
    if to_rows is None or to_cols is None:
        to_rows = img_rows
        to_cols = img_cols
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], to_rows, to_cols), dtype=np.uint8)
    for i in xrange(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (to_cols, to_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p

class Learner(object):
    
    suffix = '3'
    res_dir = os.path.join(_dir, 'res' + suffix)
    best_weight_path = os.path.join(res_dir, 'unet.hdf5')
    test_mask_res = os.path.join(res_dir, 'imgs_mask_test.npy')
    test_mask_exist_res = os.path.join(res_dir, 'imgs_mask_exist_test.npy')
    meanstd_path = os.path.join(res_dir, 'meanstd.dump')
    valid_data_path = os.path.join(res_dir, 'valid.npy')
    tensorboard_dir = os.path.join(res_dir, 'tb')
    
    def __init__(self, model_func, validation_split):
        self.model_func = model_func
        self.validation_split = validation_split
        self.__iter_res_dir = os.path.join(self.res_dir, 'res_iter')
        self.__iter_res_file = os.path.join(self.__iter_res_dir, '{epoch:02d}-{val_loss:.4f}.unet.hdf5')
        
    def _dir_init(self):
        if not os.path.exists(self.res_dir):
            os.mkdir(self.res_dir)
        #iter clean
        if os.path.exists(self.__iter_res_dir):
            shutil.rmtree(self.__iter_res_dir)
        os.mkdir(self.__iter_res_dir)
    
    def save_meanstd(self):
        data = [self.mean, self.std]
        save_pickle(self.meanstd_path, data)
        
    @classmethod
    def load_meanstd(cls):
        print ('Load meanstd from %s' % cls.meanstd_path)
        mean, std = load_pickle(cls.meanstd_path)
        return mean, std
    
    @classmethod
    def save_valid_idx(cls, idx):
        save_pickle(cls.valid_data_path, idx)
        
    @classmethod
    def load_valid_idx(cls):
        return load_pickle(cls.valid_data_path)
    
    def _init_mean_std(self, data):
        data = data.astype('float32')
        self.mean, self.std = np.mean(data), np.std(data)
        self.save_meanstd()
        return data
    
    def get_object_existance(self, mask_array):
        return np.array([int(np.sum(mask_array[i, 0]) > 0) for i in xrange(len(mask_array))])

    def standartize(self, array, to_float=False):
        if to_float:
            array = array.astype('float32')
        if self.mean is None or self.std is None:
            raise ValueError, 'No mean/std is initialised'
        
        array -= self.mean
        array /= self.std
        return array

    @classmethod
    def norm_mask(cls, mask_array):
        mask_array = mask_array.astype('float32')
        mask_array /= 255.0
        return mask_array

    @classmethod
    def shuffle_train(cls, data, mask):
        perm = np.random.permutation(len(data))
        data = data[perm]
        mask = mask[perm]
        return data, mask
        
    def __pretrain_model_load(self, model, pretrained_path):
        if pretrained_path is not None:
            if not os.path.exists(pretrained_path):
                raise ValueError, 'No such pre-trained path exists'
            model.load_weights(pretrained_path)
            
            
    def augmentation(self, X, Y):
        print('Augmentation model...')
        total = len(X)
        x_train, y_train = [], []
        
        for i in xrange(total):
            if i % 100 == 0:
                print ('Aug', i)
            x, y = X[i], Y[i]
            #standart
            x_train.append(x)
            y_train.append(y)
        
#            for _ in xrange(1):
#                _x, _y = elastic_transform(x[0], y[0], 100, 20)
#                x_train.append(_x.reshape((1,) + _x.shape))
#                y_train.append(_y.reshape((1,) + _y.shape))
            
            #flip x
            x_train.append(flip_axis(x, 2))
            y_train.append(flip_axis(y, 2))
            #flip y
            x_train.append(flip_axis(x, 1))
            y_train.append(flip_axis(y, 1))
            #continue
            #zoom
            for _ in xrange(1):
                _x, _y = random_zoom(x, y, (0.9, 1.1))
                x_train.append(_x)
                y_train.append(_y)
            for _ in xrange(0):
                _x, _y = random_rotation(x, y, 5)
                x_train.append(_x)
                y_train.append(_y)
            #intentsity
            for _ in xrange(1):
                _x = random_channel_shift(x, 5.0)
                x_train.append(_x)
                y_train.append(y)
    
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train
        
    def fit(self, x_train, y_train, nfolds=8):
        print('Creating and compiling and fitting model...')
        print('Shape:', x_train.shape)
        random_state = 51
        kf = KFold(len(x_train), n_folds=nfolds, shuffle=True, random_state=random_state)
        for i, (train_index, test_index) in enumerate(kf):
            print 'Fold %d' % i
            X_train, X_valid = x_train[train_index], x_train[test_index]
            Y_train, Y_valid = y_train[train_index], y_train[test_index]
            Y_valid_2 = self.get_object_existance(Y_valid)
            X_train, Y_train = self.augmentation(X_train, Y_train)
            Y_train_2 = self.get_object_existance(Y_train)
            #
            optimizer = Adam(lr=0.0045)
            model = self.model_func(optimizer)
            model_checkpoint = ModelCheckpoint(self.__iter_res_file + '_%d.fold' % i, monitor='val_loss')
            model_save_best = ModelCheckpoint(self.best_weight_path + '_%d.fold' % i, monitor='val_loss',
                                               save_best_only=True)
            early_s = EarlyStopping(monitor='val_loss', patience=8, verbose=1)
            #
            model.fit(
                       X_train, [Y_train, Y_train_2], 
                       validation_data=(X_valid, [Y_valid, Y_valid_2]),
                       batch_size=128, nb_epoch=40,
                       verbose=1, shuffle=True,
                       callbacks=[model_save_best, model_checkpoint, early_s]
                       ) 
        
        #augment
        return model

    def train_and_predict(self, pretrained_path=None):
        self._dir_init()
        print('Loading and preprocessing and standarize train data...')
        imgs_train, imgs_mask_train = load_train_data()
        imgs_train = preprocess(imgs_train)
        imgs_mask_train = preprocess(imgs_mask_train)
        imgs_mask_train = self.norm_mask(imgs_mask_train)
        
        self._init_mean_std(imgs_train)
        imgs_train = self.standartize(imgs_train, True)
        self.fit(imgs_train, imgs_mask_train)


def main():
    parser = OptionParser()
    parser.add_option("-s", "--suffix", action='store', type='str', dest='suffix', default = None)
    parser.add_option("-m", "--model_name", action='store', type='str', dest='model_name', default = 'u_model')
    #
    options, _ = parser.parse_args()
    suffix = options.suffix
    model_name = options.model_name
    if model_name is None:
        raise ValueError, 'model_name is not defined'
#    if suffix is None:
#        raise ValueError, 'Please specify suffix option'
#    print ('Suffix: "%s"' % suffix )
    #
    import imp
    model_ = imp.load_source('model_', model_name + '.py')
    model_func = model_.get_unet
    #
    lr = Learner(model_func, validation_split=0.2)
    lr.train_and_predict(pretrained_path=None)
    print ('Results in ', lr.res_dir)

if __name__ == '__main__':
    sys.exit(main())
