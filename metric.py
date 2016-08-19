import sys
import numpy as np
from keras import backend as K
smooth = 1


def mean_length_error(y_true, y_pred):
    y_true_f = K.sum(K.round(K.flatten(y_true)))
    y_pred_f = K.sum(K.round(K.flatten(y_pred)))
    delta = (y_pred_f - y_true_f)
    return K.mean(K.tanh(delta))

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def np_dice_coef(y_true, y_pred):
    tr = y_true.flatten()
    pr = y_pred.flatten()
    return (2. * np.sum(tr * pr) + smooth) / (np.sum(tr) + np.sum(pr) + smooth)


def main():
    a = np.random.random((420,100))
    b = np.random.random((420,100))
#    print a.flatten().shape
    res =  np_dice_coef(a,b )
    print res


if __name__ == '__main__':
    sys.exit(main())
    