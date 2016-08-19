import numpy as np
import sys
from data import load_test_data
from u_model import get_unet
from keras.optimizers import Adam
from train import preprocess, Learner
#aug
from u_model import IMG_COLS as img_cols, IMG_ROWS as img_rows
from keras.preprocessing.image import flip_axis, random_channel_shift

curry = lambda func, *args, **kw:\
            lambda *p, **n:\
                 func(*args + p, **dict(kw.items() + n.items()))
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center


def zoom(x, zoom_range, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0.):
    zx, zy = zoom_range
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


transforms = (
              {'do': curry(flip_axis, axis=1), 'undo': curry(flip_axis, axis=1)},
              {'do': curry(flip_axis, axis=2), 'undo': curry(flip_axis, axis=2)},
              {'do': curry(zoom, zoom_range=(1.05, 1.05)), 'undo': curry(zoom, zoom_range=(1/1.05, 1/1.05))},
              {'do': curry(zoom, zoom_range=(0.95, 0.95)), 'undo': curry(zoom, zoom_range=(1/0.95, 1/0.95))},
              {'do': curry(random_channel_shift, intensity=5), 'undo': lambda x: x},
              )


def run_test():
    BS = 128
    print('Loading and preprocessing test data...')
    mean, std = Learner.load_meanstd()
    
    imgs_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('Loading saved weights...')
    model = get_unet(Adam(0.001))
    print ('Loading weights from %s' % Learner.best_weight_path)
    model.load_weights(Learner.best_weight_path)
    
    print ('Augment')
    alen, dlen = len(transforms), len(imgs_test)
    test_x = np.ndarray((alen, dlen, 1, img_rows, img_cols), dtype=np.float32)
    for i in xrange(dlen):
        for j, transform in enumerate(transforms):
            test_x[j,i] = transform['do'](imgs_test[i].copy())
    #
    print('Predicting masks on test data...')
    outputs = []
    asis_res = model.predict(imgs_test, batch_size=BS, verbose=1)
    outputs.append(asis_res)
    for j, transform in enumerate(transforms):
        t_y = model.predict(test_x[j], batch_size=BS, verbose=1)
        outputs.append(t_y)
    #
    print('Analyzing')
    test_masks = np.ndarray((dlen, 1, img_rows, img_cols), dtype=np.float32)
    test_probs = np.ndarray((dlen, ), dtype=np.float32)
    for i in xrange(dlen):
        masks = np.ndarray((alen+1, 1, img_rows, img_cols), dtype=np.float32)
        probs = np.ndarray((alen+1, ), dtype=np.float32)
        for j, t_y in enumerate(outputs):
            mask, prob = t_y[0][i], t_y[1][i]
            if j:
                mask = transforms[j-1]['undo'](mask)
            masks[j] = mask
            probs[j] = prob
        #
        test_masks[i] = np.mean(masks, 0)
        test_probs[i] = np.mean(probs)
            
    print('Saving ')
    np.save(Learner.test_mask_res, test_masks)
    np.save(Learner.test_mask_exist_res, test_probs)

def main():
    run_test()

if __name__ == '__main__':
    sys.exit(main())
