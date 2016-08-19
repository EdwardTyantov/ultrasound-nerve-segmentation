from __future__ import print_function
import sys, os
import numpy as np
import cv2
from data import image_cols, image_rows, load_test_ids
from train import Learner


def prep(img):
    img = img.astype('float32')
    img = cv2.resize(img, (image_cols, image_rows)) 
    img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    return img

def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submission():
    imgs_id_test = load_test_ids()
    
    print ('Loading test_mask_res from %s' % Learner.test_mask_res)
    imgs_test = np.load(Learner.test_mask_res)
    print ('Loading imgs_exist_test from %s' % Learner.test_mask_exist_res)
    imgs_exist_test = np.load(Learner.test_mask_exist_res)

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]
    imgs_exist_test = imgs_exist_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in xrange(total):
        img = imgs_test[i, 0]
        img_exist = imgs_exist_test[i]
        img = prep(img)
        new_prob = (img_exist + min(1, np.sum(img)/10000.0 )* 5 / 3)/2
        if np.sum(img) > 0 and new_prob < 0.5:
            img = np.zeros((image_rows, image_cols))

        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 1000 == 0:
            print('{}/{}'.format(i, total))

    file_name = os.path.join(Learner.res_dir, 'submission.csv')

    with open(file_name, 'w+') as f:
        f.write('img,pixels\n')
        for i in xrange(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')

def main():
    submission()


if __name__ == '__main__':
    sys.exit(main())
