import numpy as np
import sys
from u_model import IMG_COLS as img_cols, IMG_ROWS as img_rows
from train import Learner

ensembles = {
             'ens2': (8, 'best/ens2/res3/'), 
             'ens3': (6, 'best/ens3/res3/'), 
             'ens4': (6, 'best/ens4/res3/'), 
             'ens5': (8, 'best/ens5/res3/'), 
             'ens7': (6, 'best/ens7/res3/'), 
             'ens8': (5, 'best/ens8/res3/'),  
             }


def main():
    kfold_masks, kfold_prob = [], []
    weigths = []
    for name, (kfold, prefix) in ensembles.iteritems():
        print 'Loading name=%s, prefix=%s, kfold=%d' % (name, prefix, kfold)
        ens_x_mask = np.load(prefix + 'imgs_mask_test.npy')
        ens_x_prob = np.load(prefix + 'imgs_mask_exist_test.npy')
        kfold_masks.append(ens_x_mask)
        kfold_prob.append(ens_x_prob)
        weigths.append(kfold)
    #
    total_weight = float(sum(weigths))
    total_cnt = len(weigths)
    dlen = len(kfold_masks[0])
    res_masks = np.ndarray((dlen, 1, img_rows, img_cols), dtype=np.float32)
    res_probs = np.ndarray((dlen, ), dtype=np.float32)
    
    for i in xrange(dlen):
        masks = np.ndarray((total_cnt, 1, img_rows, img_cols), dtype=np.float32)
        probs = np.ndarray((total_cnt, ), dtype=np.float32)
        for k in xrange(total_cnt):
            masks[k] = weigths[k] * kfold_masks[k][i]
            probs[k] = weigths[k] * kfold_prob[k][i]
        res_masks[i] = np.sum(masks, 0)/total_weight
        res_probs[i] = np.sum(probs)/total_weight
    print 'Saving', Learner.test_mask_res, Learner.test_mask_exist_res
    np.save(Learner.test_mask_res, res_masks)
    np.save(Learner.test_mask_exist_res, res_probs)
    

if __name__ == '__main__':
    sys.exit(main())

