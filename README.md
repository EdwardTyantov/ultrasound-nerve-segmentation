# ultrasound-nerve-segmentation
Kaggle Ultrasound Nerve Segmentation competition [Keras]

This code based on https://github.com/jocicmarko/ultrasound-nerve-segmentation/

#Install (Ubuntu, GPU )

###Theano
- http://deeplearning.net/software/theano/install_ubuntu.html#install-ubuntu
- sudo pip install pydot-ng

In ~/.theanorc
```
[global]
device = gpu0
[dnn]
enabled = True
```

###Keras
- sudo apt-get install libhdf5-dev
- sudo pip install h5py
- sudo pip install pydot
- sudo pip install nose_parameterized
- sudo pip install keras

In ~/.keras/keras.json
```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

#Prepare

Place train and test data into '../train' and '../test' folders accordingly.

```
mkdir np_data
python data.py
```

#Training

Single model training.
```
python train.py
```
Results will be generatated in "res/" folder. res/unet.hdf5 - best model

#Model

I used U-net (http://arxiv.org/abs/1505.04597) architecture. Main features:
 - inception blocks instead of VGG like
 - Conv with stride instead of MaxPooling
 - Dropout, p=0.5
 - shortcuts with residual blocks
 - BatchNorm everywhere
 - 2 heads training: one branch for scoring (in the middle of the network), on branch for segmentation
 - ELU activation
 - sigmoid in output 
 - Adam optimizer 
 - Dice coeff loss
 - output layers - sigmoid activation

Augmentation:
 - flip x,y
 - random zoom
 - random channel shift
 - elastic transormation didn't help in this configuration

Validation:
For some reason validation split by patient (which is right in this competition) didn't work for me, probably due to bug in the code. So I used random split.

Final prediction uses probability of nerve presence: (p_score + p_segment)/2, where p_segment based on number of output pixels in the mask.

Generate submission:
```
python submission.py
```

#Results and training aspects
- On GPU Titan X epoch took about 6 minutes. Best model on 16-28 epoch. 
- Best model achieved 0.694 LD score
- Ensemble of different k-fold ensembles (5,6,8) scored 0.70399
