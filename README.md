# Ultrasound nerve segmentation using Keras
Kaggle Ultrasound Nerve Segmentation competition [Keras]

#Install (Ubuntu {14,16}, GPU)

cuDNN required.

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

In ~/.keras/keras.json (it's very important, the project was running on theano backend, and some issues are possible in case of TensorFlow)
```
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

###Python deps
 - sudo apt-get install python-opencv
 - sudo apt-get install python-sklearn

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

Generate submission:
```
python submission.py
```

Generate predection with a model in res/unet.hdf5
``` 
python current.py
```

#Model

Motivation's explained in my internal pres (slides: http://www.slideshare.net/Eduardyantov/ultrasound-segmentation-kaggle-review)

I used U-net like architecture (http://arxiv.org/abs/1505.04597). Main differences:
 - inception blocks instead of VGG like
 - Conv with stride instead of MaxPooling
 - Dropout, p=0.5
 - skip connections from encoder to decoder layers with residual blocks
 - BatchNorm everywhere
 - 2 heads training: auxiliary branch for scoring nerve presence (in the middle of the network), one branch for segmentation
 - ELU activation
 - sigmoid activation in output 
 - Adam optimizer, without weight regularization in layers
 - Dice coeff loss, average per batch, without smoothing
 - output layers - sigmoid activation
 - batch_size=64,128 (for GeForce 1080 and Titan X respectively)

Augmentation:
 - flip x,y
 - random zoom
 - random channel shift
 - elastic transormation didn't help in this configuration

Augmentation generator (generate augmented data on the fly for each epoch) didn't improve the score. 
For prediction augmented images were used.

Validation:

For some reason validation split by patient (which is proper in this competition) didn't work for me, probably due to bug in the code. So I used random split.

Final prediction uses probability of a nerve presence: p_nerve = (p_score + p_segment)/2, where p_segment based on number of output pixels in the mask.

#Results and technical aspects
- On GPU Titan X an epoch took about 6 minutes. Training early stops at 15-30 epochs.
- For batch_size=64 6Gb GPU memory is required.
- Best single model achieved 0.694 LB score.
- An ensemble of 6 different k-fold ensembles (k=5,6,8) scored 0.70399

#Credits
This code was originally based on https://github.com/jocicmarko/ultrasound-nerve-segmentation/
