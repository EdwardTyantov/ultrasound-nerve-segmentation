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

Generate submission:
```
python submission.py
```
