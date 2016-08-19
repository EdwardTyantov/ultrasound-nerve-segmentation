from keras.callbacks import Callback
from keras.callbacks import warnings
import sys
import numpy as np
from keras import backend as K


class AdvancedLearnignRateScheduler(Callback):
    '''
    # Arguments
        monitor: quantity to be monitored.
        patience: number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: one of {auto, min, max}. In 'min' mode,
            training will stop when the quantity
            monitored has stopped decreasing; in 'max'
            mode it will stop when the quantity
            monitored has stopped increasing.
    '''
    def __init__(self, monitor='val_loss', patience=0,
                 verbose=0, mode='auto', decayRatio=0.5):
        super(Callback, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decayRatio = decayRatio

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Mode %s is unknown, '
                          'fallback to auto mode.'
                          % (self.mode), RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        current_lr = K.get_value(self.model.optimizer.lr)
        print(" \nLearning rate:", current_lr)
        if current is None:
            warnings.warn('AdvancedLearnignRateScheduler'
                          ' requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                assert hasattr(self.model.optimizer, 'lr'), \
                    'Optimizer must have a "lr" attribute.'
                current_lr = K.get_value(self.model.optimizer.lr)
                new_lr = current_lr * self.decayRatio
                if self.verbose > 0:
                    print(' \nEpoch %05d: reducing learning rate' % (epoch))
                    sys.stderr.write(' \nnew lr: %.5f\n' % new_lr)
                K.set_value(self.model.optimizer.lr, new_lr)
                self.wait = 0

            self.wait += 1


class LearningRateDecay(Callback):
    '''Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and returns a new
            learning rate as output (float).
    '''
    def __init__(self, decay, every_n=1, verbose=0):
        Callback.__init__(self)
        self.decay = decay
        self.every_n = every_n
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        if not (epoch and epoch % self.every_n == 0):
            return

        assert hasattr(self.model.optimizer, 'lr'), \
            'Optimizer must have a "lr" attribute.'
        current_lr = K.get_value(self.model.optimizer.lr)
        new_lr = current_lr * self.decay
        if self.verbose > 0:
            print(' \nEpoch %05d: reducing learning rate' % (epoch))
            sys.stderr.write('new lr: %.5f\n' % new_lr)
        K.set_value(self.model.optimizer.lr, new_lr)
