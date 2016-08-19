import sys, os
import numpy as np
from keras.preprocessing.image import (transform_matrix_offset_center, apply_transform, Iterator,
                                       random_channel_shift, flip_axis)
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)), '')
data_path = os.path.join(_dir, '../')
aug_data_path = os.path.join(_dir, 'aug_data')
aug_pattern = os.path.join(aug_data_path, 'train_img_%d.npy')
aug_mask_pattern = os.path.join(aug_data_path, 'train_mask_%d.npy')


def random_zoom(x, y, zoom_range, row_index=1, col_index=2, channel_index=0,
                fill_mode='nearest', cval=0.):
    if len(zoom_range) != 2:
        raise Exception('zoom_range should be a tuple or list of two floats. '
                        'Received arg: ', zoom_range)

    if zoom_range[0] == 1 and zoom_range[1] == 1:
        zx, zy = 1, 1
    else:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


def random_rotation(x, y, rg, row_index=1, col_index=2, channel_index=0,
                    fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


def random_shear(x, y, intensity, row_index=1, col_index=2, channel_index=0,
                 fill_mode='constant', cval=0.):
    shear = np.random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    y = apply_transform(y, transform_matrix, channel_index, fill_mode, cval)
    return x, y


class CustomNumpyArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='th'):
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        super(CustomNumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)


    def next(self):
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        batch_y_1, batch_y_2 = [], []
        for i, j in enumerate(index_array):
            x = self.X[j]
            y1 = self.y[0][j]
            y2 = self.y[1][j]
            _x, _y1 = self.image_data_generator.random_transform(x.astype('float32'), y1.astype('float32'))
            batch_x[i] = _x
            batch_y_1.append(_y1)
            batch_y_2.append(y2)
        return batch_x, [np.array(batch_y_1), np.array(batch_y_2)]
    

class CustomImageDataGenerator(object):
    def __init__(self, zoom_range=(1,1), channel_shift_range=0, horizontal_flip=False, vertical_flip=False,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 elastic=None,
):
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.elastic = elastic
    
    def random_transform(self, x, y, row_index=1, col_index=2, channel_index=0):
        
        if self.horizontal_flip:
            if True or np.random.random() < 0.5:
                x = flip_axis(x, 2)
                y = flip_axis(y, 2)
        
        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)
        
        h, w = x.shape[row_index], x.shape[col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        
        x = apply_transform(x, transform_matrix, channel_index,
                            fill_mode='constant')
        y = apply_transform(y, transform_matrix, channel_index,
                            fill_mode='constant')
        
        
        #        

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, 1)
                y = flip_axis(y, 1)
        
        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range)


        if self.elastic is not None:
            x, y = elastic_transform(x.reshape(h,w), y.reshape(h,w), *self.elastic)
            x, y = x.reshape(1, h, w), y.reshape(1, h, w)
        
        return x, y
    
    def flow(self, X, Y, batch_size, shuffle=True, seed=None):
        return CustomNumpyArrayIterator(
            X, Y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed)

        
def elastic_transform(image, mask, alpha, sigma, alpha_affine=None, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))


    res_x = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
    res_y = map_coordinates(mask, indices, order=1, mode='reflect').reshape(shape)
    return res_x, res_y


def test():
    X = np.random.randint(0,100, (1000, 1, 100, 200))
    YY = [np.random.randint(0,100, (1000, 1, 100, 200)), np.random.random((1000, 1))]
    cid = CustomImageDataGenerator(horizontal_flip=True)
    gen = cid.flow(X, YY, batch_size=64, shuffle=False)
    n = gen.next()[0]
    
    
if __name__ == '__main__':
    sys.exit(test())
