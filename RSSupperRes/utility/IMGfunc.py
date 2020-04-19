import numpy as np
import os
import imageio
from PIL import Image
from keras.utils import Sequence

def pixel_trimmer(image):
    ''' trim pixel values to the range of 0, 255
    '''
    image = np.round(image)
    return np.clip(image, 0, 255)


def get_mse(image1, image2, border_size=0):
    ''' compute mean square error between two images
    :param image1, image2: two input images
    :param border_size: the margin being ignored during the mse computation
    :return: mean square error
    :rtype: constant
    '''
    if image1.shape[0] != image2.shape[0] or image1.shape[1] != image2.shape[1]:
        return None

    if len(image1.shape) == 2:
        image1 = image1.reshape(image1.shape[0], image1.shape[1], 1)
    if len(image2.shape) == 2:
        image2 = image2.reshape(image2.shape[0], image2.shape[2], 1)

    image1 = pixel_trimmer(image1)
    image2 = pixel_trimmer(image2)

    diff = np.subtract(image1, image2)
    if border_size>0:
        # trim image into wanted border_size
        diff = diff[border_size: -border_size, border_size: -border_size]

    return np.mean(np.square(diff))


def img_rescale(image, scale, resampling_method):
    ''' rescale img based on different methods
    :param image: input image [rows, cols, channels]
    :param scale: a scale factor from input size to target size
    :param resampling_method: could be either bicubic, bilinear, or nearest_neighbor
    :return: a rescaled image based on basic resampling technique
    :rtype: np.array
    '''
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)

    if resampling_method == "bicubic":
        method = Image.BICUBIC
    elif resampling_method == "bilinear":
        method = Image.BILINEAR
    elif resampling_method == "nearest_neighbor":
        method = Image.NEAREST
    else:
        method = Image.LANCZOS

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # the image may has an alpha channel
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)

    return image


class DataGenerator(Sequence):
    """ Generates data for RSSuperRes
    """
    def __init__(self, img_list, x_dir, y_dir, input_dim, scale, n_channels,
                 batch_size=32, to_fit=True, shuffle=True):
        """ Initialization
        :param img_list: list of image files to use in the generator
        :param x_dir: path to input images location
        :param y_dir: path to true images location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param input_dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param scale: the scale factor maps the input size to output size
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.img_list = img_list
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.x_dim = np.array(input_dim)
        self.y_dim = np.array(input_dim) * scale
        self.scale = scale
        self.n_channels = n_channels
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.img_list) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[(index * self.batch_size):((index + 1) * self.batch_size)]
        img_files = [self.img_list[k] for k in indexes]

        # Generate data
        X = self._generate_X(img_files)
        if self.to_fit:
            y = self._generate_y(img_files)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.img_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, img_files):
        """Generates data containing batch_size images
        :param img_files: list of low resolution images
        :return: batch of images
        """
        X = np.empty((self.batch_size, *self.x_dim, self.n_channels))
        for i, iFile in enumerate(img_files):
             tmp_img = imageio.imread(self.x_dir + "/" + iFile).reshape((*self.x_dim, self.n_channels))
             X[i,] = pixel_trimmer(tmp_img) / 255

        return X

    def _generate_y(self, img_files):
        """Generates data containing batch_size residual image
        :param img_files: list of label ids to load
        :return: batch of residual images
        """
        y = np.empty((self.batch_size, *self.y_dim, self.n_channels))
        for i, iFile in enumerate(img_files):
            tmp_img_x = imageio.imread(self.x_dir + "/" + iFile).reshape((*self.x_dim, self.n_channels))
            tmp_img_y = imageio.imread(self.y_dir + "/" + iFile).reshape((*self.y_dim, self.n_channels))
            tmp_img_x = img_rescale(tmp_img_x, self.scale, 'bicubic')
            y[i,] = pixel_trimmer(tmp_img_y)/255. - pixel_trimmer(tmp_img_x)/255.

        return y
