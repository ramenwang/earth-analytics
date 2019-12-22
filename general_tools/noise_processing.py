import numpy as np
import os
import cv2

class noisy():
    '''
    an object to apply noisy function onto imagery
    '''

    def __init__(self, image, pixel_type='8 bit'):
        '''
        :param image (ndarray) with shape of (row,col,ch)
        '''
        self.image = image
        if pixel_type == '8 bit':
            self.px_min, self.px_max = 0, 255
        elif pixel_type == '16 bit':
            self.px_min, self.px_max = 0, 65535


    def gaussian(self, mean=0, std=1):
        '''
        a function to apply gaussian noise on image

        :param mean (float): mean value for gaussian noise, default = 0
        :param std (float): standard deviation for gaussian noise, default = 1
        :return an image with gaussian noise
        '''
        gauss_noise = np.random.normal(mean, std, self.image.shape)
        return self.image + gauss_noise


    def salt_pepper(self, sp_rate=0.5, spread=0.04):
        '''
        a function to spray salt and pepper noise

        :param sp_rate (float): the ratio of salt to pepper, default is 0.5
        :param spread (float): the propotion of spray
        :return an image with salt and pepper
        '''
        out_img = self.image
        img_size = out_img.size
        # spray salt
        salt_con = np.ceil(img_size * spread * sp_rate)
        coords = [np.random.randint(0, i - 1, int(salt_con))
                  for i in out_img.shape]
        out_img[coords] = self.px_max
        # spray pepper
        pepper_con = np.ceil(img_size * spread * (1-sp_rate))
        coords = [np.random.randint(0, i-1, int(pepper_con))
                  for i in out_img.shape]
        out_img[coords] = self.px_min

        return out_img


    def poisson(lamda):
        uni_vals = len(np.unique(self.image))
        uni_vals = 2 ** np.ceil(np.log2(uni_vals))

        return np.random.poisson(self.image * uni_vals) / float(vals)



    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)
        noisy = image + image * gauss
        return noisy
