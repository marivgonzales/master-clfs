import numpy as np
from math import floor
from skimage.transform import resize
import gzip
import os

class Image:

    def __init__(self, raw_image, real_label, pred_label=None):
        self.raw_image = raw_image
        self.real_label = real_label
        self.pred_label = pred_label
    
    def get_raw_image(self):
        return self.raw_image

    def set_raw_image(self, new):
        self.raw_data = new

    def get_real_label(self):
        return self.real_label

    def set_real_label(self, new):
        self.real_label = new

    def get_pred_label(self):
        return self.pred_label

    def set_pred_label(self, new):
        self.pred_label = new

    def get_shape(self):
        return raw_image.shape

    def reshape_with_prop(self, target_size):
        current = self.raw_image
        new_data = np.ones((target_size[0], target_size[1]))
        
        majorside = np.amax(current.shape)
        majorside_idx = np.argmax(current.shape)
        minorside = np.amin(current.shape)

        factor = target_size[0]/majorside
        minorside_new = floor(minorside*factor)
        minorside_pad = floor((target_size[0] - minorside_new)/2)

        if majorside_idx == 0:
            current = resize(current, (target_size[0], minorside_new), mode='constant', anti_aliasing=True)
            for j in range(current.shape[0]):
                for k in range(current.shape[1]):
                    new_data[j,(k + minorside_pad)] = current[j,k]

        if majorside_idx == 1:
            current = resize(current, (minorside_new, target_size[1]), mode='constant', anti_aliasing=True)
            for j in range(current.shape[0]):
                for k in range(current.shape[1]):
                    new_data[(j + minorside_pad),k] = current [j,k]

        return new_data.astype('float32')

    def reshape_without_prop(self, target_size):
        new_data = resize(self.raw_image, target_size).astype('float32')
        return new_data

    #TO DO: method to get static of some image object, such as average pixel value or the image histogram.
    def get_statics(self):
        pass

    #TO DO: method to get focus measure of some image object.
    def get_focus_measure(self):
        pass

    #TO DO: method to get activation of some image object.
    def get_activation_map(self):
        pass

    #TO DO: method to plot all information of some image object.
    def plot_info(self):
        pass

class Data:

    def __init__(self, raw_data = None, label_data = None, number_of_images = None, train_indices = None, validation_indices = None):
        self.raw_data = raw_data
        self.label_data = label_data
        self.number_of_images = number_of_images
        self.train_indices = train_indices
        self.validation_indices = validation_indices

    def get_raw_data(self):
        return self.raw_data

    def get_label_data(self):
        return self.label_data

    def load(self, data_path, label_path, target_size):
        
        with gzip.open(label_path, "rb") as f:
            self.label_data = np.load(f)

        with gzip.open(data_path, "rb") as f:
            raw = np.load(f)

        self.number_of_images = len(raw)

        imgs_objects = np.empty((len(raw)))
        imgs_pixels = np.empty((len(raw)))
        for i in range(len(raw)):
            imgs_objects[i] = Image(raw[i], self.label_data[i])
            imgs_objects[i] = imgs_objects[i].reshape_with_prop(target_size)
            imgs_pixels[i] = imgs_objects[i].get_raw_image

        new_shape = (len(imgs_pixels), target_shape[0], target_shape[1], target_shape[2])
        imgs_pixels = np.reshape(imgs_pixels, new_shape)
        self.raw_data = imgs_pixels

        return imgs_objects

    def get_number_of_images(self):
        return self.number_of_images

    def get_validation_indices(self):
        return self.validation_indices

    def get_train_indices(self):
        return self.train_indices

    def set_train_val_indices(self, indices_train_path, indices_valid_path):
        self.train_indices = np.load(indices_train_path)
        self.validation_indices = np.load(indices_valid_path)

    def create_train_set(self):
        return self.raw_data[self.train_indices], self.label_data[self.train_indices]

    def create_valid_set(self):
        return self.raw_data[self.valid_indices], self.label_data[self.valid_indices]


