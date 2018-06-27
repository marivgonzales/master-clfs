import glob
import numpy as np
from skimage.transform import resize
from skimage.io import imread
from math import floor
import os
import sys
import skimage.io
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

def load(paths):
    """
    Load all images into memory for faster processing
    """
    images = np.empty(len(paths), dtype='object')
    for k, path in enumerate(paths):
        img = skimage.io.imread(path, as_grey=True)
        images[k] = img

    """
    Adjust the size of images
    """
    target_shape = (95, 95, 1)
    new_imgs = np.zeros((len(images), target_shape[0], target_shape[1]))

    for l in range(len(images)):
        current = images[l]
        majorside = np.amax(current.shape)
        majorside_idx = np.argmax(current.shape)
        minorside = np.amin(current.shape)

        factor = target_shape[0]/majorside
        minorside_new = floor(minorside*factor)

        if majorside_idx == 0:
            current = resize(current, (target_shape[0],minorside_new), mode='reflect')

        if majorside_idx == 1:
            current = resize(current, (minorside_new, target_shape[1]), mode='reflect')

        for i in range(current.shape[0]):
            for j in range(current.shape[1]):
                new_imgs[l,i,j] = current[i,j]

    return new_imgs

# FIXME: better command line arg parsing?
def main():
    if len(sys.argv) < 2:
        print("usage: create_dataset.py basedir")
        sys.exit(1)

    basedir = sys.argv[1]

    if basedir[-1] != '/':
        basedir = basedir + "/"


    create_validation = False
    if len(sys.argv) > 2 and sys.argv[2] == '--validation':
        # 
        create_validation = True

    directories = glob.glob(basedir + "train/*")
    class_names = [os.path.basename(d) for d in directories]
    class_names.sort()
    num_classes = len(class_names)

    paths_train = glob.glob(basedir + "train/*/*")
    paths_train.sort()

    paths_test = glob.glob(basedir + "test/*")
    paths_test.sort()

    labels_train = np.zeros(len(paths_train), dtype='int32')
    for k, path in enumerate(paths_train):
        class_name = os.path.basename(os.path.dirname(path))
        labels_train[k] = class_names.index(class_name)

    print("Saving train labels")
    np.save(basedir + "labels_train.npy", labels_train)
    print("Gzipping train labels")
    os.system("gzip " + basedir + "labels_train.npy")

    print("Loading train images")
    images_train = load(paths_train)
    print("Saving train images")
    np.save(basedir +  "images_train.npy", images_train)
    del images_train
    print("Gzipping train images")
    os.system("gzip " + basedir + "images_train.npy")

    print("Loading test images")
    images_test = load(paths_test)
    np.save(basedir + "images_test.npy", images_test)
    del images_test
    print("Gzipping test images")
    os.system("gzip " + basedir + "images_test.npy")

    print("Done")


    if create_validation == True:
        split = StratifiedShuffleSplit(n_splits=1,test_size=0.1)
        indices_train, indices_valid = next(split.split(np.zeros(len(labels_train)), labels_train))
        np.save(basedir + "indices_train.npy", indices_train)
        np.save(basedir + "indices_valid.npy", indices_valid)

if __name__ == "__main__":
    main()
