import glob
import numpy as np
from skimage.transform import resize
from skimage.io import imread
from math import floor
import os
import sys
import skimage.io
#from sklearn.model_selection import StratifiedShuffleSplit
import pickle
import cv2


def TENG(img):
    """Implements the Tenengrad (TENG) focus measure operator.
    Based on the gradient of the image.
    :param img: the image the measure is applied to
    :type img: numpy.ndarray
    :returns: numpy.float32 -- the degree of focus
    """
    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    return np.mean(gaussianX * gaussianX +
                      gaussianY * gaussianY)

def load(paths):
    """
    Load all images into memory for faster processing
    """
    images = np.empty(len(paths), dtype='object')
    for k, path in enumerate(paths):
        img = skimage.io.imread(path, as_grey=True)
        images[k] = img

    return images

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
    i=0
    """
    for i in range(len(class_names)):
        class_names[i] = int(class_names[i])
    """
    class_names.sort()
    num_classes = len(class_names)

    paths_train = glob.glob(basedir + "train/*/*")
    paths_train.sort()

    images = load(paths_train)

    labels = np.zeros(len(paths_train), dtype='int32')
    for k, path in enumerate(paths_train):
        class_name = os.path.basename(os.path.dirname(path))
        labels[k] = class_names.index(class_name)

    labels_train = np.empty(len(labels), dtype='int32')
    images_train = np.empty(len(paths), dtype='object')

    for i in range(images):
    	focus_measure = np.log(TENG(images[i]))
    	if focus_measure > -3.3:
    		images_train.append(images[i])
    		labels_train.append(labels[i])


    print("Saving train labels")

    np.save("./laps_focus_filtered/labels_train.npy", labels_train)
    #np.save(basedir + "labels_train.npy", labels_train)
    print("Gzipping train labels")
    os.system("gzip " + "./laps_focus_filtered/labels_train.npy")
    #os.system("gzip " + basedir + "labels_train.npy")

    """    print("Loading train images")
    images_train = load(paths_train)
    print("Saving train images")
    np.save("./laps_focus_filtered/images_train.npy", images_train)
    #np.save(basedir +  "images_train.npy", images_train)
    del images_train
    print("Gzipping train images")"""
    
    os.system("gzip " + "./laps_focus_filtered/images_train.npy")
    #os.system("gzip " + basedir + "images_train.npy")
    """
    print("Loading test images")
    images_test = load(paths_test)
    np.save(basedir + "images_test.npy", images_test)
    del images_test
    print("Gzipping test images")
    os.system("gzip " + basedir + "images_test.npy")
    """
    print("Done")

    print(labels_train)


    if create_validation == True:

        split = StratifiedShuffleSplit(n_splits=1,test_size=0.1)
        indices_train, indices_valid = next(split.split(np.zeros(len(labels_train)), labels_train))
        np.save("./laps_focus_filtered/indices_train.npy", indices_train)
        np.save("./laps_focus_filtered/indices_valid.npy", indices_valid)

        #np.save(basedir + "indices_train.npy", indices_train)
        #np.save(basedir + "indices_valid.npy", indices_valid)

if __name__ == "__main__":
    main()
