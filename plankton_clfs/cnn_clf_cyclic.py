import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop, SGD
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
import gzip
from skimage.transform import resize

from keras.layers import Lambda
from keras import backend as K
import tensorflow
from tensorflow import image as tfimage

#defining some util functions to do the cyclic layers
def array_tf_0(arr):
    return arr

def array_tf_90(arr):
    return tfimage.rot90(arr,k=1)

def array_tf_180(arr):
    return tfimage.rot90(arr,k=2)

def array_tf_270(arr):
    return tfimage.rot90(arr,k=3)

def CyclicSlice(x):
    """
    This layer stacks rotations of 0, 90, 180, and 270 degrees of the input
    along the batch dimension.

    If the input has shape (batch_size, num_channels, r, c),
    then the output will have shape (4 * batch_size, r, c, num_channels).

    Note that the stacking happens on axis 0, so a reshape to
    (4, batch_size, r, c, num_channels) will separate the slice axis.
    """
    return K.concatenate([array_tf_0(x), array_tf_90(x), array_tf_180(x), array_tf_270(x)], axis=0)    

def CyclicRoll(x):
    """
    This layer turns (n_views * batch_size, num_features) into
    (n_views * batch_size, n_views * num_features) by rolling
    and concatenating feature maps.
    """
    s = x.shape
    print(s)
   
    input_unfolded = K.reshape(x, (4, s[0] // 4, s[1]))
    permuted_inputs = []
    for p in range(4):
        input_permuted = K.reshape((tensorflow.manip.roll(input_unfolded, shift=-1-p, axix=0)),(s))
        permuted_inputs.append(input_permuted)
    return K.concatenate(permuted_inputs, axis=1)
        
def CyclicConvRoll(x):
    """
    This layer turns (n_views * batch_size, num_channels, r, c) into
    (n_views * batch_size, r, c, n-views * num_channels) by rolling
    and concatenating feature maps.

    It also applies the correct inverse transforms to the r and c
    dimensions to align the feature maps.
    """
    inv_tf_funcs = [array_tf_0, array_tf_270, array_tf_180, array_tf_90]
    k = 0
    s = x.shape
    print(s)

    permuted_inputs = []
    input_unfolded = K.reshape(x, (4, s[0]// 4, s[1], s[2], s[3]))
    for inv_tf in inv_tf_funcs:
        input_permuted = inv_tf(K.reshape((tensorflow.manip.roll(input_unfolded, shift=-1-k, axis=0)),(s)))
        k = k+1
        permuted_inputs.append(input_permuted)
    return K.concatenate(permuted_inputs, axis=3)
        
def CyclicPool(x):
    """
    Utility layer that unfolds the viewpoints dimension and pools over it.

    Note that this only makes sense for dense representations, not for
    feature maps (because no inverse transforms are applied to align them).
    """
    unfolded_input = K.reshape(x, (4, x.shape[0] // 4, x.shape[1]))
    return K.mean(unfolded_input, axis=0)
   
#Preprocess
def PreprocessImgs(imgs, target_size):
    new_imgs = np.zeros((len(imgs), target_size[0], target_size[1]))

    for i in range(len(imgs)):
        # FIXME anti_aliasing?
        new_imgs[i] = resize(imgs[i], target_size, mode='wrap', anti_aliasing=True).astype('float32')
    return new_imgs

#Load data
def LoadTrainData(target_shape):
    print("Loading data...")
    train_path = "./ndsb_dataset/images_train.npy.gz"
    labels_path = "./ndsb_dataset/labels_train.npy.gz"
    with gzip.open(labels_path, "rb") as f:
        labels = np.load(f)

    train_idx = np.load("./ndsb_dataset/indices_train.npy")
    valid_idx = np.load("./ndsb_dataset/indices_valid.npy")


    with gzip.open(train_path, "rb") as f:
        imgs = np.load(f)
    imgs = PreprocessImgs(imgs, (target_shape[0], target_shape[1]))
    new_shape = (len(imgs), target_shape[0], target_shape[1], target_shape[2])
    imgs = np.reshape(imgs, new_shape)

    X_train, y_train = imgs[train_idx], labels[train_idx]
    X_valid, y_valid = imgs[valid_idx], labels[valid_idx]

    return X_train, y_train, X_valid, y_valid

#Model configurations
def LoadModel(in_shape, num_classes, batch_size):
    # FIXME: 
    # - In the original network, the bias for convolutional layers is set to 1.0
    print("Loading model...")
    model = Sequential()

    a = 0.3
    # This looks like they call l0 in the code
    model.add(Lambda(function=CyclicSlice, input_shape=in_shape, batch_size=batch_size))
    	 
    # This looks like they call l1 in the code
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(16, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Lambda(function=CyclicConvRoll))

    # This looks like what they call l2 in the code
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Lambda(function=CyclicConvRoll))

    # This looks like what they cal l3 in the code
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Lambda(function=CyclicConvRoll))

    # This looks like what they call l4 in the code
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Lambda(function=CyclicConvRoll))
    model.add(Flatten())
    

    # This looks like what they call l5
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=a))
    #model.add(cyclicLayers.CyclicRollLayer())
    model.add(Lambda(function=CyclicRoll, input_shape=(128, 256)))

    # This looks like what they call l6
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=a))
    #model.add(cyclicLayers.CyclicPoolLayer())
    model.add(Lambda(function=CyclicPool, output_shape=output_shape_CyclicPool))
    
    # This looks like what they call l7
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

batch_size = 32
num_classes= 121
epochs = 10 

img_shape = (128, 128, 1)

X_train, y_train, X_valid, y_valid = LoadTrainData(img_shape)

#FIXME: zero mean unit variance
#X_train = X_train.astype('float32')
#X_valid = X_valid.astype('float32')
#X_train /= 255.0
#X_valid /= 255.0

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

model = LoadModel(img_shape, num_classes, batch_size)
model.summary()

#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#opt = SGD(lr=0.003, momentum=0.9)
opt = keras.optimizers.Adam()


model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

# FIXME: zero mean unit variance can be implemented by setting 1st and 3rd args
# below to True

datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=[1/1.6, 1.6], # range for random zoom
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True) # randomly flip images

datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(X_valid, y_valid),
                    workers=4,
                    verbose=1)

#model_path = os.path.join(save_dir, model_name)
#model.save(model_path)
#print('Saved trained model at %s ' % model_path)


# Score trained model.
scores = model.evaluate(X_valid, y_valid, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
