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

import cyclicLayers
from keras.layers import Lambda
from keras import backend as K

def EqualInput(x):
    return x

def array_tf_0(arr):
    return arr

def array_tf_90(arr):
    arr = np.asarray(arr)
    axes_order = [range(arr.ndim - 2)] + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)

def array_tf_180(arr):
    kvar = K.variable(value=arr)
    slices = [slice(None) for _ in range(K.ndim(kvar) - 2)] + [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)]

def array_tf_270(arr):
    kvar = K.variable(value=arr)
    axes_order = range(K.ndim(kvar) - 2) + [K.ndim(kvar) - 1, K.ndim(kvar) - 2]
    slices = [slice(None) for _ in range(K.ndim(kvar) - 2)] + [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)

def CyclicSlice(x):
    return K.concatenate([
                array_tf_0(x),
                array_tf_90(x),
                array_tf_180(x),
                array_tf_270(x),
            ], axis=0)

def output_shape_CyclicSlice(x):
    return (4*input_shape[0],) + input_shape[1:]



def PreprocessImgs(imgs, target_size):
    new_imgs = np.zeros((len(imgs), target_size[0], target_size[1]))

    for i in range(len(imgs)):
        # FIXME anti_aliasing?
        # check what this resize is doing to the images
        new_imgs[i] = resize(imgs[i], target_size, mode='wrap').astype('float32')
    return new_imgs

def LoadTrainData(target_shape):
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

def LoadModel(in_shape, num_classes):
    # FIXME: 
    # - In the original network, the bias for convolutional layers is set to 1.0
   
    model = Sequential()

    a = 0.3
    #l0
    model.add(Lambda(function=EqualInput, output_shape=in_shape, input_shape=in_shape))
    model.add(Lambda(function=CyclicSlice, output_shape=output_shape_CyclicSlice))
    #model.add(cyclicLayers.CyclicSliceLayer(in_shape=in_shape,input_shape=in_shape))
	 
    # This looks like they call l1 in the code
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(16, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(cyclicLayers.CyclicConvRollLayer())

    # This looks like what they call l2 in the code
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(cyclicLayers.CyclicConvRollLayer())

    # This looks like what they cal l3 in the code
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(cyclicLayers.CyclicConvRollLayer())

    # This looks like what they call l4 in the code
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(cyclicLayers.CyclicConvRollLayer())
    model.add(Flatten())
    

    # This looks like what they call l5
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=a))
    model.add(cyclicLayers.CyclicRollLayer())
    
    # This looks like what they call l6
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=a))
    model.add(cyclicLayers.CyclicPoolLayer())
    
    # This looks like what they call l7
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

batch_size = 32
num_classes= 121
epochs = 10 

img_shape = (95, 95, 1)

X_train, y_train, X_valid, y_valid = LoadTrainData(img_shape)

#FIXME: zero mean unit variance
#X_train = X_train.astype('float32')
#X_valid = X_valid.astype('float32')
#X_train /= 255.0
#X_valid /= 255.0

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(y_valid, num_classes)

model = LoadModel(img_shape, num_classes)
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
