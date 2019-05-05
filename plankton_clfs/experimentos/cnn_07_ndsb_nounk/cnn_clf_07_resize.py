import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from math import floor
from skimage.transform import resize
import gzip

import os

def PreprocessImgs(imgs, target_size):
    new_imgs = np.ones((len(imgs), target_size[0], target_size[1]))

    for i in range(len(imgs)):
        current = imgs[i]
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
                    new_imgs[i,j,(k + minorside_pad)] = current[j,k]

        if majorside_idx == 1:
            current = resize(current, (minorside_new, target_size[1]), mode='constant', anti_aliasing=True)
            for j in range(current.shape[0]):
                for k in range(current.shape[1]):
                    new_imgs[i,(j + minorside_pad),k] = current [j,k]

        new_imgs[i] = new_imgs[i].astype('float32')
    return new_imgs

def LoadTrainData(target_shape):
    train_path = "./ndsb_dataset_nounk/images_train.npy.gz"
    labels_path = "./ndsb_dataset_nounk/labels_train.npy.gz"
    with gzip.open(labels_path, "rb") as f:
        labels = np.load(f)

    train_idx = np.load("./ndsb_dataset_nounk/indices_train.npy")
    valid_idx = np.load("./ndsb_dataset_nounk/indices_valid.npy")


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
    
    # This looks like they call l1 in the code
    model.add(Conv2D(32, (3,3), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(16, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # This looks like what they call l2 in the code
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(32, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # This looks like what they cal l3 in the code
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(64, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

    # This looks like what they call l4 in the code
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(256, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(Conv2D(128, (3,3), padding='same'))
    model.add(LeakyReLU(alpha=a))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
    model.add(Flatten())
    model.add(Dropout(0.5))

    # This looks like what they call l5
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=a))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=a))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    return model


batch_size = 32
num_classes= 118
epochs = 500

img_shape = (95, 95, 1)

X_train, y_train, X_valid, Y_valid = LoadTrainData(img_shape)

#FIXME: zero mean unit variance
#X_train = X_train.astype('float32')
#X_valid = X_valid.astype('float32')
#X_train /= 255.0
#X_valid /= 255.0

y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(Y_valid, num_classes)

model = LoadModel(img_shape, num_classes)
model.summary()

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# FIXME: zero mean unit variance can be implemented by setting 1st and 3rd args
# below to True
datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True) # randomly flip images

datagen.fit(X_train)

model_path = "./ndsb_dataset_nounk/best_model_07.hdf5" 

checkpoint = ModelCheckpoint(model_path,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')

history = model.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(X_valid, y_valid),
                    workers=4,
                    callbacks=[checkpoint],
                    verbose=1)

#model_path = os.path.join(save_dir, model_name)
#model.save(model_path)
#print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(X_valid, y_valid, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

predictions = model.predict(X_valid, batch_size=batch_size)

predicted_labels = np.argmax(predictions, axis=1)
print('Predicted labels:', predicted_labels)

print('Real labels:', Y_valid)
np.save("./ndsb_dataset_nounk/complete_predictions_07.npy", predictions)
np.save("./ndsb_dataset_nounk/predicted_labels_07.npy", predicted_labels)
np.save("./ndsb_dataset_nounk/real_labels_07.npy", Y_valid)
print("Predictions saved")

#Saving trained model.
model_json = model.to_json()
with open("./ndsb_dataset_nounk/model_07.json","w") as json_file:
    json_file.write(model_json)
print("Model saved.")




