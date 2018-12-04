import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from utils import Data, Image

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


# Loading data
train_path = "./ndsb_dataset_nounk/images_train.npy.gz"
labels_path = "./ndsb_dataset_nounk/labels_train.npy.gz"
train_idx = np.load("./ndsb_dataset_nounk/indices_train.npy")
valid_idx = np.load("./ndsb_dataset_nounk/indices_valid.npy")

target_shape = (95, 95, 1)
data = Data()
img_objects = data.load(train_path, labels_path, target_shape)
data.set_train_val_indices(train_idx, valid_idx)
X_train, y_train = data.create_train_set()
X_valid, y_valid = data.create_valid_set()
train_objects = img_objects[data.get_train_indices()]
valid_objects = img_objects[data.get_valid_indices()]


num_classes= 198
y_train = keras.utils.to_categorical(y_train, num_classes)
y_valid = keras.utils.to_categorical(Y_valid, num_classes)

# Defining model configurations
batch_size = 32
epochs = 420
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

model = LoadModel(img_shape, num_classes)
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

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

# Training the model with training data augmented
history = model.fit_generator(datagen.flow(X_train, y_train,
                                 batch_size=batch_size),
                    epochs=epochs,
                    validation_data=(X_valid, y_valid),
                    workers=4,
                    verbose=1)


# Scoring trained model with validation data
scores = model.evaluate(X_valid, y_valid, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

predictions = model.predict(X_valid, batch_size=batch_size)
predicted_labels = np.argmax(predictions, axis=1)
print('Predicted labels:', predicted_labels)

for i in range(len(valid_objects)):
    valid_objects[i].set_pred_label(predicted_labels[i])


print('Real labels:', Y_valid)
np.save("./ndsb_dataset_nounk/complete_predictions_nounk.npy", predictions)
np.save("./ndsb_dataset_nounk/predicted_labels_nounk.npy", predicted_labels)
np.save("./ndsb_dataset_nounk/real_labels_nounk.npy", Y_valid)
print("Predictions saved")

# Saving trained model.
model_json = model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Model and weights saved.")
