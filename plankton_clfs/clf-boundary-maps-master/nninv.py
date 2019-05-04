from keras.layers import Dense
from keras.initializers import Constant
from keras.models import Sequential
from keras.models import load_model
# from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
import numpy as np
import joblib


class NNInv():
    def __init__(self, loss='mean_squared_error', epochs=300):
        self.loss = loss
        self.epochs = epochs
        self.stop = EarlyStopping(verbose=1, min_delta=0.00001, mode='min',
                                  # patience=20, restore_best_weights=True)
                                  patience=10)
        self.X = None
        self.Xp = None

    def fit(self, X_nd, X_2d):
        self.m = Sequential()
        self.m.add(Dense(2048, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=l2(0.0002), input_shape=(X_2d.shape[1],)))
        self.m.add(Dense(2048, activation='relu', kernel_initializer='he_uniform', bias_initializer=Constant(0.01)))
        self.m.add(Dense(2048, activation='relu', kernel_initializer='he_uniform', bias_initializer=Constant(0.01)))
        self.m.add(Dense(2048, activation='relu', kernel_initializer='he_uniform', bias_initializer=Constant(0.01)))
        self.m.add(Dense(X_nd.shape[1], activation='sigmoid', kernel_initializer='he_uniform'))
        self.m.compile(loss=self.loss, optimizer='adam')

        hist = self.m.fit(X_2d, X_nd, batch_size=32, epochs=self.epochs, verbose=1, validation_split=0.05, callbacks=[self.stop])

        self.X = np.copy(X_nd)
        self.Xp = np.copy(X_2d)

        return hist

    def transform(self, X_2d, normalize=False):
        # TODO: normalize?
        return self.m.predict(X_2d)

    def save(self, path, clf_path):
        data = {}
        data['clf'] = clf_path
        data['X'] = self.X
        data['Xp'] = self.Xp
        self.m.save(clf_path)

        joblib.dump(data, path)

    def load(self, path):
        data = joblib.load(path)
        self.X = np.copy(data['X'])
        self.Xp = np.copy(data['Xp'])
        self.m = load_model(data['clf'])

