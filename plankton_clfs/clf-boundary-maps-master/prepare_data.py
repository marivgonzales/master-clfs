# prepare_data.py: Creates a single file containing the shuffled dataset,
# a subset of projected points and a trained classifier.

import numpy as np

import data
import lamp

# from sklearn import manifold
from sklearn import linear_model, svm, neighbors
from sklearn import preprocessing

import pickle
import json
import os
import joblib
import sys

from time import time

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

from MulticoreTSNE import MulticoreTSNE as TSNE
from umap import UMAP

from lamp import ILAMP
from lamp import RBFInv
from nninv import NNInv


np.random.seed(1)


def TrainTestSplit(X_orig, y_orig, split_sz):
    new_idx = np.random.permutation(X_orig.shape[0])
    X, y = X_orig[new_idx], y_orig[new_idx]

    train_size = int(X.shape[0]*split_sz)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    return X_train, y_train, X_test, y_test


# clfs: array of
def SaveData(name, path, train_X, train_y, test_X, test_y, proj, clfs):
    train_X_path = path + name + "_X_train.npy"
    train_y_path = path + name + "_y_train.npy"
    test_X_path =  path + name + "_X_test.npy"
    test_y_path =  path + name + "_y_test.npy"
    proj_path =    path + name + "_proj.npy"

    np.save(train_X_path, train_X)
    np.save(train_y_path, train_y)
    np.save(test_X_path,  test_X)
    np.save(test_y_path,  test_y)
    np.save(proj_path,    proj)

    data = {'X_train' : train_X_path,
            'y_train' : train_y_path,
            'X_test'  : test_X_path,
            'y_test'  : test_y_path,
            'proj'    : proj_path,
            'clfs'    : clfs}
     
    with open(path + name + ".json", 'w') as outfile:
        json.dump(data, outfile)

def Toy():
    # Load TOY dataset
    start = time()
    print("Reading TOY dataset...")
    toy_X, toy_y = data.LoadZeroOneData(); 
    print("\tFinished reading TOY dataset...", time() - start)
    
    start = time()
    print("TrainTestSplit for TOY...")
    toy_X_train, toy_y_train, toy_X_test, toy_y_test = TrainTestSplit(toy_X, toy_y, 0.7)
    print("\tFinished TrainTestSplit...", time() - start)

    # Uses LAMP to project the entire dataset
    start = time()
    print("LAMP projecting TOY dataset...")
    toy_proj = lamp.lamp2d(toy_X_train)
    print("\tFinished projecting...", time() - start)

    start = time()
    print("Training classifier...")
    toy_lr = linear_model.LogisticRegression()
    toy_lr.fit(toy_X_train, toy_y_train)
    print("\tAccuracy on test data: ", toy_lr.score(toy_X_test, toy_y_test))
    print("\tFinished training classifier...", time() - start)

    start = time()
    print("Saving data for TOY...")
    clfs = ["data/toy/toy_logistic_regression.pkl"]
    with open('data/toy/toy_logistic_regression.pkl', 'wb') as f:
        pickle.dump(toy_lr, f)
    SaveData("toy", "data/toy/", toy_X_train, toy_y_train, toy_X_test,
             toy_y_test, toy_proj, clfs)

    print("\tFinished saving data...", time() - start)


def Wine():
    start = time()
    print("Reading WINE dataset...")
    wine_X, wine_y = data.LoadWineData()
    print("\tFinished reading WINE dataset...", time() - start)

    start = time()
    print("TrainTestSplit for WINE...")
    wine_X_train, wine_y_train, wine_X_test, wine_y_test = TrainTestSplit(wine_X, wine_y, 0.7)
    print("\tFinished TrainTestSplit...", time() - start)

    # Uses LAMP to project the entire dataset
    start = time()
    print("LAMP projecting WINE dataset...")
    wine_proj = lamp.lamp2d(wine_X_train)
    print("\tFinished projecting...", time() - start)

    start = time()
    print("Training classifier...")
    wine_lr = linear_model.LogisticRegression()
    wine_lr.fit(wine_X_train, wine_y_train)
    print("\tAccuracy on test data: ", wine_lr.score(wine_X_test, wine_y_test))
    print("\tFinished training classifier...", time() - start)

    start = time()
    print("Saving data for WINE...")
    clfs = ["data/wine/wine_logistic_regression.pkl"]
    with open('data/wine/wine_logistic_regression.pkl', 'wb') as f:
        pickle.dump(wine_lr, f)

    SaveData("wine", "data/wine/", wine_X_train, wine_y_train, wine_X_test,
             wine_y_test, wine_proj, clfs)
    print("\tFinished saving data...", time() - start)


def Segmentation():
    start = time()
    print("Reading SEGMENTATION dataset...")
    seg_X, seg_y = data.LoadSegmentationData()
    print("\tFinished reading SEGMENTATION dataset...", time() - start)

    start = time()
    print("TrainTestSplit for SEGMENTATION...")
    seg_X_train, seg_y_train, seg_X_test, seg_y_test = TrainTestSplit(seg_X, seg_y, 0.7)
    print("\tFinished TrainTestSplit...", time() - start)

    # Uses LAMP to project the entire dataset
    start = time()
    print("LAMP projecting SEGMENTATION dataset...")
    seg_proj = lamp.lamp2d(seg_X_train, 150, 8.0)
    print("\tFinished projecting...", time() - start)

    start = time()
    print("Training classifier LogisticRegression...")
    seg_lr = linear_model.LogisticRegression()
    seg_lr.fit(seg_X_train, seg_y_train)
    print("\tAccuracy on test data: ", seg_lr.score(seg_X_test, seg_y_test))
    print("\tFinished training classifier...", time() - start)

    start = time()
    print("Training classifier SVM...")
    seg_svm = svm.SVC()
    seg_svm.fit(seg_X_train, seg_y_train)
    print("\tAccuracy on test data: ", seg_svm.score(seg_X_test, seg_y_test))
    print("\tFinished training classifier...", time() - start)

    start = time()
    print("Training classifier KNN...")
    seg_knn5 = neighbors.KNeighborsClassifier(n_neighbors=5)
    seg_knn5.fit(seg_X_train, seg_y_train)
    print("\tAccuracy on test data: ", seg_knn5.score(seg_X_test, seg_y_test))
    print("\tFinished training classifier...", time() - start)

    start = time()
    print("Saving data for SEGMENTATION...")
    clfs = ["data/segmentation/seg_logistic_regression.pkl",
            "data/segmentation/seg_svm.pkl",
            "data/segmentation/seg_knn5.pkl"]

    with open('data/segmentation/seg_logistic_regression.pkl', 'wb') as f:
        pickle.dump(seg_lr, f)
    with open('data/segmentation/seg_svm.pkl', 'wb') as f:
        pickle.dump(seg_svm, f)
    with open('data/segmentation/seg_knn5.pkl', 'wb') as f:
        pickle.dump(seg_knn5, f)

    SaveData("seg", "data/segmentation/", seg_X_train, seg_y_train, seg_X_test,
             seg_y_test, seg_proj, clfs)
    print("\tFinished saving data...", time() - start)


def CNNModel(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(lr=0.1),
                  metrics=['accuracy'])
    return model


def CNNModel2(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.001),
                  metrics=['accuracy'])
    return model


def MNIST(base_dir):
    print("Reading MNIST dataset...")
    start = time()

    train_X_path = base_dir + "X_train.npy"
    train_y_path = base_dir + "y_train.npy"
    test_X_path = base_dir + "X_test.npy"
    test_y_path = base_dir + "y_test.npy"
    if not os.path.exists(train_X_path):
        X_train, y_train = data.LoadMNISTData('train', base_dir + 'orig/')
        new_idx = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[new_idx], y_train[new_idx]
        X_train_base = np.copy(X_train)
        y_train_base = np.copy(y_train)

        X_test, y_test = data.LoadMNISTData('test', base_dir + 'orig/')
        X_test_base = np.copy(X_test)
        y_test_base = np.copy(y_test)

        np.save(train_X_path, X_train_base)
        np.save(train_y_path, y_train_base)
        np.save(test_X_path,  X_test_base)
        np.save(test_y_path,  y_test_base)
    else:
        X_train = np.load(train_X_path)
        y_train = np.load(train_y_path)
        X_test = np.load(test_X_path)
        y_test = np.load(test_y_path)

    print("\tFinished reading dataset...", time() - start)

    projection_size = 60000
    X_proj = np.copy(X_train[:projection_size])
    new_shape = (X_proj.shape[0], X_proj.shape[1]*X_proj.shape[2])
    X_proj = np.reshape(X_proj, new_shape)

    proj_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # Uses LAMP to project projection_size points from the dataset
    # print("LAMP projecting MNIST dataset...")
    # start = time()
    # proj_path1 = base_dir + "lamp_proj.npy"
    # if not os.path.exists(proj_path1):
    #     proj_lamp = lamp.lamp2d(X_proj, 150, 10.0)
    #     proj_lamp = proj_scaler.fit_transform(proj_lamp)
    #     np.save(proj_path1, proj_lamp)
    # else:
    #     proj_lamp = np.load(proj_path1)
    # print("\tFinished projecting...", time() - start)

    # Uses t-SNE to project projection_size points from the dataset
    print("tSNE projection")
    start = time()
    proj_path2 = base_dir + "tsne_proj.npy"
    if not os.path.exists(proj_path2):
        # tsne = manifold.TSNE(n_components=2, perplexity=20.0)
        tsne = TSNE(n_components=2, perplexity=20.0, n_jobs=8)
        proj_tsne = tsne.fit_transform(X_proj)
        proj_tsne = proj_scaler.fit_transform(proj_tsne)

        train_y_proj_path = base_dir + "y_proj_true.npy"
        np.save(train_y_proj_path, y_train_base[:projection_size])
        np.save(proj_path2, proj_tsne)
    else:
        proj_tsne = np.load(proj_path2)
    print("\tProjection finished: ", time() - start)

    subset_size = 15000
    print("ILAMP")
    start = time()
    ilamp_path = base_dir + "ilamp.joblib"
    if not os.path.exists(ilamp_path):
        k_ilamp = 20
        ilamp = ILAMP(n_neighbors=k_ilamp)
        ilamp.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        ilamp.save(ilamp_path)
    # else:
    #     ilamp = joblib.load(ilamp_path)
    print("\ttime ", time() - start)

    print("RBFInv - Control Points")
    start = time()
    irbfcp_path = base_dir + "irbf_cp.joblib"
    if not os.path.exists(irbfcp_path):
        EPS = 50000
        irbf_cp = RBFInv(num_ctrl=400, mode='rols', kernel='gaussian',
                         eps=EPS, normalize_c=True, normalize_d=True)
        irbf_cp.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        irbf_cp.save(irbfcp_path)
        # joblib.dump(irbf_cp, irbfcp_path)
    # else:
    #     irbf_cp = joblib.load(irbfcp_path)
    print("\ttime ", time() - start)

    print("RBFInv - Neighbors")
    start = time()
    irbfn_path = base_dir + "irbf_neighbors.joblib"
    if not os.path.exists(irbfn_path):
        EPS = 5000000
        irbf_neighbors = RBFInv(num_ctrl=20, mode='neighbors',
                                kernel='gaussian', eps=EPS, normalize_c=True,
                                normalize_d=True)
        irbf_neighbors.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        irbf_neighbors.save(irbfn_path)
        # joblib.dump(irbf_neighbors, irbfn_path)
    # else:
    #     irbf_neighbors = joblib.load(irbfn_path)
    print("\ttime ", time() - start)

    print("RBFInv - Cluster")
    start = time()
    irbfc_path = base_dir + "irbf_cluster.joblib"
    if not os.path.exists(irbfc_path):
        EPS = 50000
        irbf_cluster = RBFInv(num_ctrl=50, mode='cluster',
                              kernel='gaussian', eps=EPS, normalize_c=True,
                              normalize_d=True)
        irbf_cluster.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        irbf_cluster.save(irbfc_path)
        # joblib.dump(irbf_cluster, irbfc_path)
    # else:
    #     irbf_cluster = joblib.load(irbfc_path)
    print("\ttime ", time() - start)

    print("NNInv")
    start = time()
    nninv_path = base_dir + "nninv.joblib"
    if not os.path.exists(nninv_path):
        nninv = NNInv()
        nninv.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        nninv.save(nninv_path, base_dir + 'nninv_keras.hdf5')
        # joblib.dump(nninv, nninv_path)
    # else:
    #     nninv = NNInv()
    #     nninv.load(nninv_path)
    print("\ttime ", time() - start)

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    X_train = X_train.reshape((X_train.shape[0],) + input_shape)
    X_test = X_test.reshape((X_test.shape[0],) + input_shape)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    X_proj = X_proj.reshape((X_proj.shape[0],) + input_shape)

    clf1_path = base_dir + "mnist_cnn1.hdf5"
    pred_path1 = base_dir + "y_pred_clf1.npy"
    # pred_path2 = base_dir + "y_pred_clf2.npy"
    # pred_path3 = base_dir + "y_pred_clf3.npy"
    # pred_path4 = base_dir + "y_pred_clf4.npy"
    # pred_path5 = base_dir + "y_pred_clf5.npy"
    if not os.path.exists(clf1_path):
        print("Training classifier CNN...")
        start = time()
        clf1 = CNNModel(input_shape, 10)
        clf1.fit(X_train, y_train, batch_size=128, epochs=14, verbose=1,
                 validation_data=(X_test, y_test))
        print("\tAccuracy on test data: ", clf1.evaluate(X_test, y_test, verbose=0))
        print("\tFinished training classifier...", time() - start)
        clf1.save(base_dir + "mnist_cnn1.hdf5")
        y_proj_pred1 = np.argmax(clf1.predict(X_proj), axis=1)
        np.save(pred_path1, y_proj_pred1)

        # print("Training classifier CNN 2...")
        # start = time()
        # clf2 = CNNModel2(input_shape, 10)

        # print("\tEpoch 1:")
        # clf2.fit(X_train, y_train, batch_size=128, epochs=1, verbose=1,
        #          validation_data=(X_test, y_test))
        # clf2.save(base_dir + "mnist_cnn2_1e.hdf5")
        # y_proj_pred2 = np.argmax(clf2.predict(X_proj), axis=1)
        # print("\tAccuracy on test data: ", clf2.evaluate(X_test, y_test, verbose=0))

        # print("\tEpoch 5:")
        # clf2.fit(X_train, y_train, batch_size=128, epochs=4, verbose=1,
        #          validation_data=(X_test, y_test))
        # clf2.save(base_dir + "mnist_cnn2_5e.hdf5")
        # y_proj_pred3 = np.argmax(clf2.predict(X_proj), axis=1)
        # print("\tAccuracy on test data: ", clf2.evaluate(X_test, y_test, verbose=0))

        # print("\tEpoch 10:")
        # clf2.fit(X_train, y_train, batch_size=128, epochs=5, verbose=1,
        #          validation_data=(X_test, y_test))
        # clf2.save(base_dir + "mnist_cnn2_10e.hdf5")
        # y_proj_pred4 = np.argmax(clf2.predict(X_proj), axis=1)
        # print("\tAccuracy on test data: ", clf2.evaluate(X_test, y_test, verbose=0))

        # print("\tEpoch 50:")
        # clf2.fit(X_train, y_train, batch_size=128, epochs=40, verbose=1,
        #          validation_data=(X_test, y_test))
        # clf2.save(base_dir + "mnist_cnn2_50e.hdf5")
        # y_proj_pred5 = np.argmax(clf2.predict(X_proj), axis=1)
        # print("\tAccuracy on test data: ", clf2.evaluate(X_test, y_test, verbose=0))

        # np.save(pred_path2, y_proj_pred2)
        # np.save(pred_path3, y_proj_pred3)
        # np.save(pred_path4, y_proj_pred4)
        # np.save(pred_path5, y_proj_pred5)

        print("\tFinished training classifier...", time() - start)
    # else:
    #     # TODO: load clf, predict and save y

    print("Saving data for MNIST...")
    clfs = [base_dir + "mnist_cnn1.hdf5"]
            # base_dir + "mnist_cnn2_1e.hdf5",
            # base_dir + "mnist_cnn2_5e.hdf5",
            # base_dir + "mnist_cnn2_10e.hdf5",
            # base_dir + "mnist_cnn2_50e.hdf5"]

    inv_projs = [ilamp_path, irbfcp_path, irbfn_path, irbfc_path, nninv_path]
    # preds = [pred_path1, pred_path2, pred_path3, pred_path4, pred_path5]
    preds = [pred_path1]

    data_json = {'X_train'  : train_X_path,
                 'y_train'  : train_y_path,
                 'X_test'   : test_X_path,
                 'y_test'   : test_y_path,
                 # 'proj1'    : proj_path1,
                 # 'proj2'    : proj_path2,
                 'projs'    : [proj_path2],
                 'inv_projs': inv_projs,
                 'y_preds'  : preds, 
                 'y_true'   : train_y_proj_path,
                 'clfs'     : clfs}

    with open(base_dir + "mnist.json", 'w') as outfile:
        json.dump(data_json, outfile)

    print("\tFinished saving data...", time() - start)


def CNN_FM():
    model = Sequential()

    dim = 28
    nclasses = 10

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same',
                     activation='relu', input_shape=(dim, dim, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same',
                     activation='relu',))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(nclasses, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])

    return model


def FashionMNIST(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print("Reading Fashion MNIST dataset...")
    start = time()

    train_X_path = base_dir + "X_train.npy"
    train_y_path = base_dir + "y_train.npy"
    test_X_path = base_dir + "X_test.npy"
    test_y_path = base_dir + "y_test.npy"

    if not os.path.exists(train_X_path):
        print("\tSaving train data")
        X_train, y_train = data.LoadFashionMNIST('train', base_dir + 'orig')
        new_idx = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[new_idx], y_train[new_idx]
        X_train_base = np.copy(X_train)
        y_train_base = np.copy(y_train)

        X_test, y_test = data.LoadFashionMNIST('test', base_dir + 'orig')
        X_test_base = np.copy(X_test)
        y_test_base = np.copy(y_test)

        np.save(train_X_path, X_train_base)
        np.save(train_y_path, y_train_base)
        np.save(test_X_path, X_test_base)
        np.save(test_y_path, y_test_base)
    else:
        print("\tReading train data")
        X_train = np.load(train_X_path)
        y_train = np.load(train_y_path)
        X_test = np.load(test_X_path)
        y_test = np.load(test_y_path)
    print("\ttime ", time() - start)

    projection_size = 60000
    X_proj = np.copy(X_train[:projection_size])
    new_shape = (X_proj.shape[0], X_proj.shape[1]*X_proj.shape[2])
    X_proj = np.reshape(X_proj, new_shape)

    tsne_proj_path = base_dir + "tsne_proj.npy"
    if not os.path.exists(tsne_proj_path):
        # Uses t-SNE to project projection_size points from the dataset
        print("\tt-SNE projecting Fashion MNIST dataset...")
        start = time()
        # tsne = manifold.TSNE(n_components=2, perplexity=35.0)
        # tsne = TSNE(n_components=2, perplexity=35.0, n_jobs=4)
        tsne = TSNE(n_components=2, random_state=420, perplexity=10.0,
                    n_iter=1000, n_iter_without_progress=300, n_jobs=4)
        proj_tsne = tsne.fit_transform(X_proj)
        print("\ttime ", time() - start)
        proj_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        proj_tsne = proj_scaler.fit_transform(proj_tsne)
        np.save(tsne_proj_path, proj_tsne)
    else:
        print("\tReading projection")
        proj_tsne = np.load(tsne_proj_path)

    umap_proj_path = base_dir + "umap_proj.npy"
    if not os.path.exists(umap_proj_path):
        print("\tUMAP projecting Fashion MNIST dataset...")
        start = time()
        proj_umap = UMAP(n_components=2, n_neighbors=10, min_dist=0.5,
                         random_state=420).fit_transform(X_proj)

        print("\ttime ", time() - start)
        proj_scaler2 = preprocessing.MinMaxScaler(feature_range=(0, 1))
        proj_umap = proj_scaler2.fit_transform(proj_umap)
        np.save(umap_proj_path, proj_umap)
    else:
        proj_umap = np.load(umap_proj_path)

    train_y_proj_path = base_dir + "y_proj_true.npy"
    if not os.path.exists(train_y_proj_path):
        np.save(train_y_proj_path, y_train_base[:projection_size])

    subset_size = 20000

    print("ILAMP - tSNE")
    start = time()
    ilamp_tsne_path = base_dir + "ilamp_tsne.joblib"
    if not os.path.exists(ilamp_tsne_path):
        k_ilamp = 20
        ilamp_tsne = ILAMP(n_neighbors=k_ilamp)
        ilamp_tsne.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        ilamp_tsne.save(ilamp_tsne_path)
    else:
        ilamp_tsne = ILAMP()
        ilamp_tsne.load(ilamp_tsne_path)
    print("\ttime ", time() - start)

    print("ILAMP - UMAP")
    start = time()
    ilamp_umap_path = base_dir + "ilamp_umap.joblib"
    if not os.path.exists(ilamp_umap_path):
        k_ilamp = 25
        ilamp_umap = ILAMP(n_neighbors=k_ilamp)
        ilamp_umap.fit(X_proj[:subset_size], proj_umap[:subset_size])
        ilamp_umap.save(ilamp_umap_path)
    # else:
    #     ilamp_umap = ILAMP()
    #     ilamp_umap.load(ilamp_umap_path)
    print("\ttime ", time() - start)

    print("RBFInv - Control Points tSNE")
    start = time()
    irbfcp_tsne_path = base_dir + "irbfcp_tsne.joblib"
    if not os.path.exists(irbfcp_tsne_path):
        EPS = 50000
        irbfcp_tsne = RBFInv(num_ctrl=200, mode='rols', kernel='gaussian',
                             eps=EPS, normalize_c=True, normalize_d=True)
        irbfcp_tsne.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        irbfcp_tsne.save(irbfcp_tsne_path)
    # else:
    #     irbf_cp = joblib.load((irbfcp_path)
    print("\ttime ", time() - start)

    print("RBFInv - Control Points UMAP")
    start = time()
    irbfcp_umap_path = base_dir + "irbfcp_umap.joblib"
    if not os.path.exists(irbfcp_umap_path):
        EPS = 50000
        irbfcp_umap = RBFInv(num_ctrl=200, mode='rols', kernel='gaussian',
                             eps=EPS, normalize_c=True, normalize_d=True)
        irbfcp_umap.fit(X_proj[:subset_size], proj_umap[:subset_size])
        irbfcp_umap.save(irbfcp_umap_path)
    # else:
    #     irbf_cp = joblib.load((irbfcp_path)
    print("\ttime ", time() - start)

    # irbfn_tsne_path = base_dir + "irbfn_tsne.joblib"
    # print("RBFInv - Neighbors tSNE")
    # start = time()
    # if not os.path.exists(irbfn_tsne_path):
    #     EPS = 5000000
    #     irbfn_tsne = RBFInv(num_ctrl=20, mode='neighbors', eps=EPS,
    #                         kernel='gaussian', normalize_c=True,
    #                         normalize_d=True)
    #     irbfn_tsne.fit(X_proj[:subset_size], proj_tsne[:subset_size])
    #     irbfn_tsne.save(irbfn_tsne_path)
    #     # joblib.dump(irbf_neighbors, irbfn_path)
    # # else:
    # #     irbf_neighbors = joblib.load(irbfn_path)
    # print("\ttime ", time() - start)

    print("RBFInv - Cluster tSNE")
    start = time()
    irbfc_tsne_path = base_dir + "irbfc_tsne.joblib"
    if not os.path.exists(irbfc_tsne_path):
        EPS = 50000
        irbfc_tsne = RBFInv(num_ctrl=50, mode='cluster', eps=EPS,
                            kernel='gaussian', normalize_c=True,
                            normalize_d=True)
        irbfc_tsne.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        irbfc_tsne.save(irbfc_tsne_path)
        # joblib.dump(irbf_cluster, irbfc_path)
    # else:
    #     irbf_cluster = joblib.load(irbfc_path)
    print("\ttime ", time() - start)

    print("RBFInv - Cluster UMAP")
    start = time()
    irbfc_umap_path = base_dir + "irbfc_umap.joblib"
    if not os.path.exists(irbfc_umap_path):
        EPS = 50000
        irbfc_umap = RBFInv(num_ctrl=50, mode='cluster', eps=EPS,
                            kernel='gaussian', normalize_c=True,
                            normalize_d=True)
        irbfc_umap.fit(X_proj[:subset_size], proj_umap[:subset_size])
        irbfc_umap.save(irbfc_umap_path)
        # joblib.dump(irbf_cluster, irbfc_path)
    # else:
    #     irbf_cluster = joblib.load(irbfc_path)
    print("\ttime ", time() - start)

    print("NNInv tSNE")
    start = time()
    nninv_tsne_path = base_dir + "nninv_tsne.joblib"
    if not os.path.exists(nninv_tsne_path):
        nninv_tsne = NNInv()
        nninv_tsne.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        nninv_tsne.save(nninv_tsne_path, base_dir + 'nninv_tsne_keras.hdf5')
    # else:
    #     nninv = NNInv()
    #     nninv.load(nninv_path)
    print("\ttime ", time() - start)

    print("NNInv UMAP")
    start = time()
    nninv_umap_path = base_dir + "nninv_umap.joblib"
    if not os.path.exists(nninv_umap_path):
        nninv_umap = NNInv()
        nninv_umap.fit(X_proj[:subset_size], proj_umap[:subset_size])
        nninv_umap.save(nninv_umap_path, base_dir + 'nninv_umap_keras.hdf5')
    # else:
    #     nninv = NNInv()
    #     nninv.load(nninv_path)
    print("\ttime ", time() - start)

    print("Training classifier CNN...")
    start = time()
    clf_path = base_dir + "fm_cnn.hdf5"
    pred_path = base_dir + "y_pred_clf.npy"
    if not os.path.exists(clf_path):
        input_shape = (X_train.shape[1], X_train.shape[2], 1)
        X_train = X_train.reshape((X_train.shape[0],) + input_shape)
        X_test = X_test.reshape((X_test.shape[0],) + input_shape)
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        X_proj = X_proj.reshape((X_proj.shape[0],) + input_shape)

        clf1 = CNN_FM()
        clf1.fit(X_train, y_train, batch_size=64, epochs=10, verbose=1,
                 validation_data=(X_test, y_test))
        accuracy = clf1.evaluate(X_test, y_test, verbose=0)
        print("\tAccuracy on test data: ", accuracy)
        clf1.save(clf_path)
        y_proj_pred = np.argmax(clf1.predict(X_proj), axis=1)
        np.save(pred_path, y_proj_pred)
    # else:
    #     clf1 = keras.models.load_model(clf_path)
    #     input_shape = (X_train.shape[1], X_train.shape[2], 1)
    #     X_train = X_train.reshape((X_train.shape[0],) + input_shape)
    #     X_test = X_test.reshape((X_test.shape[0],) + input_shape)
    #     y_train = keras.utils.to_categorical(y_train, 10)
    #     y_test = keras.utils.to_categorical(y_test, 10)
    #     X_proj = X_proj.reshape((X_proj.shape[0],) + input_shape)
    #     accuracy = clf1.evaluate(X_test, y_test, verbose=0)
    #     print("\tAccuracy on test data: ", accuracy)
    #     y_proj_pred = np.argmax(clf1.predict(X_proj), axis=1)
    print("\ttime ", time() - start)

    print("Saving data for Fashion MNIST...")
    start = time()

    clfs = [clf_path]
    inv_projs = [ilamp_tsne_path, irbfcp_tsne_path, irbfc_tsne_path,
                 nninv_tsne_path,
                 ilamp_umap_path, irbfcp_umap_path, irbfc_umap_path,
                 nninv_umap_path]
    preds = [pred_path]

    data_json = {'X_train'  : train_X_path,
                 'y_train'  : train_y_path,
                 'X_test'   : test_X_path,
                 'y_test'   : test_y_path,
                 'projs'    : [tsne_proj_path, umap_proj_path],
                 'inv_projs': inv_projs,
                 'y_preds'  : preds, 
                 'y_true'   : train_y_proj_path,
                 'clfs'     : clfs}

    with open(base_dir + "fm.json", 'w') as outfile:
        json.dump(data_json, outfile)
    print("\tFinished saving data...", time() - start)


def CNN_Cifar():
    shp = (32, 32, 3)
    weight_decay = 1e-4
    reg = l2(weight_decay)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg,
              input_shape=shp))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=reg))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=opt_rms,
                  metrics=['accuracy'])

    return model


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate


def Cifar10(base_dir='data/cifar10/'):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print("Reading Cifar-10 dataset...")
    start = time()

    train_X_path = base_dir + "X_train.npy"
    train_y_path = base_dir + "y_train.npy"
    test_X_path = base_dir + "X_test.npy"
    test_y_path = base_dir + "y_test.npy"
    if not os.path.exists(train_X_path):
        X_train, y_train, X_test, y_test = data.LoadCifar10(base_dir)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # TODO: normalize to [0, 1] - simpler to normalize inv_proj and to
        # adversarial attacks
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)

        new_idx = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[new_idx], y_train[new_idx]
        X_train_base = np.copy(X_train)
        y_train_base = np.copy(y_train)

        X_test_base = np.copy(X_test)
        y_test_base = np.copy(y_test)
        np.save(train_X_path, X_train_base)
        np.save(train_y_path, y_train_base)
        np.save(test_X_path,  X_test_base)
        np.save(test_y_path,  y_test_base)
    else:
        X_train = np.load(train_X_path)
        y_train = np.load(train_y_path)
        X_test = np.load(test_X_path)
        y_test = np.load(test_y_path)
    print("\ttime ", time() - start)

    projection_size = X_train.shape[0]

    X_proj = np.copy(X_train[:projection_size])
    N, H, W, C = X_proj.shape
    new_shape = (N, H*W*C)
    X_proj = np.reshape(X_proj, new_shape)

    print("t-SNE projecting Cifar-10 dataset...")
    start = time()

    proj_path = base_dir + "tsne_proj.npy"
    if not os.path.exists(proj_path):
        # tsne = manifold.TSNE(n_components=2, perplexity=40.0)
        tsne = TSNE(n_components=2, perplexity=40.0, n_jobs=8)
        proj_tsne = tsne.fit_transform(X_proj)
        proj_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        proj_tsne = proj_scaler.fit_transform(proj_tsne)
        np.save(proj_path,    proj_tsne)
    else:
        proj_tsne = np.load(proj_path)
    print("\ttime ", time() - start)

    subset_size = 15000
    print("ILAMP")
    start = time()
    ilamp_path = base_dir + "ilamp.joblib"
    if not os.path.exists(ilamp_path):
        k_ilamp = 30
        ilamp = ILAMP(n_neighbors=k_ilamp)
        ilamp.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        ilamp.save(ilamp_path)
        # joblib.dump(ilamp, ilamp_path)
    # else:
    #     ilamp = joblib.load(ilamp_path)
    print("\ttime ", time() - start)

    print("RBFInv - Control Points")
    start = time()
    irbfcp_path = base_dir + "irbf_cp.joblib"
    if not os.path.exists(irbfcp_path):
        EPS = 50000
        irbf_cp = RBFInv(num_ctrl=200, mode='rols', kernel='gaussian',
                         eps=EPS, normalize_c=True, normalize_d=True)
        irbf_cp.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        irbf_cp.save(irbfcp_path)
        # joblib.dump(irbf_cp, irbfcp_path)
    # else:
    #     irbf_cp = joblib.load(irbfcp_path)
    print("\ttime ", time() - start)

    print("RBFInv - Neighbors")
    start = time()
    irbfn_path = base_dir + "irbf_neighbors.joblib"
    if not os.path.exists(irbfn_path):
        EPS = 5000000
        irbf_neighbors = RBFInv(num_ctrl=20, mode='neighbors', eps=EPS,
                                kernel='gaussian', normalize_c=True,
                                normalize_d=True)
        irbf_neighbors.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        irbf_neighbors.save(irbfn_path)
        # joblib.dump(irbf_neighbors, irbfn_path)
    # else:
    #     irbf_neighbors = joblib.load(irbfn_path)
    print("\ttime ", time() - start)

    print("RBFInv - Cluster")
    start = time()
    irbfc_path = base_dir + "irbf_cluster.joblib"
    if not os.path.exists(irbfc_path):
        EPS = 50000
        irbf_cluster = RBFInv(num_ctrl=50, mode='cluster', eps=EPS,
                              kernel='gaussian', normalize_c=True,
                              normalize_d=True)
        irbf_cluster.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        irbf_cluster.save(irbfc_path)
        # joblib.dump(irbf_cluster, irbfc_path)
    # else:
    #     irbf_cluster = joblib.load(irbfc_path)
    print("\ttime ", time() - start)

    print("NNInv")
    start = time()
    nninv_path = base_dir + "nninv.joblib"
    if not os.path.exists(nninv_path):
        nninv = NNInv()
        nninv.fit(X_proj[:subset_size], proj_tsne[:subset_size])
        nninv.save(nninv_path, base_dir + 'nninv_keras.hdf5')
    # else:
    #     nninv = NNInv()
    #     nninv.load(nninv_path)
    print("\ttime ", time() - start)

    print("Training classifier CNN...")
    start = time()
    clf_path = base_dir + "cifar_cnn.hdf5"
    pred_path = base_dir + "y_pred_clf.npy"
    X_proj = X_proj.reshape((X_proj.shape[0],) + (H, W, C))
    input_shape = (H, W, C)
    X_train = X_train.reshape((X_train.shape[0],) + input_shape)
    X_test = X_test.reshape((X_test.shape[0],) + input_shape)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    if not os.path.exists(clf_path):
        datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1,
                                     height_shift_range=0.1, horizontal_flip=True)
        datagen.fit(X_train)
        clf = CNN_Cifar()
        batch_size = 64
        steps = X_train.shape[0]//batch_size
        clf.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                          steps_per_epoch=steps, epochs=125, verbose=1,
                          validation_data=(X_test, y_test),
                          callbacks=[LearningRateScheduler(lr_schedule)])
        accuracy = clf.evaluate(X_test, y_test, verbose=0)
        y_proj_pred = np.argmax(clf.predict(X_proj), axis=1)
        np.save(pred_path, y_proj_pred)
        print("\tAccuracy on test data: ", accuracy)
        clf.save(clf_path)
    # else:
    #     clf = keras.models.load_model(clf_path)
    #     accuracy = clf.evaluate(X_test, y_test, verbose=0)
    #     y_proj_pred = np.argmax(clf.predict(X_proj), axis=1)
    print("\ttime ", time() - start)

    print("Saving data for Cifar-10...")
    start = time()

    clfs = [clf_path]
    inv_projs = [ilamp_path, irbfcp_path, irbfn_path, irbfc_path, nninv_path]
    preds = [pred_path]

    data_json = {'X_train'  : train_X_path,
                 'y_train'  : train_y_path,
                 'X_test'   : test_X_path,
                 'y_test'   : test_y_path,
                 'projs'    : [proj_path],
                 'inv_projs': inv_projs,
                 'y_preds'  : preds, 
                 'clfs'     : clfs}

    with open(base_dir + "cifar.json", 'w') as outfile:
        json.dump(data_json, outfile)
    print("\tFinished saving data...", time() - start)


def main():
    if len(sys.argv) < 3:
        print("Usage: ./prepare_data <dataset> <dir>")
        sys.exit(0)
    base_dir = sys.argv[2]
    # Toy()
    # Wine()
    # Segmentation()
    if sys.argv[1] == "fashion_mnist":
        FashionMNIST(base_dir)
    elif sys.argv[1] == "cifar":
        Cifar10(base_dir)
    elif sys.argv[1] == "mnist":
        MNIST(base_dir)


if __name__ == "__main__":
    main()
