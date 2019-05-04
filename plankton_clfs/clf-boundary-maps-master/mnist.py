import numpy as np
from boundarymap import CLF
from boundarymap import Grid
from sklearn.preprocessing import MinMaxScaler
import json
import time
import sys

import data
# import joblib
from nninv import NNInv
from lamp import ILAMP
from lamp import RBFInv

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from umap import UMAP
from MulticoreTSNE import MulticoreTSNE as TSNE


np.random.seed(0)
keras.backend.set_learning_phase(0)


def save_grid(path, R, N, clf, inv_proj, X_nd, X_2d, clf_path, inv_proj_path):
    grid = Grid()
    grid.fit(R, N, clf, inv_proj, X_nd=X_nd, X_2d=X_2d, syn_only=True)
    grid.BoundaryMapBatch()
    grid.dist2D()
    # grid.distnD_batch()
    # grid.distnD2_batch()
    # grid.distnD3_batch()
    grid.save(path, clf_path, inv_proj_path)


def save_json_ui(path, grid, clf, inv_proj_type, inv_proj, y_true, y_pred):
    data_grid = {}
    data_grid['grid'] = grid
    data_grid['clf'] = clf
    data_grid['inv_proj_type'] = inv_proj_type
    data_grid['inv_proj'] = inv_proj
    data_grid['y_true'] = y_true
    data_grid['y_pred'] = y_pred

    with open(path, 'w') as outfile:
        json.dump(data_grid, outfile)


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


def main():
    if len(sys.argv) < 2:
        print("Usage: ./fashion_mnist.py <base_dir>")
        sys.exit(0)
    base_dir = sys.argv[1]

    print("Load dataset\n\n")
    s = time.time()
    X_train, y_train = data.LoadMNISTData('train', base_dir + 'orig/')

    train_y_path = base_dir + "y_train.npy"
    np.save(train_y_path, y_train)

    X_test, y_test = data.LoadMNISTData('test', base_dir + 'orig/')

    X_nd = np.copy(X_train)
    X_nd = X_nd.reshape((X_nd.shape[0], X_nd.shape[1]*X_nd.shape[2]))

    projection_size = 60000
    X_proj = np.copy(X_train[:projection_size])
    new_shape = (X_proj.shape[0], X_proj.shape[1]*X_proj.shape[2])
    X_proj = np.reshape(X_proj, new_shape)
    print("\ttime: ", time.time() - s)

    scaler = MinMaxScaler(feature_range=(0, 1))

    print("TSNE Projection")
    s = time.time()
    tsne = TSNE(n_components=2, random_state=420, perplexity=25.0,
                n_iter=3000, n_iter_without_progress=300, n_jobs=4)
    tsne_proj = tsne.fit_transform(X_proj)
    tsne_proj = scaler.fit_transform(tsne_proj)
    print("\ttime: ", time.time() - s)

    print("UMAP Projection")
    s = time.time()
    umap_proj = UMAP(n_components=2, random_state=420, n_neighbors=5,
                     min_dist=0.3).fit_transform(X_proj)
    umap_proj = scaler.fit_transform(umap_proj)
    print("\ttime: ", time.time() - s)

    subset_size = 15000
    print("\n\nILAMP tSNE")
    s = time.time()
    k_ilamp = 20
    ilamp_tsne = ILAMP(n_neighbors=k_ilamp)
    ilamp_tsne.fit(X_proj[:subset_size], tsne_proj[:subset_size])
    ilamp_tsne_path = base_dir + "ilamp_tsne.joblib"
    ilamp_tsne.save(ilamp_tsne_path)
    print("\ttime: ", time.time() - s)

    print("\n\nILAMP UMAP")
    s = time.time()
    ilamp_umap = ILAMP(n_neighbors=k_ilamp)
    ilamp_umap.fit(X_proj[:subset_size], umap_proj[:subset_size])
    ilamp_umap_path = base_dir + "ilamp_umap.joblib"
    ilamp_umap.save(ilamp_umap_path)
    print("\ttime: ", time.time() - s)

    print("\n\nRBFInv CTRL PTS TSNE")
    s = time.time()
    EPS = 50000
    irbfcp_tsne = RBFInv(num_ctrl=400, mode='rols', kernel='gaussian',
                         eps=EPS, normalize_c=True, normalize_d=True)
    irbfcp_tsne.fit(X_proj[:subset_size], tsne_proj[:subset_size])
    irbfcp_tsne_path = base_dir + "irbfcp_tsne.joblib"
    irbfcp_tsne.save(irbfcp_tsne_path)
    print("\ttime: ", time.time() - s)

    print("\n\nRBFInv CTRL PTS UMAP")
    s = time.time()
    EPS = 50000
    irbfcp_umap = RBFInv(num_ctrl=400, mode='rols', kernel='gaussian',
                         eps=EPS, normalize_c=True, normalize_d=True)
    irbfcp_umap.fit(X_proj[:subset_size], umap_proj[:subset_size])
    irbfcp_umap_path = base_dir + "irbfcp_umap.joblib"
    irbfcp_umap.save(irbfcp_umap_path)
    print("\ttime: ", time.time() - s)

    print("\n\nRBFInv CLUSTER TSNE")
    s = time.time()
    EPS = 50000
    irbfc_tsne = RBFInv(num_ctrl=50, mode='cluster', kernel='gaussian',
                        eps=EPS, normalize_c=True, normalize_d=True)
    irbfc_tsne.fit(X_proj[:subset_size], tsne_proj[:subset_size])
    irbfc_tsne_path = base_dir + "irbfc_tsne.joblib"
    irbfc_tsne.save(irbfc_tsne_path)
    print("\ttime: ", time.time() - s)

    print("\n\nRBFInv CLUSTER UMAP")
    s = time.time()
    EPS = 50000
    irbfc_umap = RBFInv(num_ctrl=50, mode='cluster', kernel='gaussian',
                        eps=EPS, normalize_c=True, normalize_d=True)
    irbfc_umap.fit(X_proj[:subset_size], tsne_proj[:subset_size])
    irbfc_umap_path = base_dir + "irbfc_umap.joblib"
    irbfc_umap.save(irbfc_umap_path)
    print("\ttime: ", time.time() - s)

    print("\n\nNNInv TSNE")
    s = time.time()
    nninv_tsne = NNInv()
    nninv_tsne.fit(X_proj[:subset_size], tsne_proj[:subset_size])
    nninv_tsne_path = base_dir + "nninv_tsne.joblib"
    nninv_tsne.save(nninv_tsne_path, base_dir + 'nninv_tsne_keras.hdf5')
    print("\ttime: ", time.time() - s)

    print("\n\nNNInv UMAP")
    s = time.time()
    nninv_umap = NNInv()
    nninv_umap.fit(X_proj[:subset_size], umap_proj[:subset_size])
    nninv_umap_path = base_dir + "nninv_umap.joblib"
    nninv_umap.save(nninv_umap_path, base_dir + 'nninv_umap_keras.hdf5')
    print("\ttime: ", time.time() - s)

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    X_train = X_train.reshape((X_train.shape[0],) + input_shape)
    X_test = X_test.reshape((X_test.shape[0],) + input_shape)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    X_proj = X_proj.reshape((X_proj.shape[0],) + input_shape)

    print("\n\nTraining classifier")
    s = time.time()
    clf_keras = CNNModel(input_shape, 10)
    clf_keras.fit(X_train, y_train, batch_size=128, epochs=14, verbose=1,
                  validation_data=(X_test, y_test))
    print("\tAccuracy on test data: ", clf_keras.evaluate(X_test, y_test, verbose=0))
    clf_keras_path = base_dir + "mnist_cnn.hdf5"
    clf_keras.save(clf_keras_path)
    y_proj_pred = np.argmax(clf_keras.predict(X_proj), axis=1)
    pred_path = base_dir + "y_pred_clf.npy"
    np.save(pred_path, y_proj_pred)

    clf = CLF(clf=clf_keras, clf_type='keras_cnn', clf_path=clf_keras_path,
              shape=input_shape)
    clf_path = base_dir + 'mnist_cnn.json'
    clf.save_json(clf_path)
    print("\ttime: ", time.time() - s)

    N = 1
    R = 500

    print("\n\nGRID ILAMP tSNE")
    s = time.time()
    grid_ilamp_tsne_path = base_dir + 'grid_ilamp_tsne.joblib'
    save_grid(grid_ilamp_tsne_path, R, N, clf, ilamp_tsne, X_nd,
              tsne_proj, clf_path, ilamp_tsne_path)
    ui_grid_ilamp_tsne_path = base_dir + 'mnist_500_' + 'ui_ilamp_tsne.json'
    save_json_ui(ui_grid_ilamp_tsne_path, grid_ilamp_tsne_path, clf_path,
                 "ilamp", ilamp_tsne_path, train_y_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID ILAMP UMAP")
    s = time.time()
    grid_ilamp_umap_path = base_dir + 'grid_ilamp_umap.joblib'
    save_grid(grid_ilamp_umap_path, R, N, clf, ilamp_umap, X_nd,
              umap_proj, clf_path, ilamp_umap_path)
    ui_grid_ilamp_umap_path = base_dir + 'mnist_500_' + 'ui_ilamp_umap.json'
    save_json_ui(ui_grid_ilamp_umap_path, grid_ilamp_umap_path, clf_path,
                 "ilamp", ilamp_umap_path, train_y_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID NNInv tSNE")
    s = time.time()
    grid_nninv_tsne_path = base_dir + 'grid_nninv_tsne.joblib'
    save_grid(grid_nninv_tsne_path, R, N, clf, nninv_tsne, X_nd,
              tsne_proj, clf_path, nninv_tsne_path)
    ui_grid_nninv_tsne_path = base_dir + 'mnist_500_' + '_ui_nninv_tsne.json'
    save_json_ui(ui_grid_nninv_tsne_path, grid_nninv_tsne_path, clf_path,
                 "nninv", nninv_tsne_path, train_y_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID NNInv UMAP")
    s = time.time()
    grid_nninv_umap_path = base_dir + 'grid_nninv_umap.joblib'
    save_grid(grid_nninv_umap_path, R, N, clf, nninv_umap, X_nd,
              umap_proj, clf_path, nninv_umap_path)
    ui_grid_nninv_umap_path = base_dir + 'mnist_500_' + 'ui_nninv_umap.json'
    save_json_ui(ui_grid_nninv_umap_path, grid_nninv_umap_path, clf_path,
                 "nninv", nninv_umap_path, train_y_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID RBFInv CTRL PTS tSNE")
    s = time.time()
    grid_irbfcp_tsne_path = base_dir + 'grid_irbfcp_tsne.joblib'
    save_grid(grid_irbfcp_tsne_path, R, N, clf, irbfcp_tsne, X_nd,
              tsne_proj, clf_path, irbfcp_tsne_path)
    ui_grid_irbfcp_tsne_path = base_dir + 'mnist_500_' + 'ui_irbfcp_tsne.json'
    save_json_ui(ui_grid_irbfcp_tsne_path, grid_irbfcp_tsne_path, clf_path,
                 "rbf", irbfcp_tsne_path, train_y_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID RBFInv CTRL PTS UMAP")
    s = time.time()
    grid_irbfcp_umap_path = base_dir + 'grid_irbfcp_umap.joblib'
    save_grid(grid_irbfcp_umap_path, R, N, clf, irbfcp_umap, X_nd,
              umap_proj, clf_path, irbfcp_umap_path)
    ui_grid_irbfcp_umap_path = base_dir + 'mnist_500_' + 'ui_irbfcp_umap.json'
    save_json_ui(ui_grid_irbfcp_umap_path, grid_irbfcp_umap_path, clf_path,
                 "rbf", irbfcp_umap_path, train_y_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID RBFInv CLUSTER tSNE")
    s = time.time()
    grid_irbfc_tsne_path = base_dir + 'grid_irbfc_tsne.joblib'
    save_grid(grid_irbfc_tsne_path, R, N, clf, irbfc_tsne, X_nd,
              tsne_proj, clf_path, irbfc_tsne_path)
    ui_grid_irbfc_tsne_path = base_dir + 'mnist_500_' + 'ui_irbfc_tsne.json'
    save_json_ui(ui_grid_irbfc_tsne_path, grid_irbfc_tsne_path, clf_path,
                 "rbf", irbfc_tsne_path, train_y_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID RBFInv CLUSTER UMAP")
    s = time.time()
    grid_irbfc_umap_path = base_dir + 'grid_irbfc_umap.joblib'
    save_grid(grid_irbfc_umap_path, R, N, clf, irbfc_umap, X_nd,
              umap_proj, clf_path, irbfc_umap_path)
    ui_grid_irbfc_umap_path = base_dir + 'mnist_500_' + 'ui_irbfc_umap.json'
    save_json_ui(ui_grid_irbfc_umap_path, grid_irbfc_umap_path, clf_path,
                 "rbf", irbfc_umap_path, train_y_path, pred_path)
    print("\ttime: ", time.time() - s)

    # with open(base_dir + "mnist.json") as f:
    #     data_json_base = json.load(f)

    # X_train = np.load(data_json_base['X_train'])
    # X_proj = np.load(data_json_base['projs'][0])
    # input_shape = (X_train.shape[1], X_train.shape[2], 1)
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))

    # clf = CLF()
    # clf_path = data_json_base['clfs'][0]
    # clf.LoadKerasModel(clf_path, "CNN", input_shape)
    # clf.save_json(base_dir + "mnist_cnn.json")

    # inv_projs_path = data_json_base['inv_projs']
    # ilamp_path = inv_projs_path[0]
    # irbfcp_path = inv_projs_path[1]
    # irbfn_path = inv_projs_path[2]
    # irbfc_path = inv_projs_path[3]
    # nninv_path = inv_projs_path[4]

    # ilamp = ILAMP()
    # ilamp.load(ilamp_path)

    # irbf_cp = RBFInv()
    # irbf_cp.load(irbfcp_path)

    # irbf_neighbors = RBFInv()
    # irbf_neighbors.load(irbfn_path)

    # irbf_cluster = RBFInv()
    # irbf_cluster.load(irbfc_path)

    # nninv = NNInv()
    # nninv.load(nninv_path)

    # # inv_projs = []
    # # for inv in inv_projs_path[:-1]:
    # #     inv_projs.append(joblib.load(inv))

    # # ilamp = inv_projs[0]
    # # irbf_cp = inv_projs[1]
    # # irbf_neighbors = inv_projs[2]
    # # irbf_cluster = inv_projs[3]

    # N = 1
    # R = 500

    # grid_ilamp_path = base_dir + 'grid_ilamp.joblib'
    # if not os.path.isfile(grid_ilamp_path):
    #     print("MNIST ILAMP GRID")
    #     s = time.time()
    #     grid_ilamp = Grid()
    #     grid_ilamp.fit(R, N, clf, ilamp, X_nd=X_train, X_2d=X_proj)
    #     grid_ilamp.BoundaryMapBatch()
    #     grid_ilamp.dist2D()
    #     # grid_ilamp.distnD_batch()
    #     # grid_ilamp.distnD2_batch()
    #     # grid_ilamp.distnD3_batch()
    #     grid_ilamp.save(grid_ilamp_path)
    #     print("\ttime: ", time.time() - s)

    # grid_irbfcp_path = base_dir + 'grid_irbfcp.joblib'
    # if not os.path.isfile(grid_irbfcp_path):
    #     print("MNIST RBF CTRL PTS GRID")
    #     s = time.time()
    #     grid_irbfcp = Grid()
    #     grid_irbfcp.fit(R, N, clf, irbf_cp, X_nd=X_train, X_2d=X_proj)
    #     grid_irbfcp.BoundaryMapBatch()
    #     grid_irbfcp.dist2D()
    #     # grid_irbfcp.distnD_batch()
    #     # grid_irbfcp.distnD2_batch()
    #     # grid_irbfcp.distnD3_batch()
    #     grid_irbfcp.save(grid_irbfcp_path)
    #     print("\ttime: ", time.time() - s)

    # grid_irbfn_path = base_dir + 'grid_irbfn.joblib'
    # if not os.path.isfile(grid_irbfn_path):
    #     print("MNIST RBF NEIGHBORS GRID")
    #     s = time.time()
    #     grid_irbfn = Grid()
    #     grid_irbfn.fit(R, N, clf, irbf_neighbors, X_nd=X_train, X_2d=X_proj)
    #     grid_irbfn.BoundaryMapBatch()
    #     grid_irbfn.dist2D()
    #     # grid_irbfn.distnD_batch()
    #     # grid_irbfn.distnD2_batch()
    #     # grid_irbfn.distnD3_batch()
    #     grid_irbfn.save(grid_irbfn_path)
    #     print("\ttime: ", time.time() - s)

    # grid_irbfc_path = base_dir + 'grid_irbfc.joblib'
    # if not os.path.isfile(grid_irbfc_path):
    #     print("MNIST RBF CLUSTER GRID")
    #     s = time.time()
    #     grid_irbfc = Grid()
    #     grid_irbfc.fit(R, N, clf, irbf_cluster, X_nd=X_train, X_2d=X_proj)
    #     grid_irbfc.BoundaryMapBatch()
    #     grid_irbfc.dist2D()
    #     # grid_irbfc.distnD_batch()
    #     # grid_irbfc.distnD2_batch()
    #     # grid_irbfc.distnD3_batch()
    #     grid_irbfn.save(grid_irbfc_path)
    #     print("\ttime: ", time.time() - s)

    # grid_nninv_path = base_dir + 'grid_nninv.joblib'
    # if not os.path.isfile(grid_nninv_path):
    #     print("MNIST NNInv GRID")
    #     s = time.time()
    #     grid_nn = Grid()
    #     grid_nn.fit(R, N, clf, nninv, X_nd=X_train, X_2d=X_proj)
    #     grid_nn.BoundaryMapBatch()
    #     grid_nn.dist2D()
    #     # grid_nn.distnD_batch()
    #     # grid_nn.distnD2_batch()
    #     # grid_nn.distnD3_batch()
    #     grid_nn.save(grid_nninv_path)
    #     print("\ttime: ", time.time() - s)


if __name__ == "__main__":
    main()
