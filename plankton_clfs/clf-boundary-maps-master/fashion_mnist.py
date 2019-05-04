import numpy as np
from boundarymap import CLF
from boundarymap import Grid
from sklearn.preprocessing import MinMaxScaler
import json
import time
import os.path
import os
import sys

import data
# import joblib
from nninv import NNInv
from lamp import ILAMP
from lamp import RBFInv

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

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


def main():
    if len(sys.argv) < 2:
        print("Usage: ./fashion_mnist.py <base_dir>")
        sys.exit(0)
    base_dir = sys.argv[1]

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    print("Load dataset")
    s = time.time()
    X_base, y_base = data.LoadFashionMNIST('train', base_dir + 'orig/')
    X_test, y_test = data.LoadFashionMNIST('test', base_dir + 'orig/')

    train_size = 40000
    X_train = np.copy(X_base[:train_size])
    y_train = np.copy(y_base[:train_size])

    X_valid = np.copy(X_base[train_size:])
    y_valid = np.copy(y_base[train_size:])

    # X_nd is just X_train but reshaped to 1d
    X_nd = np.copy(X_train)
    X_nd = X_nd.reshape((X_nd.shape[0], X_nd.shape[1]*X_nd.shape[2]))

    y_train_path = base_dir + "y_train.npy"
    np.save(y_train_path, y_train)
    y_valid_path = base_dir + "y_valid.npy"
    np.save(y_valid_path, y_valid)
    X_valid_vec = np.copy(X_valid)
    N, W, H = X_valid_vec.shape
    X_valid_vec = X_valid_vec.reshape((N, W*H))
    X_valid_path = base_dir + "X_valid.npy"
    np.save(X_valid_path, X_valid_vec)

    projection_size = train_size
    X_proj = np.copy(X_train[:projection_size])
    new_shape = (X_proj.shape[0], X_proj.shape[1]*X_proj.shape[2])
    X_proj = np.reshape(X_proj, new_shape)
    print("\ttime: ", time.time() - s)

    scaler = MinMaxScaler(feature_range=(0, 1))

    print("\n\nTSNE Projection")
    s = time.time()
    tsne = TSNE(n_components=2, random_state=420, perplexity=10.0,
                n_iter=1000, n_iter_without_progress=300, n_jobs=4)
    tsne_proj = tsne.fit_transform(X_proj)
    tsne_proj = scaler.fit_transform(tsne_proj)
    print("\ttime: ", time.time() - s)

    print("\n\nUMAP Projection")
    s = time.time()
    umap_proj = UMAP(n_components=2, random_state=420, n_neighbors=10,
                     min_dist=0.5).fit_transform(X_proj)
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
    irbfc_umap.fit(X_proj[:subset_size], umap_proj[:subset_size])
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
    X_valid = X_valid.reshape((X_valid.shape[0],) + input_shape)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_valid = keras.utils.to_categorical(y_valid, 10)

    X_proj = X_proj.reshape((X_proj.shape[0],) + input_shape)

    print("\n\nTraining classifier")
    s = time.time()
    clf_keras = CNN_FM()
    clf_keras.fit(X_train, y_train, batch_size=128, epochs=14, verbose=1,
                  validation_data=(X_valid, y_valid))
    valid_acc = clf_keras.evaluate(X_test, y_test, verbose=0)
    print("\tAccuracy on validation data: ", valid_acc)
    test_acc = clf_keras.evaluate(X_test, y_test, verbose=0)
    print("\tAccuracy on test data: ", test_acc)
    clf_keras_path = base_dir + "fm_cnn.hdf5"
    clf_keras.save(clf_keras_path)
    y_proj_pred = np.argmax(clf_keras.predict(X_proj), axis=1)
    pred_path = base_dir + "y_pred_clf.npy"
    np.save(pred_path, y_proj_pred)

    clf = CLF(clf=clf_keras, clf_type='keras_cnn', clf_path=clf_keras_path,
              shape=input_shape)
    clf_path = base_dir + 'fm_cnn.json'
    clf.save_json(clf_path)
    print("\ttime: ", time.time() - s)

    N = 1
    R = 500

    print("\n\nGRID ILAMP tSNE")
    s = time.time()
    grid_ilamp_tsne_path = base_dir + 'grid_ilamp_tsne.joblib'
    save_grid(grid_ilamp_tsne_path, R, N, clf, ilamp_tsne, X_nd,
              tsne_proj, clf_path, ilamp_tsne_path)
    ui_grid_ilamp_tsne_path = base_dir + 'fm_500_' + 'ui_ilamp_tsne.json'
    save_json_ui(ui_grid_ilamp_tsne_path, grid_ilamp_tsne_path, clf_path,
                 "ilamp", ilamp_tsne_path, y_train_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID ILAMP UMAP")
    s = time.time()
    grid_ilamp_umap_path = base_dir + 'grid_ilamp_umap.joblib'
    save_grid(grid_ilamp_umap_path, R, N, clf, ilamp_umap, X_nd,
              umap_proj, clf_path, ilamp_umap_path)
    ui_grid_ilamp_umap_path = base_dir + 'fm_500_' + 'ui_ilamp_umap.json'
    save_json_ui(ui_grid_ilamp_umap_path, grid_ilamp_umap_path, clf_path,
                 "ilamp", ilamp_umap_path, y_train_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID NNInv tSNE")
    s = time.time()
    grid_nninv_tsne_path = base_dir + 'grid_nninv_tsne.joblib'
    save_grid(grid_nninv_tsne_path, R, N, clf, nninv_tsne, X_nd,
              tsne_proj, clf_path, nninv_tsne_path)
    ui_grid_nninv_tsne_path = base_dir + 'fm_500_' + 'ui_nninv_tsne.json'
    save_json_ui(ui_grid_nninv_tsne_path, grid_nninv_tsne_path, clf_path,
                 "nninv", nninv_tsne_path, y_train_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID NNInv UMAP")
    s = time.time()
    grid_nninv_umap_path = base_dir + 'grid_nninv_umap.joblib'
    save_grid(grid_nninv_umap_path, R, N, clf, nninv_umap, X_nd,
              umap_proj, clf_path, nninv_umap_path)
    ui_grid_nninv_umap_path = base_dir + 'fm_500_' + 'ui_nninv_umap.json'
    save_json_ui(ui_grid_nninv_umap_path, grid_nninv_umap_path, clf_path,
                 "nninv", nninv_umap_path, y_train_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID RBFInv CTRL PTS tSNE")
    s = time.time()
    grid_irbfcp_tsne_path = base_dir + 'grid_irbfcp_tsne.joblib'
    save_grid(grid_irbfcp_tsne_path, R, N, clf, irbfcp_tsne, X_nd,
              tsne_proj, clf_path, irbfcp_tsne_path)
    ui_grid_irbfcp_tsne_path = base_dir + 'fm_500_' + 'ui_irbfcp_tsne.json'
    save_json_ui(ui_grid_irbfcp_tsne_path, grid_irbfcp_tsne_path, clf_path,
                 "rbf", irbfcp_tsne_path, y_train_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID RBFInv CTRL PTS UMAP")
    s = time.time()
    grid_irbfcp_umap_path = base_dir + 'grid_irbfcp_umap.joblib'
    save_grid(grid_irbfcp_umap_path, R, N, clf, irbfcp_umap, X_nd,
              umap_proj, clf_path, irbfcp_umap_path)
    ui_grid_irbfcp_umap_path = base_dir + 'fm_500_' + 'ui_irbfcp_umap.json'
    save_json_ui(ui_grid_irbfcp_umap_path, grid_irbfcp_umap_path, clf_path,
                 "rbf", irbfcp_umap_path, y_train_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID RBFInv CLUSTER tSNE")
    s = time.time()
    grid_irbfc_tsne_path = base_dir + 'grid_irbfc_tsne.joblib'
    save_grid(grid_irbfc_tsne_path, R, N, clf, irbfc_tsne, X_nd,
              tsne_proj, clf_path, irbfc_tsne_path)
    ui_grid_irbfc_tsne_path = base_dir + 'fm_500_' + 'ui_irbfc_tsne.json'
    save_json_ui(ui_grid_irbfc_tsne_path, grid_irbfc_tsne_path, clf_path,
                 "rbf", irbfc_tsne_path, y_train_path, pred_path)
    print("\ttime: ", time.time() - s)

    print("\n\nGRID RBFInv CLUSTER UMAP")
    s = time.time()
    grid_irbfc_umap_path = base_dir + 'grid_irbfc_umap.joblib'
    save_grid(grid_irbfc_umap_path, R, N, clf, irbfc_umap, X_nd,
              umap_proj, clf_path, irbfc_umap_path)
    ui_grid_irbfc_umap_path = base_dir + 'fm_500_' + 'ui_irbfc_umap.json'
    save_json_ui(ui_grid_irbfc_umap_path, grid_irbfc_umap_path, clf_path,
                 "rbf", irbfc_umap_path, y_train_path, pred_path)
    print("\ttime: ", time.time() - s)


if __name__ == "__main__":
    main()


# import numpy as np
# from boundarymap import CLF
# from boundarymap import Grid
# import json
# import time
# import os.path
# import joblib
# from nninv import NNInv
# from lamp import ILAMP
# from lamp import RBFInv
# import keras
# import sys
# 
# 
# np.random.seed(0)
# keras.backend.set_learning_phase(0)
# 
# 
# def load_inv_proj(path, ip_type):
#     if ip_type == 'RBFInv':
#         inv_proj = RBFInv()
#     elif ip_type == 'ILAMP':
#         inv_proj = ILAMP()
#     elif ip_type == 'NNInv':
#         inv_proj = NNInv()
# 
#     inv_proj.load(path)
#     return inv_proj
# 
# 
# def save_grid(path, R, N, clf, inv_proj, X_nd, X_2d, clf_path, inv_proj_path):
#     grid = Grid()
#     grid.fit(R, N, clf, inv_proj, X_nd=X_nd, X_2d=X_2d)
#     grid.BoundaryMapBatch()
#     grid.dist2D()
#     # grid.distnD_batch()
#     # grid.distnD2_batch()
#     # grid.distnD3_batch()
#     grid.save(path, clf_path, inv_proj_path)
# 
# 
# def main():
#     if len(sys.argv) < 2:
#         print("Usage: ./fashion_mnist.py <base_dir>")
#         sys.exit(0)
#     base_dir = sys.argv[1]
# 
#     with open(base_dir + "fm.json") as f:
#         data_json_base = json.load(f)
# 
#     X_train = np.load(data_json_base['X_train'])
#     Xp_tsne = np.load(data_json_base['projs'][0])
#     Xp_umap = np.load(data_json_base['projs'][1])
#     input_shape = (X_train.shape[1], X_train.shape[2], 1)
#     X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
# 
#     clf = CLF()
#     clf_path = data_json_base['clfs'][0]
#     clf.LoadKerasModel(clf_path, "CNN", input_shape)
#     clf_path = base_dir + "fm_cnn.json"
#     clf.save_json(clf_path)
# 
#     inv_projs_path = data_json_base['inv_projs']
#     ilamp_tsne_path = inv_projs_path[0]
#     irbfcp_tsne_path = inv_projs_path[1]
#     # irbfn_path = inv_projs_path[2]
#     irbfc_tsne_path = inv_projs_path[2]
#     nninv_tsne_path = inv_projs_path[3]
# 
#     ilamp_umap_path = inv_projs_path[4]
#     irbfcp_umap_path = inv_projs_path[5]
#     # irbfn_path = inv_projs_path[2]
#     irbfc_umap_path = inv_projs_path[6]
#     nninv_umap_path = inv_projs_path[7]
# 
#     ilamp_tsne = load_inv_proj(ilamp_tsne_path, 'ILAMP')
#     irbfcp_tsne = load_inv_proj(irbfcp_tsne_path, 'RBFInv')
#     irbfc_tsne = load_inv_proj(irbfc_tsne_path, 'RBFInv')
#     nninv_tsne = load_inv_proj(nninv_tsne_path, 'NNInv')
# 
#     ilamp_umap = load_inv_proj(ilamp_umap_path, 'ILAMP')
#     irbfcp_umap = load_inv_proj(irbfcp_umap_path, 'RBFInv')
#     irbfc_umap = load_inv_proj(irbfc_umap_path, 'RBFInv')
#     nninv_umap = load_inv_proj(nninv_umap_path, 'NNInv')
# 
#     N = 1
#     R = 500
# 
#     grid_ilamp_tsne_path = base_dir + 'grid_ilamp_tsne.joblib'
#     if not os.path.isfile(grid_ilamp_tsne_path):
#         print("GRID ILAMP TSNE")
#         save_grid(grid_ilamp_tsne_path, R, N, clf, ilamp_tsne, X_train,
#                   Xp_tsne, clf_path, ilamp_tsne_path)
#         # grid_ilamp = Grid()
#         # grid_ilamp.fit(R, N, clf, ilamp, X_nd=X_train, X_2d=X_proj)
#         # grid_ilamp.BoundaryMapBatch()
#         # grid_ilamp.dist2D()
#         # # grid_ilamp.distnD_batch()
#         # # grid_ilamp.distnD2_batch()
#         # # grid_ilamp.distnD3_batch()
#         # grid_ilamp.save(grid_ilamp_path, clf_path, ilamp_path)
# 
#     grid_irbfcp_tsne_path = base_dir + 'grid_irbfcp_tsne.joblib'
#     if not os.path.isfile(grid_irbfcp_tsne_path):
#         print("GRID RBF CTRL PTS TSNE")
#         save_grid(grid_irbfcp_tsne_path, R, N, clf, irbfcp_tsne, X_train,
#                   Xp_tsne, clf_path, irbfcp_tsne_path)
#         # grid_irbfcp = Grid()
#         # grid_irbfcp.fit(R, N, clf, irbf_cp, X_nd=X_train, X_2d=X_proj)
#         # grid_irbfcp.BoundaryMapBatch()
#         # grid_irbfcp.dist2D()
#         # # grid_irbfcp.distnD_batch()
#         # # grid_irbfcp.distnD2_batch()
#         # # grid_irbfcp.distnD3_batch()
#         # grid_irbfcp.save(grid_irbfcp_path, clf_path, irbfcp_path)
# 
#     # grid_irbfn_tsne_path = base_dir + 'grid_irbfn_tsne.joblib'
#     # if not os.path.isfile(grid_irbfn_tsne_path):
#     #     save_grid(grid_irbfn_tsne_path, R, N, clf, X_train, X_proj,
#     #               irbfn_tsne, irbfn_tsne_path)
#     #     grid_irbfn = Grid()
#     #     grid_irbfn.fit(R, N, clf, irbf_neighbors, X_nd=X_train, X_2d=X_proj)
#     #     grid_irbfn.BoundaryMapBatch()
#     #     grid_irbfn.dist2D()
#     #     # grid_irbfn.distnD_batch()
#     #     # grid_irbfn.distnD2_batch()
#     #     # grid_irbfn.distnD3_batch()
#     #     grid_irbfn.save(grid_irbfn_path, clf_path, irbfn_path)
# 
#     grid_irbfc_tsne_path = base_dir + 'grid_irbfc_tsne.joblib'
#     if not os.path.isfile(grid_irbfc_tsne_path):
#         print("GRID RBF CLUSTERS TSNE")
#         save_grid(grid_irbfc_tsne_path, R, N, clf, irbfc_tsne, X_train,
#                   Xp_tsne, clf_path, irbfc_tsne_path)
#         # grid_irbfc = Grid()
#         # grid_irbfc.fit(R, N, clf, irbf_cluster, X_nd=X_train, X_2d=X_proj)
#         # grid_irbfc.BoundaryMapBatch()
#         # grid_irbfc.dist2D()
#         # # grid_irbfc.distnD_batch()
#         # # grid_irbfc.distnD2_batch()
#         # # grid_irbfc.distnD3_batch()
#         # grid_irbfn.save(grid_irbfc_path, clf_path, irbfc_path)
# 
#     grid_nninv_tsne_path = base_dir + 'grid_nninv_tsne.joblib'
#     if not os.path.isfile(grid_nninv_tsne_path):
#         print("GRID NNInv TSNE")
#         save_grid(grid_nninv_tsne_path, R, N, clf, nninv_tsne, X_train,
#                   Xp_tsne, clf_path, nninv_tsne_path)
#         # grid_nn = Grid()
#         # grid_nn.fit(R, N, clf, nninv, X_nd=X_train, X_2d=X_proj)
#         # grid_nn.BoundaryMapBatch()
#         # grid_nn.dist2D()
#         # # grid_nn.distnD_batch()
#         # # grid_nn.distnD2_batch()
#         # # grid_nn.distnD3_batch()
#         # grid_nn.save(grid_nninv_path, clf_path, nninv_path)
# 
#     grid_ilamp_umap_path = base_dir + 'grid_ilamp_umap.joblib'
#     if not os.path.isfile(grid_ilamp_umap_path):
#         print("GRID ILAMP UMAP")
#         save_grid(grid_ilamp_umap_path, R, N, clf, ilamp_umap, X_train,
#                   Xp_umap, clf_path, ilamp_umap_path)
# 
#     grid_irbfcp_umap_path = base_dir + 'grid_irbfcp_umap.joblib'
#     if not os.path.isfile(grid_irbfcp_umap_path):
#         print("GRID RBF CTRL PTS UMAP")
#         save_grid(grid_irbfcp_umap_path, R, N, clf, irbfcp_umap, X_train,
#                   Xp_umap, clf_path, irbfcp_umap_path)
# 
#     grid_irbfc_umap_path = base_dir + 'grid_irbfc_umap.joblib'
#     if not os.path.isfile(grid_irbfc_umap_path):
#         print("GRID RBF CLUSTERS UMAP")
#         save_grid(grid_irbfc_umap_path, R, N, clf, irbfc_umap, X_train,
#                   Xp_umap, clf_path, irbfc_umap_path)
# 
#     grid_nninv_umap_path = base_dir + 'grid_nninv_umap.joblib'
#     if not os.path.isfile(grid_nninv_umap_path):
#         print("GRID NNInv UMAP")
#         save_grid(grid_nninv_umap_path, R, N, clf, nninv_umap, X_train,
#                   Xp_umap, clf_path, nninv_umap_path)
# 
# 
# if __name__ == "__main__":
#     main()
