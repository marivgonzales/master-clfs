import numpy as np
from boundarymap import CLF
from boundarymap import Grid
import json
import time
import os.path
import joblib
from nninv import NNInv

import keras

np.random.seed(0)
keras.backend.set_learning_phase(0)


def main():
    base_dir = "data/cifar10/"

    with open(base_dir + "cifar.json") as f:
        data_json_base = json.load(f)

    X_train = np.load(data_json_base['X_train'])
    input_shape = (X_train.shape[1], X_train.shape[2], 3)

    clf = CLF()
    clf_path = data_json_base['clfs'][0]
    clf.LoadKerasModel(clf_path, "CNN", input_shape)

    inv_projs_path = data_json_base['inv_projs']
    ilamp_path = inv_projs_path[0]
    irbfcp_path = inv_projs_path[1]
    irbfn_path = inv_projs_path[2]
    irbfc_path = inv_projs_path[3]
    nninv_path = inv_projs_path[4]

    inv_projs = []
    for inv in inv_projs_path[:-1]:
        inv_projs.append(joblib.load(inv))

    ilamp = inv_projs[0]
    irbf_cp = inv_projs[1]
    irbf_neighbors = inv_projs[2]
    irbf_cluster = inv_projs[3]
    nninv = NNInv()
    nninv.load(nninv_path)

    N = 1
    R = 250

    grid_ilamp_path = base_dir + 'grid_ilamp.joblib'
    if not os.path.isfile(grid_ilamp_path):
        print("CIFAR ILAMP GRID")
        s = time.time()
        grid_ilamp = Grid()
        grid_ilamp.fit(R, N, clf, ilamp)
        grid_ilamp.BoundaryMapBatch()
        grid_ilamp.distnD3_batch()
        grid_ilamp.dist2D()
        grid_ilamp.distnD_batch()
        grid_ilamp.distnD2_batch()
        grid_ilamp.save(grid_ilamp_path, clf_path, ilamp_path)
        print("\ttime: ", time.time() - s)

    grid_irbfcp_path = base_dir + 'grid_irbfcp.joblib'
    if not os.path.isfile(grid_irbfcp_path):
        print("CIFAR RBF CTRL PTS GRID")
        s = time.time()
        grid_irbfcp = Grid()
        grid_irbfcp.fit(R, N, clf, irbf_cp)
        grid_irbfcp.BoundaryMapBatch()
        grid_irbfcp.dist2D()
        grid_irbfcp.distnD_batch()
        grid_irbfcp.distnD2_batch()
        grid_irbfcp.distnD3_batch()
        grid_irbfcp.save(grid_irbfcp_path, clf_path, irbfcp_path)
        print("\ttime: ", time.time() - s)

    grid_irbfn_path = base_dir + 'grid_irbfn.joblib'
    if not os.path.isfile(grid_irbfn_path):
        print("CIFAR RBF NEIGHBORS GRID")
        s = time.time()
        grid_irbfn = Grid()
        grid_irbfn.fit(R, N, clf, irbf_neighbors)
        grid_irbfn.BoundaryMapBatch()
        grid_irbfn.dist2D()
        grid_irbfn.distnD_batch()
        grid_irbfn.distnD2_batch()
        grid_irbfn.distnD3_batch()
        grid_irbfn.save(grid_irbfn_path, clf_path, irbfn_path)
        print("\ttime: ", time.time() - s)

    grid_irbfc_path = base_dir + 'grid_irbfc.joblib'
    if not os.path.isfile(grid_irbfc_path):
        print("CIFAR RBF CLUSTER GRID")
        s = time.time()
        grid_irbfc = Grid()
        grid_irbfc.fit(R, N, clf, irbf_cluster)
        grid_irbfc.BoundaryMapBatch()
        grid_irbfc.dist2D()
        grid_irbfc.distnD_batch()
        grid_irbfc.distnD2_batch()
        grid_irbfc.distnD3_batch()
        grid_irbfn.save(grid_irbfc_path, clf_path, irbfc_path)
        print("\ttime: ", time.time() - s)

    grid_nninv_path = base_dir + 'grid_nninv.joblib'
    if not os.path.isfile(grid_nninv_path):
        print("CIFAR NNInv GRID")
        s = time.time()
        grid_nn = Grid()
        grid_nn.fit(R, N, clf, nninv)
        grid_nn.BoundaryMapBatch()
        grid_nn.dist2D()
        grid_nn.distnD_batch()
        grid_nn.distnD2_batch()
        grid_nn.distnD3_batch()
        grid_nn.save(grid_nninv_path, clf_path, nninv_path)
        print("\ttime: ", time.time() - s)


if __name__ == "__main__":
    main()
