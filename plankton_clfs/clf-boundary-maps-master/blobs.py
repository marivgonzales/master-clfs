from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from umap import UMAP
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
import os
import sys

from lamp import ILAMP
from lamp import RBFInv
from nninv import NNInv

import joblib
import json

from boundarymap import CLF
from boundarymap import Grid


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


if len(sys.argv) < 2:
    print("Usage: ./fashion_mnist.py <base_dir>")
    sys.exit(0)
base_dir = sys.argv[1]

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

print("Creating dataset")
X_blobs, y_blobs = datasets.make_blobs(n_samples=60000, n_features=50,
                                       centers=5, cluster_std=0.1,
                                       random_state=420)
scaler = MinMaxScaler()
X_blobs = scaler.fit_transform(X_blobs)

# train_X_path = base_dir + "X_train.npy"
# np.save(train_X_path, X_blobs)
train_y_path = base_dir + "y_train.npy"
np.save(train_y_path, y_blobs)

X_proj = np.copy(X_blobs)

print("\n\ntSNE projection")
tsne = TSNE(n_components=2, random_state=420, perplexity=10.0,
            n_iter=1000, n_iter_without_progress=300, n_jobs=4)
tsne_proj = tsne.fit_transform(X_proj)
tsne_proj = scaler.fit_transform(tsne_proj)
# tsne_proj_path = base_dir + "tsne_proj.npy"
# np.save(tsne_proj_path, tsne_proj)

print("\n\nUMAP projection")
umap_proj = UMAP(n_components=2, random_state=420, n_neighbors=10,
                 min_dist=0.001).fit_transform(X_proj)
umap_proj = scaler.fit_transform(umap_proj)
# umap_proj_path = base_dir + "umap_proj.npy"

subset_size = 15000
print("\n\nILAMP tSNE")
k_ilamp = 20
ilamp_tsne = ILAMP(n_neighbors=k_ilamp)
ilamp_tsne.fit(X_proj[:subset_size], tsne_proj[:subset_size])
ilamp_tsne_path = base_dir + "ilamp_tsne.joblib"
ilamp_tsne.save(ilamp_tsne_path)

print("\n\nILAMP UMAP")
ilamp_umap = ILAMP(n_neighbors=k_ilamp)
ilamp_umap.fit(X_proj[:subset_size], umap_proj[:subset_size])
ilamp_umap_path = base_dir + "ilamp_umap.joblib"
ilamp_umap.save(ilamp_umap_path)

print("\n\nRBFInv CTRL PTS TSNE")
EPS = 50000
irbfcp_tsne = RBFInv(num_ctrl=200, mode='rols', kernel='gaussian',
                     eps=EPS, normalize_c=True, normalize_d=True)
irbfcp_tsne.fit(X_proj[:subset_size], tsne_proj[:subset_size])
irbfcp_tsne_path = base_dir + "irbfcp_tsne.joblib"
irbfcp_tsne.save(irbfcp_tsne_path)

print("\n\nRBFInv CTRL PTS UMAP")
EPS = 50000
irbfcp_umap = RBFInv(num_ctrl=200, mode='rols', kernel='gaussian',
                     eps=EPS, normalize_c=True, normalize_d=True)
irbfcp_umap.fit(X_proj[:subset_size], umap_proj[:subset_size])
irbfcp_umap_path = base_dir + "irbfcp_umap.joblib"
irbfcp_umap.save(irbfcp_umap_path)

print("\n\nRBFInv CLUSTER TSNE")
EPS = 50000
irbfc_tsne = RBFInv(num_ctrl=50, mode='cluster', kernel='gaussian',
                     eps=EPS, normalize_c=True, normalize_d=True)
irbfc_tsne.fit(X_proj[:subset_size], tsne_proj[:subset_size])
irbfc_tsne_path = base_dir + "irbfc_tsne.joblib"
irbfc_tsne.save(irbfc_tsne_path)

print("\n\nRBFInv CLUSTER UMAP")
EPS = 50000
irbfc_umap = RBFInv(num_ctrl=50, mode='cluster', kernel='gaussian',
                     eps=EPS, normalize_c=True, normalize_d=True)
irbfc_umap.fit(X_proj[:subset_size], umap_proj[:subset_size])
irbfc_umap_path = base_dir + "irbfc_umap.joblib"
irbfc_umap.save(irbfc_umap_path)


print("\n\nNNInv TSNE")
nninv_tsne = NNInv()
nninv_tsne.fit(X_proj[:subset_size], tsne_proj[:subset_size])
nninv_tsne_path = base_dir + "nninv_tsne.joblib"
nninv_tsne.save(nninv_tsne_path, base_dir + 'nninv_tsne_keras.hdf5')

print("\n\nNNInv UMAP")
nninv_umap = NNInv()
nninv_umap.fit(X_proj[:subset_size], umap_proj[:subset_size])
nninv_umap_path = base_dir + "nninv_umap.joblib"
nninv_umap.save(nninv_umap_path, base_dir + 'nninv_umap_keras.hdf5')

print("\n\nTraining LR classifier")
lr = linear_model.LogisticRegression()
lr.fit(X_blobs, y_blobs)
y_proj_pred = lr.predict(X_proj)
pred_path = base_dir + "y_pred_clf.npy"
np.save(pred_path, y_proj_pred)

clf_sklearn_path = base_dir + "lr.joblib"
joblib.dump(lr, clf_sklearn_path)

clf = CLF(clf=lr, clf_type='sklearn', clf_path=clf_sklearn_path)
clf_path = base_dir + 'lr.json'
clf.save_json(clf_path)

N = 1
R = 500

print("\n\nGRID ILAMP tSNE")
grid_ilamp_tsne_path = base_dir + 'grid_ilamp_tsne.joblib'
save_grid(grid_ilamp_tsne_path, R, N, clf, ilamp_tsne, X_blobs,
          tsne_proj, clf_path, ilamp_tsne_path)
ui_grid_ilamp_tsne_path = base_dir + 'blobs500_ui_ilamp_tsne.json'
save_json_ui(ui_grid_ilamp_tsne_path, grid_ilamp_tsne_path, clf_path, "ilamp",
             ilamp_tsne_path, train_y_path, pred_path)

print("\n\nGRID ILAMP UMAP")
grid_ilamp_umap_path = base_dir + 'grid_ilamp_umap.joblib'
save_grid(grid_ilamp_umap_path, R, N, clf, ilamp_umap, X_blobs,
          umap_proj, clf_path, ilamp_umap_path)
ui_grid_ilamp_umap_path = base_dir + 'blobs500_ui_ilamp_umap.json'
save_json_ui(ui_grid_ilamp_umap_path, grid_ilamp_umap_path, clf_path, "ilamp",
             ilamp_umap_path, train_y_path, pred_path)

print("\n\nGRID RBFInv CTRL PTS tSNE")
grid_irbfcp_tsne_path = base_dir + 'grid_irbfcp_tsne.joblib'
save_grid(grid_irbfcp_tsne_path, R, N, clf, irbfcp_tsne, X_blobs,
          tsne_proj, clf_path, irbfcp_tsne_path)
ui_grid_irbfcp_tsne_path = base_dir + 'blobs500_ui_irbfcp_tsne.json'
save_json_ui(ui_grid_irbfcp_tsne_path, grid_irbfcp_tsne_path, clf_path, "rbf",
             irbfcp_tsne_path, train_y_path, pred_path)

print("\n\nGRID RBFInv CTRL PTS UMAP")
grid_irbfcp_umap_path = base_dir + 'grid_irbfcp_umap.joblib'
save_grid(grid_irbfcp_umap_path, R, N, clf, irbfcp_umap, X_blobs,
          umap_proj, clf_path, irbfcp_umap_path)
ui_grid_irbfcp_umap_path = base_dir + 'blobs500_ui_irbfcp_umap.json'
save_json_ui(ui_grid_irbfcp_umap_path, grid_irbfcp_umap_path, clf_path, "rbf",
             irbfcp_umap_path, train_y_path, pred_path)

print("\n\nGRID RBFInv CLUSTER tSNE")
grid_irbfc_tsne_path = base_dir + 'grid_irbfc_tsne.joblib'
save_grid(grid_irbfc_tsne_path, R, N, clf, irbfc_tsne, X_blobs,
          tsne_proj, clf_path, irbfc_tsne_path)
ui_grid_irbfc_tsne_path = base_dir + 'blobs500_ui_irbfc_tsne.json'
save_json_ui(ui_grid_irbfc_tsne_path, grid_irbfc_tsne_path, clf_path, "rbf",
             irbfc_tsne_path, train_y_path, pred_path)

print("\n\nGRID RBFInv CLUSTER UMAP")
grid_irbfc_umap_path = base_dir + 'grid_irbfc_umap.joblib'
save_grid(grid_irbfc_umap_path, R, N, clf, irbfc_umap, X_blobs,
          umap_proj, clf_path, irbfc_umap_path)
ui_grid_irbfc_umap_path = base_dir + 'blobs500_ui_irbfc_umap.json'
save_json_ui(ui_grid_irbfc_umap_path, grid_irbfc_umap_path, clf_path, "rbf",
             irbfc_umap_path, train_y_path, pred_path)

print("\n\nGRID NNInv tSNE")
grid_nninv_tsne_path = base_dir + 'grid_nninv_tsne.joblib'
save_grid(grid_nninv_tsne_path, R, N, clf, nninv_tsne, X_blobs,
          tsne_proj, clf_path, nninv_tsne_path)
ui_grid_nninv_tsne_path = base_dir + 'blobs500_ui_nninv_tsne.json'
save_json_ui(ui_grid_nninv_tsne_path, grid_nninv_tsne_path, clf_path, "nninv",
             nninv_tsne_path, train_y_path, pred_path)

print("\n\nGRID NNInv UMAP")
grid_nninv_umap_path = base_dir + 'grid_nninv_umap.joblib'
save_grid(grid_nninv_umap_path, R, N, clf, nninv_umap, X_blobs,
          umap_proj, clf_path, nninv_umap_path)
ui_grid_nninv_umap_path = base_dir + 'blobs500_ui_nninv_umap.json'
save_json_ui(ui_grid_nninv_umap_path, grid_nninv_umap_path, clf_path, "nninv",
             nninv_umap_path, train_y_path, pred_path)

