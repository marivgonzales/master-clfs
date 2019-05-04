from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from umap import UMAP
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np

from lamp import ILAMP
from nninv import NNInv

import joblib
import json
import time

from boundarymap import CLF
from boundarymap import Grid


def save_grid(path, R, N, clf, inv_proj, X_nd, X_2d, clf_path, inv_proj_path):
    grid = Grid()
    grid.fit(R, N, clf, inv_proj, X_nd=X_nd, X_2d=X_2d, syn_only=False)
    grid.BoundaryMapBatch(compute_hsv=True)
    # grid.dist2D()
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


base_dir = 'data/laps_10/'
train_y_path = base_dir + "y_train.npy"
X_train = np.load(base_dir + 'X_laps_train.npy')
y_train = np.load(train_y_path)

projection_size = X_train.shape[0]

X_proj = np.copy(X_train[:projection_size])
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

subset_size = 4659
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

# print("\n\nNNInv TSNE")
# s = time.time()
# nninv_tsne = NNInv()
# nninv_tsne.fit(X_proj[:subset_size], tsne_proj[:subset_size])
# nninv_tsne_path = base_dir + "nninv_tsne.joblib"
# nninv_tsne.save(nninv_tsne_path, base_dir + 'nninv_tsne_keras.hdf5')
# print("\ttime: ", time.time() - s)
# 
# print("\n\nNNInv UMAP")
# s = time.time()
# nninv_umap = NNInv()
# nninv_umap.fit(X_proj[:subset_size], umap_proj[:subset_size])
# nninv_umap_path = base_dir + "nninv_umap.joblib"
# nninv_umap.save(nninv_umap_path, base_dir + 'nninv_umap_keras.hdf5')
# print("\ttime: ", time.time() - s)

print("\n\nTRAINING CLASSIFIER")
s = time.time()
lr = LogisticRegression()
lr.fit(X_train, y_train)
print("train acc: ", lr.score(X_train, y_train))

y_proj_pred = lr.predict(X_proj)
pred_path = base_dir + "y_pred_clf.npy"
np.save(pred_path, y_proj_pred)

clf_sklearn_path = base_dir + "lr.joblib"
joblib.dump(lr, clf_sklearn_path)

clf = CLF(clf=lr, clf_type='sklearn', clf_path=clf_sklearn_path)
clf_path = base_dir + 'lr.json'
clf.save_json(clf_path)
print("\ttime: ", time.time() - s)

N = 1
R = 100

print("\n\nGRID ILAMP tSNE")
grid_ilamp_tsne_path = base_dir + 'grid_ilamp_tsne.joblib'
save_grid(grid_ilamp_tsne_path, R, N, clf, ilamp_tsne, X_train,
          tsne_proj, clf_path, ilamp_tsne_path)
ui_grid_ilamp_tsne_path = base_dir + 'laps_features_ilamp_tsne.json'
save_json_ui(ui_grid_ilamp_tsne_path, grid_ilamp_tsne_path, clf_path, "ilamp",
             ilamp_tsne_path, train_y_path, pred_path)

print("\n\nGRID ILAMP UMAP")
grid_ilamp_umap_path = base_dir + 'grid_ilamp_umap.joblib'
save_grid(grid_ilamp_umap_path, R, N, clf, ilamp_umap, X_train,
          umap_proj, clf_path, ilamp_umap_path)
ui_grid_ilamp_umap_path = base_dir + 'laps_features_ilamp_umap.json'
save_json_ui(ui_grid_ilamp_umap_path, grid_ilamp_umap_path, clf_path, "ilamp",
             ilamp_umap_path, train_y_path, pred_path)

# print("\n\nGRID NNInv tSNE")
# grid_nninv_tsne_path = base_dir + 'grid_nninv_tsne.joblib'
# save_grid(grid_nninv_tsne_path, R, N, clf, nninv_tsne, X_train,
#           tsne_proj, clf_path, nninv_tsne_path)
# ui_grid_nninv_tsne_path = base_dir + 'laps_features_nninv_tsne.json'
# save_json_ui(ui_grid_nninv_tsne_path, grid_nninv_tsne_path, clf_path, "nninv",
#              nninv_tsne_path, train_y_path, pred_path)
# 
# print("\n\nGRID NNInv UMAP")
# grid_nninv_umap_path = base_dir + 'grid_nninv_umap.joblib'
# save_grid(grid_nninv_umap_path, R, N, clf, nninv_umap, X_train,
#           umap_proj, clf_path, nninv_umap_path)
# ui_grid_nninv_umap_path = base_dir + 'laps_features_nninv_umap.json'
# save_json_ui(ui_grid_nninv_umap_path, grid_nninv_umap_path, clf_path, "nninv",
#              nninv_umap_path, train_y_path, pred_path)

