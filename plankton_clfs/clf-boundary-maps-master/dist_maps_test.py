#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn import linear_model
from sklearn import manifold
from sklearn import preprocessing
import matplotlib.pyplot as plt

from boundarymap import CLF
from boundarymap import Grid

import dist_maps

import json

# Synthetic dataset suggestions:
# 1. 2 gaussian blobs
# 2. 3 gaussian blobs
# 3. 4 gaussian blobs
# 4. 8 class uniform random: generate 3d random uniformlt distributed points
#    in a cube. Each of the 8 quadrants is a class.
# 5. 2 class balls: create two balls with same center and different radius,
#    can be nD.
# 6. 

n_samples = 100
COLORS = ["#377eb8", "#ff7f00", '#4daf4a']
# noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=.05)
X_circles, y_circles = datasets.make_circles(n_samples=n_samples, factor=.5, noise=None)

num_train = int(X_circles.shape[0]*0.7)
X_circles_train = X_circles[:num_train]
y_circles_train = y_circles[:num_train]
X_circles_test = X_circles[num_train:]
y_circles_test = y_circles[num_train:]


clf_circles = svm.SVC()
clf_circles.fit(X_circles_train, y_circles_train)
print("train acc: ", clf_circles.score(X_circles_train, y_circles_train))
print("test acc: ", clf_circles.score(X_circles_test, y_circles_test))


colors = [COLORS[v] for v in y_circles_train]
plt.scatter(X_circles_train[:, 0], X_circles_train[:, 1], color=colors)

colors = [COLORS[v] for v in y_circles_test]
plt.scatter(X_circles_test[:, 0], X_circles_test[:, 1], color=colors)


X_moons, y_moons = datasets.make_moons(n_samples=n_samples, noise=None)

num_train = int(X_moons.shape[0]*0.7)
X_moons_train = X_moons[:num_train]
y_moons_train = y_moons[:num_train]
X_moons_test = X_moons[num_train:]
y_moons_test = y_moons[num_train:]

colors = [COLORS[v] for v in y_moons_train]
plt.scatter(X_moons_train[:, 0], X_moons_train[:, 1], color=colors)


X_blobs, y_blobs = datasets.make_blobs(n_samples=n_samples, centers=3)
num_train = int(X_blobs.shape[0]*0.7)
X_blobs_train = X_blobs[:num_train]
y_blobs_train = y_blobs[:num_train]
X_blobs_test = X_blobs[num_train:]
y_blobs_test = y_blobs[num_train:]

colors = [COLORS[v] for v in y_blobs_train]
plt.scatter(X_blobs_train[:, 0], X_blobs_train[:, 1], color=colors)


no_structure = np.random.rand(n_samples, 2), None


# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)


# generate dataset
X, y = datasets.make_classification(n_samples=n_samples, n_classes=2, 
                                   n_features=16, n_redundant=0, 
                                   n_informative=10, random_state=1, 
                                   n_clusters_per_class=1)

num_train = int(X.shape[0]*0.7)
X_train = X[:num_train]
y_train = y[:num_train]

X_test = X[num_train:]
y_test = y[num_train:]

# FIXME: normalize train data, apply normalization to test data
scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

clf_lr = linear_model.LogisticRegression()
clf_lr.fit(X_train, y_train)
print("train acc: ", clf_lr.score(X_train, y_train))
print("test acc: ", clf_lr.score(X_test, y_test))

tsne = manifold.TSNE(perplexity=10)
X_proj = tsne.fit_transform(X_train)
proj_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
X_proj = proj_scaler.fit_transform(X_proj)

colors = [COLORS[v] for v in y_train]
plt.scatter(X_proj[:, 0], X_proj[:, 1], color=colors)

clf = CLF(clf=clf_lr, clf_type="sklearn")
basedir = '/tmp/syn_test/'

R = 200
N = [5]
grid1 = Grid(X_proj, R)
_, dmap = grid1.BoundaryMap(X_train, N[0], clf)

H, W, _ = dmap.shape
GRID_SIZE = dmap.shape[0]


dist_nd = dist_maps.dist_nd(dmap, X_train, X_proj, clf=clf_lr)
dist_nd /= dist_nd.max()
dist_nd = 1.0 - dist_nd

dist_nd_2 = dist_maps.distance_nd_2(X_train, X_proj, clf_lr, GRID_SIZE)
dist_nd_2 /= dist_nd_2.max()
dist_nd_2 = 1.0 - dist_nd_2

np.save(basedir + 'X_train.npy', X_train)
np.save(basedir + 'y_train.npy', y_train)
np.save(basedir + 'y_pred.npy', y_train)
np.save(basedir + 'X_proj.npy', X_proj)
np.save(basedir + 'dmap_syn.npy', dmap)

dist_nd_path = basedir + "dist_nd.npy"
dist_nd_2_path = basedir + "dist_nd_2.npy"
np.save(dist_nd_path, dist_nd)
np.save(dist_nd_2_path, dist_nd_2)

json_data = {}
json_data['X'] = basedir + 'X_train.npy'
json_data['y'] = basedir + 'y_train.npy'
json_data['y_pred'] = basedir + 'y_pred.npy'
json_data['proj'] = basedir + 'X_proj.npy'
json_data['dense_map'] = basedir + 'dmap_syn.npy'
json_data['dist_map'] = basedir + 'dist_nd_2.npy'

with open(basedir + 'data.json', 'w') as fp:
    json.dump(json_data, fp)


import time
import dist_maps

s = time.time()
dist_nd_old = dist_maps.dist_nd(dmap, X_train, X_proj, clf=clf_lr)
print("time: ", time.time() - s)

s = time.time()
dist_nd_new = dist_maps.dist_nd_new(dmap, X_train, X_proj, clf=clf_lr)
print("time: ", time.time() - s)

plt.imshow(hsv_to_rgb(dmap))

s = time.time()
on_db = dist_maps.boundary_cells_new2(dmap)
edt, inds = ndimage.distance_transform_edt(on_db, return_indices=True)
db_map= np.dstack((inds[0], inds[1]))
print("time: ", time.time() - s)

s = time.time()
_, db_map_mine = dist_maps.boundary_cells_new(dmap)
print("time: ", time.time() - s)

#s = time.time()
#on_boundary, dbmap = dist_maps.boundary_cells_new(dmap)
#print("time: ", time.time() - s)
