#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import data
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from boundarymap import CLF, Grid
import dist_maps

data_path = '/home/fcmr/workspace/data/mnist/orig/'
img_size = 28
X, y = data.LoadMNISTData(dataset='train', path=data_path)

X_zeros = X[y == 0]
X_ones = X[y == 1]

# two samples dataset
plt.imshow(X_zeros[0], cmap='gray')
plt.show()
plt.imshow(X_ones[0], cmap='gray')
plt.show()
X_train = np.array([X_zeros[0], X_ones[0]])
X_train = X_train.reshape((X_train.shape[0], img_size**2))
y_train = [0, 1]

X_proj = np.array([[0.25, 0.5], [0.75, 0.5]])

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)
clf.predict(X_train)

dmap_clf = CLF(clf, "sklearn")
dmap_clf.Predict(X_train)

grid_size = 30
g = Grid(X_proj, grid_size)
_, dmap = g.BoundaryMap(X_train, 2, dmap_clf, k_ilamp=2)

dmap_rgb = hsv_to_rgb(dmap)
plt.imshow(dmap_rgb)
plt.show()

dist_2d = dist_maps.dist2d_grid(dmap)
dist_nd = dist_maps.distnd_grid(dmap, X_train, X_proj, k_ilamp=2)
plt.imshow(dist_2d, cmap='gray')
plt.show()
plt.imshow(dist_nd, cmap='gray')
plt.show()


# np.save('../clf-boundary-map-ui/data/dist_tests/X.npy', X_train)
# np.save('../clf-boundary-map-ui/data/dist_tests/y.npy', y_train)
# np.save('../clf-boundary-map-ui/data/dist_tests/y_pred.npy', y_train)
# np.save('../clf-boundary-map-ui/data/dist_tests/X_proj.npy', X_proj)
# np.save('../clf-boundary-map-ui/data/dist_tests/dmap.npy', dmap)


dist_nd_2 = dist_maps.distance_nd_2(X_train, X_proj, clf, grid_size, k_ilamp=2)
plt.imshow(dist_nd_2, cmap='gray')
