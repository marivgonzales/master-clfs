import numpy as np

import json
from boundarymap import CLF
from boundarymap import Grid
from boundarymap import PlotDenseMap
from boundarymap import PlotProjection

import time

from mnist import load_json_data

np.random.seed(0)

base_dir = "data/mnist/"
# Ideal path:
# 1 - Load dataset, projection and a trained classifier
X_train, projs, clfs = load_json_data(base_dir, "mnist.json")

proj_lamp = projs[0]
proj_tsne = projs[1]
X = X_train[:len(proj_tsne)]

total = 0.0

# print("Create densemap for clf: ", clfs[0].name)
# s = time.time()
# Run boundary map construction function
R = 100
N = 15
print("Creating grid")
s = time.time()
grid = Grid(X, proj_tsne, R)
dt = time.time() - s; total += dt; print("\ttime: ", dt)

print("\nComputing syn coords")
s = time.time()
grid.ComputeSynCoords(N)
dt = time.time() - s; total += dt; print("\ttime: ", dt)

print("\nNumber of synthetic coords: ", len(grid.syn_coords))

print("\nComputing syn samples")
s = time.time()
grid.ComputeSynSamples()
dt = time.time() - s; total += dt; print("\ttime: ", dt)

print("\nPredicting syn samples")
s = time.time()
grid.PredictSynSamples(clfs[0])
dt = time.time() - s; total += dt; print("\ttime: ", dt)

print("\nComputing color scheme")
s = time.time()
grid.ComputeColorScheme()
dt = time.time() - s; total += dt; print("\ttime: ", dt)

print("\n\nTotal time: ", total)


print("\n\nPrevious method: ")
s = time.time()
grid.BoundaryMap(N, clfs[0])
dt = time.time() - s; total += dt; print("\ttime: ", dt)
