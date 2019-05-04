#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from lamp import ILAMP
# from sklearn.neighbors import KDTree
# from scipy.spatial import distance_matrix
from scipy.spatial import distance
from scipy import ndimage

import annoy
import foolbox


def hue_cmp(h1, h2):
    CMAP_ORIG = np.array([234, 0, 108, 288, 252, 72, 180, 324, 36, 144])/360.0
    CMAP_SYN = np.array([216, 18, 126, 306, 270, 90, 198, 342, 54, 162])/360.0

    # print('h1, h2: ', h1, h2)
    if h1 in CMAP_ORIG:
        idx = np.where(CMAP_ORIG == h1)[0][0]
        alt_h1 = CMAP_SYN[idx]
    else:
        # h1 must be in CMAP_SYN
        idx = np.where(CMAP_SYN == h1)[0][0]
        alt_h1 = CMAP_ORIG[idx]
    if h1 == h2 or alt_h1 == h2:
        return 0
    return 1


# get 4-neighborhood, n8 = True for 8-neighborhood
def get_neighbors(row, col, h, w, n8=False):
        neighbors = []
        if row - 1 >= 0:
            neighbors.append([row - 1, col])

        if col - 1 >= 0:
            neighbors.append([row, col - 1])

        if row + 1 < h:
            neighbors.append([row + 1, col])

        if col + 1 < w:
            neighbors.append([row, col + 1])

        if n8 is True:
            if row - 1 >= 0 and col - 1 >= 0:
                neighbors.append([row - 1, col - 1])
            if row + 1 < h and col - 1 >= 0:
                neighbors.append([row + 1, col - 1])
            if row + 1 < h and col + 1 < w:
                neighbors.append([row + 1, col + 1])
            if row - 1 >= 0 and col + 1 < w:
                neighbors.append([row - 1, col + 1])

        return neighbors


# Creates a grid of cells of (grid_size x grid_size).
# Each cell holds the index of each point in proj that maps to it.
def build_grid(proj, grid_size):
    cells = [[] for i in range(grid_size)]

    for i in range(grid_size):
        cells[i] = [[] for _ in range(grid_size)]

    tile_size = 1.0/grid_size
    # Adds point's indices to the corresponding cell
    for idx in range(len(proj)):
        p = proj[idx]
        row = int(abs(p[1] - 1e-5)/tile_size)
        col = int(abs(p[0] - 1e-5)/tile_size)
        cells[row][col].append(idx)
    return cells


# TODO - Idea for faster boundary_cells_new:
# Create 4 versions of shifted dmap_hsv:
# 1. Shift rows by 1
# 2. Shift rows by -1
# 3. Shift cols by 1
# 4. Shift cols by -1
# Subtract dmap by each shifted dmap.
# If value is 0, then hues match
# Else, check for possible other hue (syn/orig hues)
def boundary_cells(dmap_hsv):
    H, W, _ = dmap_hsv.shape
    on_boundary = np.full((H, W), 1.0)
    for row in range(H):
        for col in range(W):
            neighbors = get_neighbors(row, col, H, W, n8=True)
            cell_hue = dmap_hsv[row, col, 0]

            for n in neighbors:
                neighbor_hue = dmap_hsv[n[0], n[1], 0]
                if hue_cmp(cell_hue, neighbor_hue) == 1:
                    on_boundary[row, col] = 0.0
                    on_boundary[n[0], n[1]] = 0.0
    return on_boundary


def dist2d(dmap):
    return ndimage.distance_transform_edt(boundary_cells(dmap))


def refine_boundaries(dmap, db_map):
    H, W, _ = dmap.shape
    for row in range(H):
        for col in range(W):
            cell_hue = dmap[row, col, 0]
            db = db_map[row, col]
            other_hue = dmap[db[0], db[1], 0]

            if hue_cmp(cell_hue, other_hue) != 0:
                continue

            neighbors = get_neighbors(db[0], db[1], H, W, n8=True)
            for n in neighbors:
                n_hue = dmap[n[0], n[1], 0]
                if hue_cmp(cell_hue, n_hue) != 1:
                    continue
                db_map[row, col] = n
                break


def distnd_batch(dmap, X, proj, inv_proj, clf=None): 
    grid_size = dmap.shape[0]
    H, W = grid_size, grid_size

    import time
    print("computing boundary cells")
    s = time.time()
    on_db = boundary_cells(dmap)
    print("\ttime: ", time.time() - s)
    print("computing distance transform")
    s = time.time()
    _, inds = ndimage.distance_transform_edt(on_db, return_indices=True)
    db_map = np.dstack((inds[0], inds[1]))
    print("\ttime: ", time.time() - s)



# Tries to estimate the distance from the samples mapped to a cell in 2D
# to the closest decision boudary in nD. It does so by finding the closest
# decision boundary
def distnd(dmap, X, proj, clf=None, inv_proj=None):
    if inv_proj is None:
        inv_proj = ILAMP()
        inv_proj.fit(X, proj)

    grid_size = dmap.shape[0]
    H, W = grid_size, grid_size

    import time
    print("computing boundary cells")
    s = time.time()
    on_db = boundary_cells(dmap)
    print("\ttime: ", time.time() - s)
    print("computing distance transform")
    s = time.time()
    _, inds = ndimage.distance_transform_edt(on_db, return_indices=True)
    db_map = np.dstack((inds[0], inds[1]))
    print("\ttime: ", time.time() - s)

    # Refine db_map: by the way the boundary map is constructed, there is no
    # guarantee that a cell and its closet boundary as stored in on_db will
    # have different labels.
    # The following lines adjust that.
    print("refining boundaries")
    s = time.time()
    refine_boundaries(dmap, db_map)
    # for row in range(H):
    #     for col in range(W):
    #         cell_hue = dmap[row, col, 0]
    #         db = db_map[row, col]
    #         other_hue = dmap[db[0], db[1], 0]
    #         if hue_cmp(cell_hue, other_hue) != 0:
    #             continue
    #         neighbors = get_neighbors(db[0], db[1], H, W, n8=True)
    #         for n in neighbors:
    #             n_hue = dmap[n[0], n[1], 0]
    #             if hue_cmp(cell_hue, n_hue) == 1:
    #                 db_map[row, col] = n
    #                 break
    print("\ttime: ", time.time() - s)

    cells = build_grid(proj, grid_size)
    dist_nd = np.zeros((grid_size, grid_size))
    print("computing nd distances")
    s = time.time()
    for row in range(H):
        # print("[dist_nd] row: ", row)
        for col in range(W):
            db_r, db_c = db_map[row, col, 0], db_map[row, col, 1]
            dist_nd[row, col] = distance_nd(row, col, db_r, db_c, grid_size,
                                            X, cells, proj, inv_proj=inv_proj,
                                            clf=clf)
    print("\ttime: ", time.time() - s)
    return dist_nd


# Returns an estimate distance in nD from cell [src_r, src_c] to cell
# [dst_r, dst_c]. This estimate is computed based on the nD samples that map
# to both cells, thus synthetic samples might be used.
def distance_nd(src_r, src_c, dst_r, dst_c, grid_size, X, cells, X_proj,
                inv_proj, clf=None):
    coords = np.array([(src_c + 0.5)/grid_size, (src_r + 0.5)/grid_size])
    db_coords = np.array([dst_c/grid_size, dst_r/grid_size])
    # get a sample from row, col cell
    # - if a orig one is available, use orig, else use iLAMP's
    samples = X[cells[src_r][src_c]]
    if len(samples) == 0:
        # samples = np.array([lamp.ilamp(X, X_proj, coords, k=k_ilamp)])
        samples = inv_proj.transform([coords], normalize=True)

    # get sample from src_r, src_c
    # - if there are no original samples in this cells, create a new one
    # using iLAMP
    samples_db = X[cells[dst_r][dst_c]]
    if len(samples_db) == 0:
        # samples_db = np.array([lamp.ilamp(X, X_proj, db_coords, k=k_ilamp)])
        samples_db = inv_proj.transform([db_coords], normalize=True)

    # compute dist_nD
    dist = 0
    for i in range(len(samples)):
        for j in range(len(samples_db)):
            if clf is None:
                dist += np.linalg.norm(samples[i] - samples_db[j])
            else:
                dist += dist_nd_bisection(samples[i], samples_db[j], clf)

    dist /= len(samples)*len(samples_db)
    return dist


# try to estimate the distance from a to a decision boundary given b and clf
# b: closest data point to a such that clf(a) != clf(b)
# clf: classifier trained on this dataset and used to create the densemap
# FIXME: use  CLF wrapper from boundarymap.py
def dist_nd_bisection(a, b, clf):
    tol = 1e-3
    max_iter = 5
    init = a
    end = b

    label_a = clf.Predict(np.array([a]))[0]
    for i in range(max_iter):
        new_pt = (init + end)*0.5
        new_pt_label = clf.Predict(np.array([new_pt]))[0]

        if new_pt_label != label_a:
            end = new_pt
        else:
            init = new_pt
        dist = np.linalg.norm(end - init)
        if dist <= tol:
            break
    # At this point, label(end) != label(a). end is used as a proxy to measure
    # distance to the decision boundary.
    return np.linalg.norm(a - end)


# Estimates the distance every cell is from a real decision boundary in nD.
# To accomplish this, both real and synthetic samples used to build a dense
# map are considered. For each cell
#   1. get all samples S1 that map to this cell
#   2. get all samples S2 with different labels than the ones in S1
#   3. compute the distance between S1 and S2
# Assumes that X_proj is normalized to [0, 1] range
def distnd2(X, X_proj, clf, grid_size, inv_proj=None):
    if inv_proj is None:
        inv_proj = ILAMP()
        inv_proj.fit(X, X_proj)

    cells_orig = build_grid(X_proj, grid_size)

    num_features = X.shape[1]
    ilamp_samples = []
    ilamp_proj = []

    import time
    print("generating ilamp samples")
    s = time.time()
    for row in range(grid_size):
        for col in range(grid_size):
            if len(cells_orig[row][col]) == 0:
                coords = np.array([(col + 0.5)/grid_size, (row + 0.5)/grid_size])
                sample = inv_proj.transform([coords], normalize=True)[0]
                ilamp_samples.append(sample)
                ilamp_proj.append(coords)
    print("\ttime: ", time.time() - s)

    ilamp_samples = np.array(ilamp_samples)
    ilamp_proj = np.array(ilamp_proj)

    num_syn = ilamp_samples.shape[0]
    num_orig = X.shape[0]
    num_total = num_orig + num_syn
    X_all = np.zeros((num_total, num_features))
    X_all[:num_orig] = X
    X_all[num_orig:] = ilamp_samples

    X_proj_all = np.zeros((num_total, 2))
    X_proj_all[:num_orig] = X_proj
    X_proj_all[num_orig:] = ilamp_proj
    print("predicting all samples")
    s = time.time()
    y_all = clf.Predict(X_all)
    print("\ttime: ", time.time() - s)
    # print("computing distances nd")

    dist_nd_2 = np.zeros((grid_size, grid_size))
    cells = build_grid(X_proj_all, grid_size)

    print("constructing annoy structure")
    s = time.time()
    t = annoy.AnnoyIndex(num_features)
    for i in range(num_total):
        t.add_item(i, X_all[i])
    t.build(num_total//1000)
    print("\ttime: ", time.time() - s)

    # print("constructing kdtree")
    # s = time.time()
    # tree = KDTree(X_all, leaf_size=100, metric='euclidean')
    # print("\ttime: ", time.time() - s)

    # distances_all = distance.cdist(X_all, X_all)
    print("computing distances between nd samples")
    s = time.time()
    for row in range(grid_size):
        # print("[distance_nd_2] row: ", row)
        for col in range(grid_size):
            # print("row, col: ", row,  col)
            # s0 = time.time()
            sample_idx = cells[row][col][0]
            sample = X_all[cells[row][col][0]]
            # label_sample = clf.Predict(np.array([sample]))[0]
            label_sample = y_all[sample_idx]

            # distances_sample = distance.cdist([sample], X_all)[0]
            # sorted_idx = np.argsort(distances_sample)
            # found = False
            # for idx in sorted_idx:
            #     if label_sample != y_all[idx]:
            #         dist_nd_2[row, col] = dist_nd_bisection(sample, X_all[idx], clf)
            #         found = True
            #         break
            # if found is False:
            #     print("error on pixel ", row, col)

            # print("row, col: ", row,  col)
            # s0 = time.time()

            # FIXME: this query makes no sense: all nodes are returned.
            # Should instead take all nodes that label is different and sort
            # them by distance.
            # X_diff = X_all[y_all != label_sample]
            # distances = distance_matrix([sample], X_diff).ravel()
            # distances = distance.cdist([sample], X_diff)[0]

            # idx = np.argmin(distances)
            # FIXME: this matrix can be computed once: distance betweeen
            # all the points and the select only the lines and columns with
            # different labels
            # dist_nd_2[row, col] = dist_nd_bisection(sample, X_diff[idx], clf)

            # dist, ind = tree.query([sample], k=num_total//3)
            # found = 0
            # # print("\tlooking for samples with different label")
            # # s = time.time()
            # for i in range(len(ind[0])):
            #     idx = ind[0][i]
            #     label_idx = clf.Predict(np.array([X_all[idx]]))[0]
            #     if label_idx != label_sample:
            #         dist_nd_2[row, col] = dist_nd_bisection(sample, X_all[idx],
            #                                                 clf)
            #         found += 1
            #         break
            # if found == 0:
            #     print("error on pixel ", row, col)
            # print("\ttime: ", time.time() - s0)

            found = False
            num_n_prev = 0
            target_idx = -1
            while found is False:
                num_n = num_n_prev + 1000
                nns = t.get_nns_by_item(sample_idx, num_n)
                for nn in nns[num_n_prev:num_n]:
                    if y_all[nn] != label_sample:
                        found = True
                        target_idx = nn
                        break
                num_n_prev = num_n

            if target_idx == -1:
                print("error on pixel ", row, col)
            dist_nd_2[row, col] = dist_nd_bisection(sample, X_all[target_idx], clf)
            # print("\ttime: ", time.time() - s0)

    print("\ttime: ", time.time() - s)
    return dist_nd_2


def get_neighbors_rb(row, col, grid_size):
    neighbors = []

    if row + 1 < grid_size:
        neighbors.append([row + 1, col])
    if col + 1 < grid_size:
        neighbors.append([row, col + 1])
    if row + 1 < grid_size and col + 1 < grid_size:
        neighbors.append([row + 1, col + 1])
    return neighbors, [1.0, 1.0, np.sqrt(2.0)]


def get_neighbors_la(row, col, grid_size):
    neighbors = []
    if row - 1 >= 0:
        neighbors.append([row - 1, col])
    if col - 1 >= 0:
        neighbors.append([row, col - 1])
    if row - 1 >= 0 and col - 1 >= 0:
        neighbors.append([row - 1, col - 1])
    return neighbors, [1.0, 1.0, np.sqrt(2.0)]


def boundary_cells_new(dmap_hsv):
    H, W, _ = dmap_hsv.shape

    on_boundary = np.full((H, W), np.inf)
    boundary_map = np.full((H, W, 2), -1, dtype=np.int)
    
    for row in range(H):
        for col in range(W):
            neighbors = get_neighbors(row, col, H, n8=False)
            cell_hue = dmap_hsv[row, col, 0]

            for n in neighbors:
                neighbor_hue = dmap_hsv[n[0], n[1], 0]
                if hue_cmp(cell_hue, neighbor_hue) == 1:
                    on_boundary[row, col] = 0
                    on_boundary[n[0], n[1]] = 0
                    boundary_map[row, col] = np.array(n)
                    boundary_map[n[0], n[1]] = np.array([row, col])

    # forward pass
    for row in range(H):
        for col in range(W):
            n_la, values = get_neighbors_la(row, col, H)
            for n, v in zip(n_la, values):
                nv = on_boundary[n[0], n[1]] + v
                if nv < on_boundary[row, col]:
                    on_boundary[row, col] = nv
                    boundary_map[row, col] = boundary_map[n[0], n[1]]
    
    # backward pass
    for row in range(H):
        for col in range(W):
            n_rb, values = get_neighbors_rb(row, col, H)
            for n, v in zip(n_rb, values):
                nv = on_boundary[n[0], n[1]] + v
                if nv < on_boundary[row, col]:
                    on_boundary[row, col] = nv
                    boundary_map[row, col] = boundary_map[n[0], n[1]]
    
    return on_boundary, boundary_map


def stress_func(dist_2d, dist_nd):
    H, W = dist_2d.shape
    dist_2d_norm = dist_2d/dist_2d.max()
    dist_nd_norm = dist_nd/dist_nd.max()
    sum_2d_min_nd = 0.0
    sum_nd = 0.0
    for i in range(H):
        for j in range(W):
            sum_2d_min_nd += (dist_2d_norm[i, j] - dist_nd_norm[i, j])**2
            sum_nd += dist_nd_norm[i, j]**2
    
    return sum_2d_min_nd/sum_nd


def closest_diff_label(X_all, y_all, sample_idx, sample_label, t):
    found = False
    num_n_prev = 0
    target_idx = -1
    while found is False:
        num_n = num_n_prev + 1000
        nns = t.get_nns_by_item(sample_idx, num_n)
        for nn in nns[num_n_prev:num_n]:
            if y_all[nn] != sample_label:
                found = True
                target_idx = nn
                break
        num_n_prev = num_n
    return target_idx


def distnd_adv(X, X_proj, clf, grid_size, inv_proj=None):
    if inv_proj is None:
        inv_proj = ILAMP()
        inv_proj.fit(X, X_proj)

    cells_orig = build_grid(X_proj, grid_size)

    num_features = X.shape[1]
    # list of samples generated by inverse projection
    invproj_samples = []
    # 2D points used to create back projection
    syn_proj = []

    import time
    print("generating inverse projection samples")
    s = time.time()
    for row in range(grid_size):
        for col in range(grid_size):
            if len(cells_orig[row][col]) > 0:
                continue

            coords = np.array([(col + 0.5)/grid_size, (row + 0.5)/grid_size])
            sample = inv_proj.transform([coords], normalize=True)[0]
            invproj_samples.append(sample)
            syn_proj.append(coords)
    print("\ttime: ", time.time() - s)

    invproj_samples = np.array(invproj_samples)
    syn_proj = np.array(syn_proj)

    num_syn = invproj_samples.shape[0]
    num_orig = X.shape[0]
    num_total = num_orig + num_syn
    X_all = np.zeros((num_total, num_features))
    X_all[:num_orig] = X
    X_all[num_orig:] = invproj_samples

    X_proj_all = np.zeros((num_total, 2))
    X_proj_all[:num_orig] = X_proj
    X_proj_all[num_orig:] = syn_proj
    cells = build_grid(X_proj_all, grid_size)

    print("predicting all samples")
    s = time.time()
    y_all = clf.Predict(X_all)
    print("\ttime: ", time.time() - s)

    # foolbox model
    # TODO: compute bounds from X_min and X_max
    # TODO: make CLF class compute the adversarial model
    model = foolbox.models.KerasModel(clf.clf, bounds=(0.0, 1.0))
    attack = foolbox.attacks.FGSM(model)
    # attack_fallback = foolbox.attacks.BoundaryAttack(model)

    print("constructing annoy structure")
    s = time.time()
    t = annoy.AnnoyIndex(num_features)
    for i in range(num_total):
        t.add_item(i, X_all[i])
    t.build(num_total//1000)
    print("\ttime: ", time.time() - s)

    dist_nd_adv = np.zeros((grid_size, grid_size))

    print("computing distance to boundary nd by adversarial examples")
    s = time.time()
    for row in range(grid_size):
        for col in range(grid_size):
            # print("row, col: ", row,  col)
            # s0 = time.time()
            sample_idx = cells[row][col][0]
            sample = X_all[sample_idx]
            # label_sample = clf.Predict(np.array([sample]))[0]
            sample_label = y_all[sample_idx]

            sample = sample.reshape(clf.shape)
            adversarial = attack(sample, sample_label)
            if adversarial is None:
                print("adversarial is None: ", row, col, "bisection")
                # adversarial = attack_fallback(sample, sample_label)
                # if adversarial is None:
                adv_idx = closest_diff_label(X_all, y_all, sample_idx,
                                             sample_label, t)
                if adv_idx == -1:
                    print("problem on: ", row, col)
                dist = dist_nd_bisection(X_all[sample_idx], X_all[adv_idx], clf)
                dist_nd_adv[row, col] = dist 
                continue

            adversarial_label = np.argmax(model.predictions(adversarial))
            if sample_label == adversarial_label:
                print("error on: ", row, col)
                continue
            dist_nd_adv[row, col] = np.linalg.norm(sample - adversarial)
            # print("\ttime: ", time.time() - s0)

    print("\ttime: ", time.time() - s)
    return dist_nd_adv

#
## def main():
#from sklearn import datasets
#from sklearn import linear_model
#from sklearn import manifold
#from sklearn import preprocessing
#import matplotlib.pyplot as plt
#
#from boundarymap import CLF
#from boundarymap import Grid
#import json
#
#n_samples = 100
#COLORS = ["#377eb8", "#ff7f00", '#4daf4a']
#
## generate dataset
#X, y = datasets.make_classification(n_samples=n_samples, n_classes=2,
#                                    n_features=16, n_redundant=0,
#                                    n_informative=10, random_state=1,
#                                    n_clusters_per_class=1)
#
#num_train = int(X.shape[0]*0.7)
#X_train = X[:num_train]
#y_train = y[:num_train]
#
#X_test = X[num_train:]
#y_test = y[num_train:]
#
## FIXME: normalize train data, apply normalization to test data
#scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
#
#clf_lr = linear_model.LogisticRegression()
#clf_lr.fit(X_train, y_train)
#print("train acc: ", clf_lr.score(X_train, y_train))
#print("test acc: ", clf_lr.score(X_test, y_test))
#
#tsne = manifold.TSNE(perplexity=10)
#X_proj = tsne.fit_transform(X_train)
#proj_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
#X_proj = proj_scaler.fit_transform(X_proj)
#
#colors = [COLORS[v] for v in y_train]
#plt.scatter(X_proj[:, 0], X_proj[:, 1], color=colors)
#
#clf = CLF(clf=clf_lr, clf_type="sklearn")
#basedir = '/tmp/syn_test/'
#
#R = 200
#N = [5]
#grid1 = Grid(X_proj, R)
#_, dmap = grid1.BoundaryMap(X_train, N[0], clf)
#
#dist_2d = dist2d(dmap)
#dist_2d /= dist_2d.max()
#dist_2d = 1.0 - dist_2d
#
#H, W, _ = dmap.shape
#GRID_SIZE = dmap.shape[0]
#
#dist_nd = distnd(dmap, X_train, X_proj, clf=clf_lr)
#dist_nd /= dist_nd.max()
#dist_nd = 1.0 - dist_nd
#
#dist_nd_2 = distnd2(X_train, X_proj, clf_lr, GRID_SIZE)
#dist_nd_2 /= dist_nd_2.max()
#dist_nd_2 = 1.0 - dist_nd_2
#
#from matplotlib.colors import hsv_to_rgb
#plt.subplot(411)
#plt.imshow(hsv_to_rgb(dmap))
#plt.subplot(412)
#plt.imshow(dist_2d)
#plt.subplot(413)
#plt.imshow(dist_nd)
#plt.subplot(414)
#plt.imshow(dist_nd_2)
#plt.show()
#
#np.save(basedir + 'X_train.npy', X_train)
#np.save(basedir + 'y_train.npy', y_train)
#np.save(basedir + 'y_pred.npy', y_train)
#np.save(basedir + 'X_proj.npy', X_proj)
#np.save(basedir + 'dmap_syn.npy', dmap)
#
#dist_2d_path = basedir + "dist_nd.npy"
#dist_nd_path = basedir + "dist_nd.npy"
#dist_nd_2_path = basedir + "dist_nd_2.npy"
#np.save(dist_nd_path, dist_nd)
#np.save(dist_nd_2_path, dist_nd_2)
#
#json_data = {}
#json_data['X'] = basedir + 'X_train.npy'
#json_data['y'] = basedir + 'y_train.npy'
#json_data['y_pred'] = basedir + 'y_pred.npy'
#json_data['proj'] = basedir + 'X_proj.npy'
#json_data['dense_map'] = basedir + 'dmap_syn.npy'
#json_data['dist_map'] = basedir + 'dist_nd_2.npy'
#
#with open(basedir + 'data.json', 'w') as fp:
#    json.dump(json_data, fp)
#    
#
#    
#if __name__ == '__main__':
#    main()
