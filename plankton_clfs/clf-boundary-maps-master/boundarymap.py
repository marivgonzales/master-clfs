import numpy as np
import joblib
from utils import Lerp, SampleSquare, get_neighbors
# from util import TransferFunc
from scipy import ndimage
import annoy
import foolbox

# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# from matplotlib.colors import hsv_to_rgb

import time
from tqdm import tqdm
import random
import json

from keras.models import load_model


# check colorscheme.png for a visual explanation
def SV(c, d):
    if d <= 0.5:
        # a: dark red rgb(0.372, 0.098, 0.145) - hsv(0.972, 1.0, 0.5)
        Sa = 1.0
        Va = 0.5
        # b: dark gray rgb(0.2, 0.2, 0.2) - hsv(0.0, 0.0, 0.2)
        Sb = 0.0
        Vb = 0.2
        # c: half gray rgb(0.5, 0.5, 0.5) - hsv(0.0, 0.0, 0.5)
        Sc = 0.0
        Vc = 0.5
        # d: pure red rgb(1.0, 0.0, 0.0) - hsv(0.0, 1.0, 1.0)
        Sd = 1.0
        Vd = 1.0
        S = Lerp(Lerp(Sb, Sa, c), Lerp(Sc, Sd, c), 2.0*d)
        V = Lerp(Lerp(Vb, Va, c), Lerp(Vc, Vd, c), 2.0*d)
    else:
        # a: pure red rgb(1.0, 0.0, 0.0) - hsv(0.0, 1.0, 1.0)
        Sa = 1.0
        Va = 1.0
        # b: half gray rgb(0.5, 0.5, 0.5) - hsv(0.0, 0.0, 0.5)
        Sb = 0.0
        Vb = 0.5
        # c: light gray rgb(0.8, 0.8, 0.8) - hsv(0.0, 0.0, 0.8)
        Sc = 0.0
        Vc = 0.8
        # d: bright pink rgb(? , ?, ?) - hsv(0.0, 0.2, 1.0)
        Sd = 0.2
        Vd = 1.0
        S = Lerp(Lerp(Sb, Sa, c), Lerp(Sc, Sd, c), 2.0*d - 1.0)
        V = Lerp(Lerp(Vb, Va, c), Lerp(Vc, Vd, c), 2.0*d - 1.0)
    return S, V


def closest_diff_label(X_all, y_all, sample_idx, sample_label, t):
    found = False
    num_n_prev = 0
    target_idx = -1
    while found is False:
        num_n = num_n_prev + 20000
        nns = t.get_nns_by_item(sample_idx, num_n)
        for nn in nns[num_n_prev:num_n]:
            if y_all[nn] != sample_label:
                found = True
                target_idx = nn
                break
        num_n_prev = num_n
    return target_idx


# Creates a grid of cells of (grid_size x grid_size).
# Each cell holds the index of each point in proj that maps to it.
# def build_grid(proj, grid_size):
#     cells = [[] for i in range(grid_size)]
# 
#     for i in range(grid_size):
#         cells[i] = [[] for _ in range(grid_size)]
# 
#     tile_size = 1.0/grid_size
#     # Adds point's indices to the corresponding cell
#     for idx in range(len(proj)):
#         p = proj[idx]
#         row = int(abs(p[1] - 1e-9)/tile_size)
#         col = int(abs(p[0] - 1e-9)/tile_size)
#         cells[row][col].append(idx)
#     return cells


def build_grid(proj, R, x_intrvls, y_intrvls):
    cells = [[] for i in range(R)]

    for i in range(R):
        cells[i] = [[] for _ in range(R)]

    for idx in range(len(proj)):
        p = proj[idx]
        # FIXME: should never happen that p[i] > 1.0, so this minimum
        # function should be removed
        row = np.minimum(R - 1, np.searchsorted(x_intrvls[1:], p[1]))
        col = np.minimum(R - 1, np.searchsorted(y_intrvls[1:], p[0]))
        cells[row][col].append(idx)
    return cells


# try to estimate the distance from a to a decision boundary given b and clf
# b: closest data point to a such that clf(a) != clf(b)
# clf: classifier trained on this dataset and used to create the densemap
# FIXME: create all possible samples and predict all of them in one step:
# - init = a
# - end = b
# - (b - a)/N
# -
# FIXME: remove square root?
# def dist_nd_bisection(a, b, clf):
#     N = 62
#     samples = np.zeros((N + 2, a.shape[0]))
#     samples[0] = a
#     samples[-1] = b
#     step = (b - a)/(N + 1)
#     for i in range(1, N + 1):
#         samples[i] = a + i*step
#     labels = clf.Predict(samples)
#     for i in range(1, N+2):
#         if labels[i] != labels[0]:
#             return np.linalg.norm(samples[0] - samples[i])
#     # print("-- error should not get to this point")
#     return np.linalg.norm(a - b)
def dist_nd_bisection(a, b, clf):
    tol = 1e-2
    max_iter = 3
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


# TODO: state pattern?
class CLF:
    def __init__(self, clf=None, clf_type="", shape=None, clf_path='',
                 clf_name=''):
        self.clf = clf

        if clf_type == "":
            self.clf_type = -1
        elif clf_type == "sklearn":
            self.clf_type = 0
        elif clf_type == "keras_cnn":
            self.clf_type = 1

        self.shape = shape
        self.clf_path = clf_path
        self.name = clf_name

    def save_json(self, path):
        data = {}
        data['clf_type'] = self.clf_type
        if self.shape is not None:
            data['shape'] = self.shape
        else:
            data['shape'] = 'None'
        data['name'] = self.name
        data['clf_path'] = self.clf_path

        with open(path, 'w') as outfile:
            json.dump(data, outfile)

    def save_joblib(self, path):
        data = {}
        data['clf_type'] = self.clf_type
        data['shape'] = self.shape
        data['name'] = self.name
        data['clf_path'] = self.clf_path

        joblib.dump(data, path)

    def load_json(self, path):
        with open(path) as f:
            data_json = json.load(f)

        self.clf_type = data_json['clf_type']
        if data_json['shape'] == 'None':
            self.shape = None
        else:
            self.shape = tuple(data_json['shape'])
        self.name = data_json['name']
        self.clf_path = data_json['clf_path']

        # TODO: load scikit clf
        if self.clf_type == 1:
            self.clf = load_model(self.clf_path)
        elif self.clf_type == 0:
            self.clf = joblib.load(self.clf_path)

    # FIXME: try to save all of it in one single file
    # If it is not possible, save at least a human readable text file (json)
    def load(self, path, bin_clf_path=None):
        data = joblib.load(path)

        self.clf_type = data['clf_type']
        self.shape = data['shape']
        self.name = data['name']
        if bin_clf_path is None:
            self.clf_path = data['clf_path']
        else:
            self.clf_path = bin_clf_path

        # TODO: load scikit clf
        if self.clf_type == 1:
            self.clf = load_model(self.clf_path)

    def LoadSKLearn(self, path, name=""):
        self.clf_type = 0
        # self.clf = pickle.load(open(path, "rb"))
        self.clf = joblib.load(path)
        self.name = name
        self.clf_path = path

    # TODO: remove this function
    def LoadKeras(self, arch_path, weight_path, name="", shape=None):
        from keras.models import model_from_json
        self.clf_type = 1

        self.name = name
        self.shape = shape

        # Model reconstruction from JSON file
        with open(arch_path, 'r') as f:
            self.clf = model_from_json(f.read())
        self.clf.load_weights(weight_path)

    def LoadKerasModel(self, model_path, name="", shape=None):
        self.clf_type = 1
        self.name = name
        self.shape = shape
        self.clf = load_model(model_path)
        self.clf_path = model_path

    def Predict(self, X):
        if self.shape is not None:
            # print("X shape: ", X.shape)
            # print("clf.shape: ", self.shape)
            X_new = np.reshape(X, (X.shape[0],) + self.shape)
        else:
            X_new = X
        y_pred = self.clf.predict(X_new)

        if self.clf_type == 1:
            y_pred = np.argmax(y_pred, axis=1)
        return y_pred

    def Score(self, X, y):
        if self.clf_type == 0:
            return self.clf.score(X, y)


class Grid:
    def __init__(self):
        self.cells = None

        orig_hues = [234, 0, 108, 288, 252, 72, 180, 324, 36, 144]
        syn_hues = [216, 18, 126, 306, 270, 90, 198, 342, 54, 162]
        self.CMAP_ORIG = np.array(orig_hues)/360.0
        self.CMAP_SYN = np.array(syn_hues)/360.0

        self.X_nd = None
        self.X_2d = None

    # TODO: make it optional to save the clf and inv_proj
    def save(self, path, clf_path='', inv_proj_path=''):
        print("save")
        data = {}
        data['R'] = self.R
        data['N'] = self.N
        data['clf_path'] = clf_path

        data['inv_proj_path'] = inv_proj_path
        # data['cells'] = self.cells
        # data['x_intrvls'] = self.x_intrvls
        # data['y_intrvls'] = self.y_intrvls

        data['dmap'] = self.dmap
        data['dist_2d'] = self.dist_2d
        data['dist_nd'] = self.dist_nd
        data['dist_nd2'] = self.dist_nd2
        data['dist_nd3'] = self.dist_nd3

        data['X_nd'] = self.X_nd
        data['X_2d'] = self.X_2d

        # data['syn_coords'] = self.syn_coords
        # data['syn_cells'] = self.syn_cells
        # data['syn_samples'] = self.syn_samples

        # data['syn_labels'] = self.syn_labels
        # data['orig_labels'] = self.orig_labels
        # data['X_all'] = self.X_all
        # data['Xp_all'] = self.Xp_all
        # data['cells_all'] = self.cells_all

        joblib.dump(data, path)

    # FIXME: remove base_dir and get only files from the same dir as path
    def load(self, path, clf_path=None, inv_proj_path=None):
        data = joblib.load(path)

        self.R = data['R']
        self.N = data['N']

        if clf_path is not None:
            self.clf = CLF()
            # self.clf.load(clf_path, bin_clf_path=bin_clf_path)
            self.clf.load_json(clf_path)
        if inv_proj_path is not None:
            # FIXME: how to load inverse projections?
            self.inv_proj = joblib.load(inv_proj_path)

        self.dmap = data['dmap']
        self.dist_2d = data['dist_2d']
        self.dist_nd = data['dist_nd']
        self.dist_nd2 = data['dist_nd2']
        self.dist_nd3 = data['dist_nd3']

        self.X_2d = data['X_2d']
        self.X_nd = data['X_nd']

        if 'cells' in data:
            self.cells = data['cells']
        if 'x_intrvls' in data:
            self.x_intrvls = data['x_intrvls']
        if 'y_intrvls' in data:
            self.y_intrvls = data['y_intrvls']
        if 'syn_coords' in data:
            self.syn_coords = data['syn_coords']
        if 'syn_cells' in data:
            self.syn_cells = data['syn_cells']
        if 'syn_samples' in data:
            self.syn_samples = data['syn_samples']

        if 'syn_labels' in data:
            self.syn_labels = data['syn_labels']
        if 'orig_labels' in data:
            self.orig_labels = data['orig_labels']
        if 'X_all' in data:
            self.X_all = data['X_all']
        if 'Xp_all' in data:
            self.Xp_all = data['Xp_all']
        if 'cells_all' in data:
            self.cells_all = data['cells_all']

    def fit(self, R, N, clf, inv_proj, X_nd=None, X_2d=None, syn_only=False):
        self.R = R
        self.N = N
        self.clf = clf

        self.syn_only = syn_only

        self.inv_proj = inv_proj

        if X_2d is not None:
            self.X_2d = np.copy(X_2d)
            self.X_nd = np.copy(X_nd)
        else:
            self.X_2d = np.copy(inv_proj.Xp)
            self.X_nd = np.copy(inv_proj.X)

        # proj = inv_proj.Xp
        xmin = np.min(self.X_2d[:, 0]) - 1e-6
        xmax = np.max(self.X_2d[:, 0]) + 1e-6
        ymin = np.min(self.X_2d[:, 1]) - 1e-6
        ymax = np.max(self.X_2d[:, 1]) + 1e-6
        self.x_intrvls = np.linspace(xmin, xmax, num=R + 1)
        self.y_intrvls = np.linspace(ymin, ymax, num=R + 1)

        # cells will store the indices of the points that fall inside each cell
        # of the grid
        if self.syn_only is False:
            self.cells = build_grid(self.X_2d, R, self.x_intrvls,
                                    self.y_intrvls)
        else:
            self.cells = build_grid([], R, self.x_intrvls,
                                    self.y_intrvls)

        self.dmap = None

        self.dist_2d = np.zeros((self.R, self.R))
        self.dist_nd = np.zeros((self.R, self.R))
        self.dist_nd2 = np.zeros((self.R, self.R))
        self.dist_nd3 = np.zeros((self.R, self.R))

    # compute the max and average number of points in the grid, assumes that
    # each cell will have at least self.N points
    def GetMaxAvgPts(self):
        num_pts = np.zeros((self.R, self.R))
        for row in range(self.R):
            for col in range(self.R):
                num_pts[row, col] = len(self.cells[row][col])
        num_pts[num_pts < self.N] = self.N
        return np.max(num_pts), np.mean(num_pts)

    # num_samples: number of samples to create
    def GenNewSamples(self, num_samples, row, col):
        if num_samples <= 0:
            return []

        limits = [self.x_intrvls[col], self.y_intrvls[row],
                  self.x_intrvls[col + 1], self.y_intrvls[row + 1]]
        sampled = SampleSquare(num_samples, limits)
        new_X = self.inv_proj.transform(np.array(sampled))
        return new_X

    def ComputeSynCoords(self):
        self.syn_coords = []
        self.syn_cells = [[] for i in range(self.R)]
        for i in range(self.R):
            self.syn_cells[i] = [[] for _ in range(self.R)]

        syn_idx = 0
        for row in range(self.R):
            for col in range(self.R):
                num_pts = len(self.cells[row][col])
                if num_pts >= self.N:
                    continue

                num_samples = self.N - num_pts
                limits = [self.x_intrvls[col], self.y_intrvls[row],
                          self.x_intrvls[col + 1], self.y_intrvls[row + 1]]
                sampled = SampleSquare(num_samples, limits)
                self.syn_coords.extend(sampled)

                for i in range(syn_idx, syn_idx + num_samples):
                    self.syn_cells[row][col].append(i)
                syn_idx += num_samples

    def ComputeSynSamples(self):
        self.syn_coords = np.array(self.syn_coords)
        # self.syn_coords = np.array(self.syn_coords)
        self.syn_samples = self.inv_proj.transform(self.syn_coords,
                                                   normalize=True)

    def PredictSynSamples(self):
        self.syn_labels = self.clf.Predict(self.syn_samples)

        if self.syn_only is False:
            self.orig_labels = self.clf.Predict(self.X_nd)
            num_orig = self.X_nd.shape[0]
        else:
            self.orig_labels = []
            num_orig = 0

        num_syn = self.syn_samples.shape[0]
        num_total = num_orig + num_syn
        num_features = self.X_nd.shape[1]
        self.X_all = np.zeros((num_total, num_features))
        self.Xp_all = np.zeros((num_total, 2))

        if self.syn_only is False:
            self.X_all[:num_orig] = self.X_nd
            self.X_all[num_orig:] = self.syn_samples
            self.Xp_all[:num_orig] = self.X_2d
            self.Xp_all[num_orig:] = self.syn_coords
        else:
            self.X_all[:] = self.syn_samples
            self.Xp_all[:] = self.syn_coords

        self.y_all = np.concatenate((self.orig_labels, self.syn_labels))
        self.cells_all = build_grid(self.Xp_all, self.R, self.x_intrvls,
                                    self.y_intrvls)

    def getMaxAvg(self):
        num_pts = 0
        max_pts = 0
        for row in range(self.R):
            for col in range(self.R):
                pts_in_cell = len(self.cells[row][col]) + len(self.syn_cells[row][col])
                num_pts += pts_in_cell
                if pts_in_cell > max_pts:
                    max_pts = pts_in_cell
        return num_pts, max_pts

    def ComputeColorScheme(self, H, compute_hsv):
        dmap = np.zeros((self.R, self.R, 3))
        max_pts, avg_pts = self.getMaxAvg()
        cmap = self.CMAP_SYN
        for row in range(self.R):
            for col in range(self.R):
                # labels_orig = self.orig_labels[self.cells[row][col]]
                labels_orig = []
                if self.syn_only is False:
                    labels_orig = self.orig_labels[self.cells[row][col]]
                labels_syn = self.syn_labels[self.syn_cells[row][col]]
                labels = np.concatenate((labels_orig, labels_syn))
                labels = labels.astype(int)

                # num_pts = len(labels)
                num_pts = len(labels)
                # counts = np.bincount(labels)
                counts = np.bincount(labels)
                num_winning = np.max(counts)

                # decision
                decision = np.argmax(counts)
                # confusion c
                c = num_winning/num_pts
                # density d
                # d = num_pts/max_pts
                d = min(H*num_pts/avg_pts, 1.0)

                if compute_hsv is True:
                    hue = cmap[decision]
                    s, v = SV(c, d)
                    dmap[row, col] = np.array([hue, s, v])
                else:
                    dmap[row, col] = np.array([decision, c, d])

        return dmap

    # This function breaks the boundary map computation in batches:
    # 1. Compute cells and check where synthetic points are needed
    # 2. Backproject the newly computed 2D coords into synthetic points
    # 3. Classify both orig and synthetic samples
    # 4. Compute the densemaps
    # It was done hoping that classifying all samples at once would be faster,
    # but uses a lot more memory.
    def BoundaryMapBatch(self, H=0.05, compute_hsv=True):
        print("\t\tCompute syn coords")
        s = time.time()
        self.ComputeSynCoords()
        print("\t\t\ttime: ", time.time() - s)
        print("\t\tCompute syn samples")
        s = time.time()
        self.ComputeSynSamples()
        print("\t\t\ttime: ", time.time() - s)
        print("\t\tPredict syn samples")
        s = time.time()
        self.PredictSynSamples()
        print("\t\t\ttime: ", time.time() - s)
        print("\t\tComputeColorScheme")
        s = time.time()
        self.dmap = self.ComputeColorScheme(H, compute_hsv)
        print("\t\t\ttime: ", time.time() - s)
        return self.dmap

    # TODO: make this code work when num_per_cell == 0 -> sparse map
    def BoundaryMap(self, H=0.05):
        self.dmap = np.zeros((self.R, self.R, 3))

        max_pts, avg_pts = self.GetMaxAvgPts(self.N)
        for row in range(self.R):
            for col in range(self.R):
                num_pts = len(self.cells[row][col])
                if num_pts != 0:
                    cmap = self.CMAP_ORIG
                else:
                    cmap = self.CMAP_SYN

                X_sub = [x for x in self.X_nd[self.cells[row][col]]]
                X_sub = np.array(X_sub)

                # number of synthetic samples that will be created
                num_samples = self.N - num_pts
                if num_samples > 0:
                    new_samples = self.GenNewSamples(num_samples, row, col)
                    if len(X_sub) > 0:
                        X_sub = np.concatenate((X_sub, new_samples))
                    else:
                        X_sub = new_samples

                # X_sub.extend(new_samples)
                # X_sub = np.array(X_sub)

                # now that new data points were created, there are at least
                # num_pts samples in this cell
                if num_pts < self.N:
                    num_pts = self.N

                # Compute color for this cell
                labels = self.clf.Predict(X_sub)

                counts = np.bincount(labels)
                num_winning = np.max(counts)
                # decision
                hue = cmap[np.argmax(counts)]
                # cconfusion c
                c = num_winning/num_pts
                # density d
                # d = num_pts/max_pts
                d = min(H*num_pts/avg_pts, 1.0)

                s, v = SV(c, d)
                self.dmap[row, col] = np.array([hue, s, v])
        return self.dmap

    def hue_cmp(self, h1, h2):
        if h1 in self.CMAP_ORIG:
            idx = np.where(self.CMAP_ORIG == h1)[0][0]
            alt_h1 = self.CMAP_SYN[idx]
        else:
            # h1 must be in CMAP_SYN
            idx = np.where(self.CMAP_SYN == h1)[0][0]
            alt_h1 = self.CMAP_ORIG[idx]
        if h1 == h2 or alt_h1 == h2:
            return 0
        return 1

    def boundary_cells(self):
        H, W, _ = self.dmap.shape
        on_boundary = np.full((H, W), 1.0)
        for row in range(H):
            for col in range(W):
                neighbors = get_neighbors(row, col, H, W, n8=True)
                cell_hue = self.dmap[row, col, 0]

                for n in neighbors:
                    neighbor_hue = self.dmap[n[0], n[1], 0]
                    if self.hue_cmp(cell_hue, neighbor_hue) == 1:
                        on_boundary[row, col] = 0.0
                        on_boundary[n[0], n[1]] = 0.0
        return on_boundary

    def dist2D(self):
        print("\n\tDist 2D")
        s = time.time()
        # FIXME: if self.dmap is None, return an error
        self.dist_2d = ndimage.distance_transform_edt(self.boundary_cells())
        print("\t\ttime: ", time.time() - s)
        return self.dist_2d

    def refine_boundaries(self, db_map):
        H, W, _ = self.dmap.shape
        for row in range(H):
            for col in range(W):
                cell_hue = self.dmap[row, col, 0]
                db = db_map[row, col]
                other_hue = self.dmap[db[0], db[1], 0]

                if self.hue_cmp(cell_hue, other_hue) != 0:
                    continue

                neighbors = get_neighbors(db[0], db[1], H, W, n8=True)
                for n in neighbors:
                    n_hue = self.dmap[n[0], n[1], 0]
                    if self.hue_cmp(cell_hue, n_hue) != 1:
                        continue
                    db_map[row, col] = n
                    break

    def distance_nd_cells(self, src_r, src_c, dest_r, dest_c):
        # src_orig = self.inv_proj.X[self.cells[src_r][src_c]]
        # src_syn = self.syn_samples[self.syn_cells[src_r][src_c]]
        # src_samples = np.concatenate((src_orig, src_syn))

        # dest_orig = self.inv_proj.X[self.cells[dest_r][dest_c]]
        # dest_syn = self.syn_samples[self.syn_cells[dest_r][dest_c]]
        # dest_samples = np.concatenate((dest_orig, dest_syn))

        # dist = 0.0
        # for src in src_samples:
        #     for dest in dest_samples:
        #         dist += dist_nd_bisection(src, dest, self.clf)

        if len(self.cells[src_r][src_c]) > 0:
            src_sample = self.X_nd[self.cells[src_r][src_c]][0]
        else:
            src_sample = self.syn_samples[self.syn_cells[src_r][src_c]][0]

        if len(self.cells[dest_r][dest_c]) > 0:
            dest_sample = self.X_nd[self.cells[dest_r][dest_c]][0]
        else:
            dest_sample = self.syn_samples[self.syn_cells[dest_r][dest_c]][0]

        return dist_nd_bisection(src_sample, dest_sample, self.clf)

    def distnD_batch(self):
        # FIXME: if self.dmap is None, return an error
        print("\n\tDist nD batch")
        s_total = time.time()
        print("\t\tboundary cells")
        s = time.time()
        on_db = self.boundary_cells()
        print("\t\t\ttime: ", time.time() - s)
        _, inds = ndimage.distance_transform_edt(on_db, return_indices=True)
        db_map = np.dstack((inds[0], inds[1]))
        print("\t\trefine boundries")
        s = time.time()
        self.refine_boundaries(db_map)
        print("\t\t\ttime: ", time.time() - s)

        print("\t\tdistance nd for each cell")
        self.dist_nd = np.zeros((self.R, self.R))
        for row in tqdm(range(self.R)):
            for col in range(self.R):
                db_r, db_c = db_map[row, col, 0], db_map[row, col, 1]
                self.dist_nd[row, col] = self.distance_nd_cells(row, col, db_r, db_c)
        print("\t\ttotal time: ", time.time() - s_total)
        return self.dist_nd

    def distnD2_batch(self):
        print("\n\tDist nD2 batch")
        s = time.time()
        self.dist_nd2 = np.zeros((self.R, self.R))
        num_features = self.X_nd.shape[1]
        num_total = self.X_all.shape[0]

        print("\t\tconstructing annoy structure")
        s0 = time.time()
        t = annoy.AnnoyIndex(num_features)
        for i in range(num_total):
            t.add_item(i, self.X_all[i])
        t.build(1)
        print('\t\t\ttime: ', time.time() - s0)
        print("\t\tcomputing distances between nd samples")

        y_all = np.concatenate((self.orig_labels, self.syn_labels))
        for row in tqdm(range(self.R)):
            for col in range(self.R):
                num_samples_cell = len(self.cells_all[row][col])
                r = 0
                if num_samples_cell > 1:
                    r = random.randint(0, num_samples_cell - 1)
                sample_idx = self.cells_all[row][col][r]
                sample = self.X_all[sample_idx]
                # label_sample = clf.Predict(np.array([sample]))[0]
                label_sample = y_all[sample_idx]

                found = False
                num_n_prev = 0
                target_idx = -1
                while found is False:
                    num_n = num_n_prev + 20000
                    nns = t.get_nns_by_item(sample_idx, num_n)
                    for nn in nns[num_n_prev:num_n]:
                        if y_all[nn] != label_sample:
                            found = True
                            target_idx = nn
                            break
                    num_n_prev = num_n
                d = dist_nd_bisection(sample, self.X_all[target_idx], self.clf)
                self.dist_nd2[row, col] = d

        print("\ttime: ", time.time() - s)
        return self.dist_nd2

    def distnD3_batch(self):
        print("\n\tDist nD3 batch")
        s = time.time()
        # foolbox model
        # TODO: make CLF class compute the adversarial model
        xmin = self.X_all.min()
        xmax = self.X_all.max()
        model = foolbox.models.KerasModel(self.clf.clf, bounds=(xmin, xmax))
        attack = foolbox.attacks.FGSM(model)
        # attack_fallback = foolbox.attacks.BoundaryAttack(model)

        num_features = self.X_all.shape[1]
        num_total = self.X_all.shape[0]

        print("\t\tconstructing annoy structure")
        t = annoy.AnnoyIndex(num_features)
        for i in range(num_total):
            t.add_item(i, self.X_all[i])
        t.build(1)

        self.dist_nd3 = np.zeros((self.R, self.R))

        y_all = np.concatenate((self.orig_labels, self.syn_labels))
        print("\t\tcomputing distance to boundary nd by adversarial examples")
        for row in tqdm(range(self.R)):
            for col in range(self.R):
                # print("row, col: ", row,  col)
                # s0 = time.time()
                num_samples_cell = len(self.cells_all[row][col])
                if num_samples_cell > 1:
                    r = random.randint(0, num_samples_cell - 1)
                else:
                    r = 0
                sample_idx = self.cells_all[row][col][r]
                sample = self.X_all[sample_idx]
                # label_sample = clf.Predict(np.array([sample]))[0]
                sample_label = y_all[sample_idx]

                sample = sample.reshape(self.clf.shape)
                adversarial = attack(sample, sample_label)
                if adversarial is None:
                    # print("\t\t\t\tadversarial is None: ", row, col, "bisection")
                    # adversarial = attack_fallback(sample, sample_label)
                    # if adversarial is None:
                    adv_idx = closest_diff_label(self.X_all, y_all,
                                                 sample_idx, sample_label, t)
                    if adv_idx == -1:
                        print("problem on: ", row, col)
                    dist = dist_nd_bisection(self.X_all[sample_idx],
                                             self.X_all[adv_idx], self.clf)
                    continue
                else:
                    dist = np.linalg.norm(sample - adversarial)

                self.dist_nd3[row, col] = dist
                # adversarial_label = np.argmax(model.predictions(adversarial))
                # if sample_label == adversarial_label:
                #     print("error on: ", row, col)
                #     continue
                # print("\ttime: ", time.time() - s0)
        print("\ttime: ", time.time() - s)
        return self.dist_nd3


# def PlotDenseMap(dense_map, title, filename, format='pdf'):
#     tmp_dense = np.flip(dense_map, axis=0)
#     tmp_dense = TransferFunc(tmp_dense, 0.7)
#     rgb_img = hsv_to_rgb(tmp_dense)
# 
#     plt.xticks([])
#     plt.yticks([])
# 
#     plt.imshow(rgb_img, interpolation='none')
#     plt.title(title)
#     plt.savefig(filename + "." + format, format=format)
#     plt.clf()
# 
# 
# def PlotLegend(path, colors, labels):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
# 
#     handles = []
#     for c in colors:
#         handles.append(ax.scatter([], [], color=c))
# 
#     figlegend = plt.figure()
#     figlegend.legend(handles, labels, 'center')
#     figlegend.savefig(path, format='pdf')
#     plt.clf()
# 
# 
# def PlotProjection(proj, y_pred, path, title, leg_path="", labels=[]):
#     # COLORS are the rgb counterparts of Grid.CMAP_SYN
#     COLORS = np.array([[0.09, 0.414, 0.9, 0.5],
#                        [0.9, 0.333, 0.09, 0.5],
#                        [0.09, 0.9, 0.171, 0.5],
#                        [0.9, 0.09, 0.819, 0.5],
#                        [0.495, 0.09, 0.9, 0.5],
#                        [0.495, 0.9, 0.09, 0.5],
#                        [0.09, 0.657, 0.9, 0.5],
#                        [0.9, 0.09, 0.333, 0.5],
#                        [0.9, 0.819, 0.09, 0.5],
#                        [0.09, 0.9, 0.657, 0.5]])
# 
#     colors = [COLORS[i] for i in y_pred]
# 
#     plt.axes().set_aspect('equal')
#     plt.scatter(proj[:, 0], proj[:, 1], color=colors, s=10.0)
#     plt.title(title)
#     plt.xticks([])
#     plt.yticks([])
#     plt.savefig(path, format='pdf')
#     plt.clf()
# 
#     if leg_path != "":
#         PlotLegend(leg_path, COLORS[:len(labels)], labels)
# 
# 
# def PlotProjectionErr(grid, proj, y_pred, y_true, path, title, leg_path="", labels=[]):
#     # COLORS are the rgb counterparts of Grid.CMAP_SYN
#     COLORS = np.array([[0.09, 0.414, 0.9, 0.25],
#                        [0.9, 0.333, 0.09, 0.25],
#                        [0.09, 0.9, 0.171, 0.25],
#                        [0.9, 0.09, 0.819, 0.25],
#                        [0.495, 0.09, 0.9, 0.25],
#                        [0.495, 0.9, 0.09, 0.25],
#                        [0.09, 0.657, 0.9, 0.25],
#                        [0.9, 0.09, 0.333, 0.25],
#                        [0.9, 0.819, 0.09, 0.25],
#                        [0.09, 0.9, 0.657, 0.25]])
# 
#     colors = [COLORS[i] for i in y_pred]
#     colors = np.array(colors)
# 
#     #edge_colors = [COLORS[i] for i in y_true]
# 
#     plt.axes().set_aspect('equal')
#     x_min, x_max = np.min(grid.x_intrvls), np.max(grid.x_intrvls)
#     y_min, y_max = np.min(grid.y_intrvls), np.max(grid.y_intrvls)
#     print(x_min, x_max)
#     print(y_min, y_max)
#     for x in grid.x_intrvls:
#         plt.plot([x,  x], [y_min, y_max], color='k')
#     for y in grid.y_intrvls:
#         plt.plot([x_min,  x_max], [y, y], color='k')
# 
#     plt.scatter(proj[y_pred == y_true][:, 0], proj[y_pred == y_true][:, 1], color=colors[y_pred == y_true], s=10.0)
#     plt.scatter(proj[y_pred != y_true][:, 0], proj[y_pred != y_true][:, 1], color=[0.0, 0.0, 0.0, 0.8], s=10.0, marker='*')
#     plt.title(title)
#     plt.xticks([])
#     plt.yticks([])
# 
#     plt.show()
#     plt.clf()
# 
#     if leg_path != "":
#         legend_colors = np.zeros((len(labels) + 1, COLORS.shape[1]))
#         legend_colors[:len(labels)] = COLORS[:len(labels)]
#         legend_colors[len(labels)] = np.array([0.0, 0.0, 0.0, 0.8])
#         PlotLegend(leg_path, COLORS[:len(labels)], labels)

