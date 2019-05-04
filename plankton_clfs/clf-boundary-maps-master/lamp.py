import numpy as np
from scipy.spatial import distance
from scipy.linalg import svd
from scipy.linalg import solve
from scipy.linalg import qr
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

import joblib
import time

# from tqdm import tqdm


# Forced projection method as decribed in the paper "On improved projection
# techniques to support visual exploration of multi-dimensional data sets"
def force_method(X, init='random', delta_frac=10.0, max_iter=50):
    # TODO: tSNE from scikit learn can initialize projections based on PCA
    if init == 'random':
        X_proj = np.random.rand(X.shape[0], 2)
    # TODO: something like:
    # else if init == ' PCA':
    #   X_proj = PCA(X)

    vec_dist = distance.pdist(X, 'euclidean')
    dmin = np.min(vec_dist)
    dmax = np.max(vec_dist)
    dist_matrix = distance.squareform(vec_dist)

    dist_diff = dmax - dmin

    index = np.random.permutation(X.shape[0])
    # TODO: better stopping criteria?
    # TODO: this is _slow_: consider using squared distances when possible
    #       - using sqeuclidean it is faster but results are worse
    for k in range(max_iter):
        for i in range(X_proj.shape[0]):
            # x_prime = X_proj[i]
            instance1 = index[i]
            x_prime = X_proj[instance1]
            for j in range(X_proj.shape[0]):
                instance2 = index[j]
                if instance1 == instance2:
                    # FIXME: the paper compares x\prime to q\prime, here I'm
                    # comparing only the indices
                    continue
                q_prime = X_proj[instance2]

                # if np.allclose(x_prime, q_prime):
                #     continue

                v = q_prime - x_prime
                dist_xq = distance.euclidean(x_prime, q_prime)
                delta = dist_matrix[instance1, instance2]/dist_diff - dist_xq
                # FIXME the algorithm desbribed in the paper states:
                # "move q_prime in the direction of v by a fraction of delta"
                # what is a good value for delta_frac?
                delta /= delta_frac

                X_proj[instance2] = X_proj[instance2] + v*delta

    # TODO: is normalization really necessary?
    X_proj = (X_proj - X_proj.min(axis=0))/(X_proj.max(axis=0) - X_proj.min(axis=0))

    return X_proj


# Heavily based on lamp implementation from:
# https://github.com/thiagohenriquef/mppy
# In my tests, this method worked reasonably well when data was normalized
# in range [0,1].
def lamp2d(X, num_ctrl_pts=None, delta=10.0, ctrl_pts_idx=None, fastSVD=True):
    # k: the number of control points
    # LAMP paper argues that few control points are needed. sqrt(|X|) is used
    # here as it the necessary number for other methods
    if ctrl_pts_idx is None:
        if num_ctrl_pts is None:
            k = int(np.sqrt(X.shape[0]))
        else:
            k = num_ctrl_pts
        ctrl_pts_idx = np.random.randint(0, X.shape[0], k)

    X_s = X[ctrl_pts_idx]
    Y_s = force_method(X_s, delta_frac=delta)

    X_proj = np.zeros((X.shape[0], 2))
    # LAMP algorithm
    for idx in range(X.shape[0]):
        skip = False

        # 1. compute weighs alpha_i
        alpha = np.zeros(X_s.shape[0])
        for i in range(X_s.shape[0]):
            diff = X_s[i] - X[idx]
            diff2 = np.dot(diff, diff)
            if diff2 < 1e-4:
                # X_s[i] and X[idx] are almost the same point, so
                # project to the same point (Y_s[i]
                X_proj[idx] = Y_s[i]
                skip = True
                break
            alpha[i] = 1.0/diff2

        if skip is True:
            continue

        # 2. compute x_tilde, y_tilde
        sum_alpha = np.sum(alpha)
        x_tilde = np.sum(alpha[:, np.newaxis]*X_s, axis=0)/sum_alpha
        y_tilde = np.sum(alpha[:, np.newaxis]*Y_s, axis=0)/sum_alpha

        # 3. build matrices A and B
        x_hat = X_s - x_tilde
        y_hat = Y_s - y_tilde

        alpha_sqrt = np.sqrt(alpha)
        A = alpha_sqrt[:, np.newaxis]*x_hat
        B = alpha_sqrt[:, np.newaxis]*y_hat

        # 4. compute the SVD decomposition UDV from (A^T)B
        u, s, vh = svd(np.dot(A.T, B), full_matrices=False)
        M = np.dot(u, vh)

        # 6. Compute the mapping (x - x_tilde)M + y_tilde
        X_proj[idx] = np.dot(X[idx] - x_tilde, M) + y_tilde

    return X_proj


# FIXME: precompute kdtree?
def ilamp(data, data_proj, p, k=6):
    # 0. compute X_s and Y_s
    tree = KDTree(data_proj)

    dist, ind = tree.query([p], k=k)
    # ind is a (1xdim) array
    ind = ind[0]
    X_proj = data_proj[ind]
    X = data[ind]

    # 1. compute weights alpha_i
    alpha = np.zeros(X_proj.shape[0])
    for i in range(X_proj.shape[0]):
        diff = X_proj[i] - p
        diff2 = np.dot(diff, diff)

        # FIXME: in the original paper, if the point is too close to a "real"
        # data point, the real one is returned. Keep it this way?
        if diff2 < 1e-6:
            # difference is too small, the counter part to p
            # precisely X[i]
            return X[i]
        alpha[i] = 1.0/diff2

    sum_alpha = np.sum(alpha)
    # 2. compute x_tilde, y_tilde
    x_tilde = np.sum(alpha[:, np.newaxis]*X, axis=0)/sum_alpha
    y_tilde = np.sum(alpha[:, np.newaxis]*X_proj, axis=0)/sum_alpha

    # 3. build matrices A and B
    x_hat = X - x_tilde
    y_hat = X_proj - y_tilde

    alpha_sqrt = np.sqrt(alpha)
    A = alpha_sqrt[:, np.newaxis]*y_hat
    B = alpha_sqrt[:, np.newaxis]*x_hat

    # 4. compute the SVD decomposition UDV from (A^T)B
    u, s, vh = svd(np.dot(A.T, B), full_matrices=False)

    # 5. let M = UV
    M = np.dot(u, vh)

    return np.dot(p - y_tilde, M) + x_tilde


def theta(r):
    c = 0.0
    eps = 1.0
    return np.sqrt(c + eps*(r**2))


# computes control points by regularized orthgonal least squares (rols)
def ctrl_rols(X_in, Xp_in, kernel, num_ctrl, sub_sample=0):
    if sub_sample > 0:
        indices = np.random.permutation(X_in.shape[0])
        X = X_in[indices[:sub_sample]]
        Xp = Xp_in[indices[:sub_sample]]
    else:
        X = X_in
        Xp = Xp_in

    phi = distance.cdist(X, X)
    phi = kernel(phi)
    W, A = qr(phi)

    # make diag(A) == 1
    # FIXME: assumes that A is M x M
    diagA = np.copy(np.diag(A))
    for i in range(diagA.shape[0]):
        A[i, :] /= diagA[i]
        W[:, i] *= diagA[i]

    # lambda_vec = solve(phi, Xp, assume_a="sym")
    lambda_vec = solve(phi, Xp)
    gs = np.dot(A, lambda_vec)

    # gs is N x 2 vector, now take gsTgs
    gtg = np.zeros(gs.shape[0])
    # Xp is a N x 2 vector, now take the dot product of each component store in
    # yty
    yty = np.zeros(gs.shape[0])
    for i in range(gtg.shape[0]):
        gtg[i] = np.dot(gs[i], gs[i])
        yty[i] = np.dot(Xp[i], Xp[i])

    beta = 0.5

    wtw = np.zeros(W.shape[1])
    for i in range(W.shape[1]):
        wtw[i] = np.dot(W[:, i], W[:, i])

    rerrs = (wtw + beta)*gtg/yty
    sorted_idx = np.argsort(rerrs)

    return sorted_idx[-num_ctrl:]


def rbf_coefs(X, Xp, kernel, normalize=False, reg_coef=0.0):
    phi = distance.cdist(Xp, Xp)
    if normalize is True:
        phi = (phi - phi.min())/(phi.max() - phi.min())
    phi = kernel(phi)

    coefs = []
    if reg_coef > 0.0:
        # regularizing
        ptp = np.dot(np.transpose(phi), phi)
        lI = reg_coef*np.eye(X.shape[0])
        inv = np.linalg.inv(ptp + lI)
        M = np.dot(inv, np.transpose(phi))
        for i in range(X.shape[1]):
            # regularized
            li = np.dot(M, X[:, i])
            coefs.append(li)
    else:
        M = np.linalg.inv(phi)
        for i in range(X.shape[1]):
            li = np.dot(M, X[:, i])
            coefs.append(li)

        # FIXME: the following code solves X.shape[1]
        # for i in range(X.shape[1]):
        #     # TODO: set assume_a to "pos" (positive definite)
        #     # li = solve(phi, X[:, i], assume_a='sym')
        #     coefs.append(li)

    return np.array(coefs)


class ILAMP():
    def __init__(self, n_neighbors=10):
        self.X = None
        self.Xp = None
        self.nbrs = None
        self.n_neighbors = n_neighbors

    def fit(self, X, Xp):
        self.X = np.copy(X)
        self.Xp = np.copy(Xp)
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors,
                                     algorithm='ball_tree').fit(Xp)

    def save(self, path):
        data = {}
        data['X'] = self.X
        data['Xp'] = self.Xp
        data['n_neighbors'] = self.n_neighbors
        # FIXME: save knn with joblib
        data['nbrs'] = self.nbrs

        joblib.dump(data, path)

    def load(self, path):
        data = joblib.load(path)
        self.X = np.copy(data['X'])
        self.Xp = np.copy(data['Xp'])
        self.n_neighbors = data['n_neighbors']
        # FIXME: load knn with joblib
        # self.nbrs = data['nbrs']
        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors,
                                     algorithm='ball_tree').fit(self.Xp)

    def transform(self, ps, normalize=False):
        num_pts = len(ps)
        invs = np.zeros((num_pts, self.X.shape[1]))
        # print("array: ", ps, ps.shape, ps.dtype)
        _, indices = self.nbrs.kneighbors(ps)

        for i in range(num_pts):
            X_proj = self.Xp[indices[i]]
            X = self.X[indices[i]]

            # 1. compute weights alpha_i
            diff = X_proj - ps[i]
            diff = np.einsum('ij,ij->i', diff, diff)

            idx_zero = np.argwhere(np.isclose(diff, 0, atol=1e-6))
            # FIXME: in the original paper, if the point is too close to a
            # "real" data point, the real one is returned. Keep it this way?
            if idx_zero.size > 0:
                # invs.append(X[idx_zero[0][0]])
                invs[i] = X[idx_zero[0][0]]
                continue

            alpha = 1.0/diff
            sum_alpha = np.sum(alpha)
            # 2. compute x_tilde, y_tilde
            x_tilde = np.sum(alpha[:, np.newaxis]*X, axis=0)/sum_alpha
            y_tilde = np.sum(alpha[:, np.newaxis]*X_proj, axis=0)/sum_alpha

            # 3. build matrices A and B
            x_hat = X - x_tilde
            y_hat = X_proj - y_tilde

            alpha_sqrt = np.sqrt(alpha)
            A = alpha_sqrt[:, np.newaxis]*y_hat
            B = alpha_sqrt[:, np.newaxis]*x_hat

            u, s, vh = svd(np.dot(A.T, B), full_matrices=False)
            # 5. let M = UV
            M = np.dot(u, vh)
            inv = np.dot(ps[i] - y_tilde, M) + x_tilde
            if normalize is True:
                inv = (inv - inv.min())/(inv.max() - inv.min())

            invs[i] = inv

        return invs


class RBFInv():
    def __init__(self, mode='cluster', num_ctrl=10, kernel='gaussian', eps=1.0,
                 c=1.0, normalize_c=False, normalize_d=False):
        self.mode = mode
        self.num_ctrl = num_ctrl

        self.kernel_str = kernel
        if kernel == 'gaussian':
            self.kernel = self.gaussian
        elif kernel == 'multi_quadrics':
            self.kernel = self.multi_quadrics
        elif kernel == 'inv_mq':
            self.kernel = self.inv_mq
        # FIXME: else return error

        self.eps = eps
        self.c = c
        self.normalize_c = normalize_c
        self.normalize_d = normalize_d

    def _compute_clusters(self):
        distances = distance.cdist(self.X_ctrl, self.X)
        self.clusters = [[] for _ in range(self.num_ctrl)]
        for i in range(distances.shape[1]):
            c = np.argmin(distances[:, i])
            self.clusters[c].append(i)

    # Does the necessary pre-computations before back-projecting a point.
    # X/Xp: nD and 2D points are not copied.
    # mode: selects how control points will be used.
    #   - 'all': uses all the points as control points
    #   - 'rols': uses num_ctrl as control points obtained from the entire
    #       dataset using ROLS.
    #   - 'cluster': create num_ctrl clusters and uses all the points belonging
    #       to a cluster as control points
    #   - 'neighbors': uses the num_ctrl-nearest-neighbors as control points
    # scale_c/normalize_c: defines if the distance matrix computed using
    #   Xp should be normalized and scaled
    # kernel: kernel function used to interpolate points
    def fit(self, X, Xp):
        self.X = np.copy(X)
        self.Xp = np.copy(Xp)

        if self.mode == 'cluster':
            # new_array = [tuple(row) for row in self.X]
            # uniques, unique_indices = np.unique(new_array, return_index=True,
            #                                     axis=0)
            # if len(unique_indices) < len(self.X):
            #     print("has duplicates!")
            #     self.X = self.X[unique_indices]
            #     self.Xp = self.Xp[unique_indices]
            print("\t\tComputing ctrl pts")
            s = time.time()
            self.ctrl_pts = ctrl_rols(X, Xp, self.kernel, self.num_ctrl)
            print("\t\t\ttime ", time.time() - s)
            self.X_ctrl = X[self.ctrl_pts]
            self.Xp_ctrl = Xp[self.ctrl_pts]
            print("\t\tComputing clusters")
            s = time.time()
            self._compute_clusters()
            print("\t\t\ttime ", time.time() - s)
            print("\t\tComputing coefs for each clusters")
            s = time.time()

            self.coefs = []
            for cluster in self.clusters:
                X_sel = self.X[cluster]
                Xp_sel = self.Xp[cluster]
                coef = rbf_coefs(X_sel, Xp_sel, self.kernel, self.normalize_c)
                self.coefs.append(coef)
            print("\t\t\ttime ", time.time() - s)
        elif self.mode == 'neighbors':
            print("\t\tComputing neighbors")
            s = time.time()
            self.nbrs = NearestNeighbors(n_neighbors=self.num_ctrl,
                                         algorithm='ball_tree').fit(Xp)
            print("\t\t\ttime ", time.time() - s)
        elif self.mode == 'rols':
            print("\n\t\tComputing control points")
            s = time.time()
            self.ctrl_pts = ctrl_rols(X, Xp, self.kernel, self.num_ctrl)
            print("\t\t\ttime ", time.time() - s)
            self.X_ctrl = X[self.ctrl_pts]
            self.Xp_ctrl = Xp[self.ctrl_pts]
            print("\n\t\tComputing coefs")
            s = time.time()
            self.coefs = rbf_coefs(self.X_ctrl, self.Xp_ctrl, self.kernel,
                                   self.normalize_c)
            print("\t\t\ttime ", time.time() - s)

    def _transform_cluster(self, ps, normalize):
        num_pts = len(ps)
        invs = np.zeros((num_pts, self.X.shape[1]))
        for i in range(num_pts):
            # find Xp[ctrl_pts] closest to p
            p = ps[i]
            closest = np.argmin(((self.Xp_ctrl - p)**2).sum(axis=1))
            # special case: only one point in the cluster, return itself?
            if len(self.clusters[closest]) == 1:
                # invs.append(self.X[self.clusters[closest]][0])
                invs[i] = self.X[self.clusters[closest]][0]
                continue

            #  now use all points in the cluster as control points
            # X_sel = self.X[self.clusters[closest]]
            Xp_sel = self.Xp[self.clusters[closest]]
            # coefs = rbf_coefs(X_sel, Xp_sel, self.kernel, self.normalize_c)
            coefs = self.coefs[closest]
            diffs = np.linalg.norm(Xp_sel - p, axis=1)
            if self.normalize_d is True:
                diffs = (diffs - diffs.min())/(diffs.max() - diffs.min())
            diffs = self.kernel(diffs)

            inv = np.dot(coefs, diffs)
            if normalize is True:
                inv = (inv - inv.min())/(inv.max() - inv.min())
            invs[i] = inv
        return invs

    def _transform_neighbors(self, ps, normalize):
        num_pts = len(ps)
        invs = np.zeros((num_pts, self.X.shape[1]))
        _, indices = self.nbrs.kneighbors(ps)
        for i in range(num_pts):
            ctrl_pts = indices[i]
            p = ps[i]

            X_sel = self.X[ctrl_pts]
            Xp_sel = self.Xp[ctrl_pts]
            coefs = rbf_coefs(X_sel, Xp_sel, self.kernel, self.normalize_c)
            # diffs = self.kernel(np.linalg.norm(Xp_sel - p, axis=1))
            diffs = np.linalg.norm(Xp_sel - p, axis=1)
            if self.normalize_d is True:
                diffs = (diffs - diffs.min())/(diffs.max() - diffs.min())
            diffs = self.kernel(diffs)
            inv = np.dot(coefs, diffs)
            if normalize is True:
                inv = (inv - inv.min())/(inv.max() - inv.min())
            invs[i] = inv
        return invs

    def _transform_rols(self, ps, normalize):
        num_pts = len(ps)
        invs = np.zeros((num_pts, self.X.shape[1]))
        for i in range(num_pts):
            p = ps[i]
            diffs = np.linalg.norm(self.Xp_ctrl - p, axis=1)
            if self.normalize_d is True:
                diffs = (diffs - diffs.min())/(diffs.max() - diffs.min())
            diffs = self.kernel(diffs)
            inv = np.dot(self.coefs, diffs)
            if normalize is True:
                inv = (inv - inv.min())/(inv.max() - inv.min())
            invs[i] = inv
        return invs

    def transform(self, ps, normalize=False):
        if self.mode == 'cluster':
            return self._transform_cluster(ps, normalize)
        elif self.mode == 'neighbors':
            return self._transform_neighbors(ps, normalize)
        elif self.mode == 'rols':
            return self._transform_rols(ps, normalize)

    def gaussian(self, r):
        return np.exp(-(self.eps*(r**2.0)))

    def multi_quadrics(self, r):
        return np.sqrt(self.c + self.eps*(r**2))

    def inv_mq(self, r):
        return 1.0/np.sqrt(self.c + self.eps*(r**2))

    def save(self, path):
        data = {}
        data['mode'] = self.mode
        data['num_ctrl'] = self.num_ctrl
        data['kernel'] = self.kernel_str
        data['eps'] = self.eps
        data['c'] = self.c
        data['normalize_c'] = self.normalize_c
        data['normalize_d'] = self.normalize_d

        data['X'] = self.X
        data['Xp'] = self.Xp

        if self.mode == 'cluster':
            data['ctrl_pts'] = self.ctrl_pts
            data['coefs'] = self.coefs
            data['clusters'] = self.clusters
        elif self.mode == 'neighbors':
            data['nbrs'] = self.nbrs
        elif self.mode == 'rols':
            data['ctrl_pts'] = self.ctrl_pts
            data['coefs'] = self.coefs

        joblib.dump(data, path)

    def load(self, path):
        data = joblib.load(path)
        self.mode = data['mode']
        self.num_ctrl = data['num_ctrl']
        self.eps = data['eps']
        self.c = data['c']
        self.normalize_c = data['normalize_c']
        self.normalize_d = data['normalize_d']

        self.X = np.copy(data['X'])
        self.Xp = np.copy(data['Xp'])

        self.kernel_str = data['kernel']

        if self.kernel_str == 'gaussian':
            self.kernel = self.gaussian
        elif self.kernel_str == 'multi_quadrics':
            self.kernel = self.multi_quadrics
        elif self.kernel_str == 'inv_mq':
            self.kernel = self.inv_mq

        if self.mode == 'cluster':
            self.ctrl_pts = data['ctrl_pts']
            self.coefs = data['coefs']
            self.clusters = data['clusters']
            self.X_ctrl = self.X[self.ctrl_pts]
            self.Xp_ctrl = self.Xp[self.ctrl_pts]
        elif self.mode == 'neighbors':
            self.nbrs = data['nbrs']
        elif self.mode == 'rols':
            self.ctrl_pts = data['ctrl_pts']
            self.coefs = data['coefs']
            self.X_ctrl = self.X[self.ctrl_pts]
            self.Xp_ctrl = self.Xp[self.ctrl_pts]

