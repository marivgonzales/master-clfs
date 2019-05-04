from boundarymap import Grid
from boundarymap import build_grid
import numpy as np
from utils import bmap_to_categorical
import matplotlib.pyplot as plt


def bmap_errs(grid_path, clf_path, inv_proj_path):
    g = Grid()
    g.load(grid_path, clf_path, inv_proj_path)

    xmin = np.min(g.X_2d[:, 0]) - 1e-6
    xmax = np.max(g.X_2d[:, 0]) + 1e-6
    ymin = np.min(g.X_2d[:, 1]) - 1e-6
    ymax = np.max(g.X_2d[:, 1]) + 1e-6
    x_intrvls = np.linspace(xmin, xmax, num=g.R + 1)
    y_intrvls = np.linspace(ymin, ymax, num=g.R + 1)
    g.cells = build_grid(g.X_2d, g.R, x_intrvls, y_intrvls)

    # FIXME: it was saved in a file already
    y = g.clf.Predict(g.X_nd)

    errs = 0
    for i in range(g.R):
        for j in range(g.R):
            if len(g.cells[i][j]) == 0:
                continue
            pts_in_cell = g.cells[i][j]
            for p in pts_in_cell:
                if g.dmap[i, j, 0] != g.CMAP_SYN[y[p]]:
                    errs += 1
    return errs


def bmap_errs2(grid_path, clf_path, inv_proj_path, bmap_title, bmap_path):
    COLORS = np.array([[166, 206, 227, 255],
                       [31,  120, 180, 255],
                       [178, 223, 138, 255],
                       [51,  160,  44, 255],
                       [251, 154, 153, 255],
                       [227, 26,   28, 255],
                       [253, 191, 111, 255],
                       [255, 127,   0, 255],
                       [202, 178, 214, 255],
                       [106, 61,  154, 255]])/255.0

    g = Grid()
    g.load(grid_path, clf_path, inv_proj_path)

    xmin = np.min(g.X_2d[:, 0]) - 1e-6
    xmax = np.max(g.X_2d[:, 0]) + 1e-6
    ymin = np.min(g.X_2d[:, 1]) - 1e-6
    ymax = np.max(g.X_2d[:, 1]) + 1e-6
    x_intrvls = np.linspace(xmin, xmax, num=g.R + 1)
    y_intrvls = np.linspace(ymin, ymax, num=g.R + 1)
    g.cells = build_grid(g.X_2d, g.R, x_intrvls, y_intrvls)

    # FIXME: it was saved in a file already
    y = g.clf.Predict(g.X_nd)

    bmap_rgb = bmap_to_categorical(g.dmap)
    bmap_rgb[:, :, 3] = 0.3
    errs = 0
    errs_pts = []
    errs_colors = []
    for row in range(g.R):
        for col in range(g.R):
            if len(g.cells[row][col]) == 0:
                continue
            labels = y[g.cells[row][col]]
            counts = np.bincount(labels)
            color_idx = np.argmax(counts)
            hue = g.CMAP_SYN[color_idx]
            if g.dmap[row, col, 0] != hue:
                errs += 1
                bmap_rgb[row, col, 3] = 1.0
                errs_pts.append([col + 0.5, row + 0.5])
                errs_colors.append(COLORS[color_idx])

    errs_pts = np.array(errs_pts)
    plot_grid(bmap_path, bmap_rgb, errs_pts, errs_colors,
              bmap_title + ' ({})'.format(errs))
    return errs, bmap_rgb


def plot_grid(fig_path, bmap_rgb, errs_pts, errs_colors, title):
    plt.clf()
    plt.xticks([])
    plt.yticks([])

    plt.imshow(bmap_rgb, interpolation='none', origin='lower')
    if len(errs_pts) > 0:
        plt.scatter(errs_pts[:, 0], errs_pts[:, 1], color=errs_colors, s=0.2)
    plt.title(title)
    plt.savefig(fig_path, figsize=(20, 20))
    plt.clf()


inv_projs = ['ilamp', 'irbfc', 'irbfcp', 'nninv']
projs = ['tsne', 'umap']

# MNIST_SYN
base_dir = 'data/mnist_syn/'
grid_paths = []
inv_proj_paths = []

clf_path = base_dir + 'mnist_cnn.json'

for ip in inv_projs:
    for p in projs:
        grid_path = base_dir + 'grid_{}_{}.joblib'.format(ip, p)
        inv_proj_path = base_dir + '{}_{}.joblib'.format(ip, p)
        bmap_title = 'MNIST - Grid Errs {} {}'.format(ip, p)
        bmap_path = 'figs/errs/mnist_{}_{}.pdf'.format(ip, p)
        n_errs, bmap_rgb = bmap_errs2(grid_path, clf_path, inv_proj_path,
                                      bmap_title, bmap_path)
        print("{}: {}".format(grid_path, n_errs))


# FASHION MNIST SYN
base_dir = 'data/fm_syn/'
grid_paths = []
inv_proj_paths = []

clf_path = base_dir + 'fm_cnn.json'

for ip in inv_projs:
    for p in projs:
        grid_path = base_dir + 'grid_{}_{}.joblib'.format(ip, p)
        inv_proj_path = base_dir + '{}_{}.joblib'.format(ip, p)
        bmap_title = 'FashionMNIST- Grid Errs {} {}'.format(ip, p)
        bmap_path = 'figs/errs/fm_{}_{}.pdf'.format(ip, p)
        n_errs, bmap_rgb = bmap_errs2(grid_path, clf_path, inv_proj_path,
                                      bmap_title, bmap_path)
        print("{}: {}".format(grid_path, n_errs))


# BLOBS_SYN
base_dir = 'data/blobs_syn/'
grid_paths = []
inv_proj_paths = []

clf_path = base_dir + 'lr.json'

for ip in inv_projs:
    for p in projs:
        grid_path = base_dir + 'grid_{}_{}.joblib'.format(ip, p)
        inv_proj_path = base_dir + '{}_{}.joblib'.format(ip, p)
        bmap_title = 'Blobs - Grid Errs {} {}'.format(ip, p)
        bmap_path = 'figs/errs/blobs_{}_{}.pdf'.format(ip, p)
        n_errs, bmap_rgb = bmap_errs2(grid_path, clf_path, inv_proj_path,
                                      bmap_title, bmap_path)
        print("{}: {}".format(grid_path, n_errs))

