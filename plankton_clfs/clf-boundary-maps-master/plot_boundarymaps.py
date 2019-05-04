from boundarymap import Grid
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from utils import bmap_to_categorical
import numpy as np
import os


def plot_grid(fig_path, bmap_rgb, title):
    plt.clf()
    plt.xticks([])
    plt.yticks([])

    # plt.imshow(bmap_to_categorical(grid.dmap), interpolation='none',
    #            origin='lower')

    plt.imshow(bmap_rgb, interpolation='none', origin='lower')
    plt.title(title)
    plt.savefig(fig_path)
    plt.clf()


def plot_legend(path, colors, labels):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    handles = []
    for c in colors:
        handles.append(ax.scatter([], [], color=c))

    figlegend = plt.figure()
    figlegend.legend(handles, labels, 'center')
    figlegend.savefig(path, format='pdf')
    plt.clf()


# def plot_projection(gpath, y_pred_path, proj_path, title,
#                     leg_path="", labels=[]):
def plot_projection(fig_path, proj, y_pred, title, labels=None):
    # COLORS are the rgb from color brewer
    COLORS = np.array([[166, 206, 227],
                       [31, 120, 180],
                       [178, 223, 138],
                       [51, 160, 44],
                       [251, 154, 153],
                       [227, 26, 28],
                       [253, 191, 111],
                       [255, 127, 0],
                       [202, 178, 214],
                       [106, 61, 154]])/360.0

    colors = [COLORS[i] for i in y_pred]

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(proj[:, 0], proj[:, 1], color=colors, s=10.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    if labels is not None:
        handles = []
        for i in range(len(labels)):
            handles.append(ax.scatter([], [], color=COLORS[i]))
        ax.legend(handles, labels, loc="lower left", borderaxespad=0.0,
                  bbox_to_anchor=(1.05, 0), fontsize="small")

    fig.savefig(fig_path)
    plt.clf()

    # if leg_path != "":
    #     plot_legend(leg_path, COLORS[:len(labels)], labels)


def plot_misclf(fig_path, bmap_rgb, misclf, title):
    R = bmap_rgb.shape[0]
    # misclf = proj[y_true != y_pred]
    # print('NUM MISCLF: ', len(misclf))

    plt.clf()
    plt.xticks([])
    plt.yticks([])

    plt.axes().set_aspect('equal')
    plt.imshow(bmap_rgb, interpolation='none', origin='lower')

    plt.scatter(R*misclf[:, 0], R*misclf[:, 1], s=5.0, linewidth=0.0,
                color=[1.0, 1.0, 1.0, 0.7])

    # plt.scatter(100*proj[y_pred != y_true][:, 0],
    #             100*(1.0 - proj[y_pred != y_true][:, 1]),
    #             color=colors[y_pred != y_true], s=3.0,
    #             edgecolor=edge_colors[y_pred != y_true], linewidth=0.5)

    plt.title(title)
    plt.savefig(fig_path)
    plt.clf()


def plot(base_dir, base_dest, data_name, labels=None):
    grid_names = ['grid_ilamp_tsne.joblib', 'grid_ilamp_umap.joblib',
                  'grid_irbfcp_tsne.joblib', 'grid_irbfcp_umap.joblib',
                  'grid_irbfc_tsne.joblib', 'grid_irbfc_umap.joblib',
                  'grid_nninv_tsne.joblib', 'grid_nninv_umap.joblib']
    fig_names = ['ilamp_tsne', 'ilamp_umap',
                 'irbfcp_tsne', 'irbfcp_umap',
                 'irbfc_tsne', 'irbfc_umap',
                 'nninv_tsne', 'nninv_umap']

    title_names = ['ILAMP (tSNE)', 'ILAMP(UMAP)',
                   'RBF Inv (fixed/tSNE)', 'RBF Inv (fixed/UMAP)',
                   'RBF Inv (clusters/tSNE)', 'RBF Inv (clusters/UMAP)',
                   'Neural Network (tSNE)', 'Neural Network (UMAP)']

    y_pred = np.load(base_dir + 'y_pred_clf.npy')
    y_true = np.load(base_dir + 'y_train.npy')
    misclf_idx = y_true != y_pred
    num_misclf = np.sum(misclf_idx)
    print("dataset: ", data_name, ' num misclf: ', num_misclf)
    for i in range(len(grid_names)):
        grid_name = grid_names[i]
        fig_name = fig_names[i]
        title = title_names[i]

        if os.path.exists(base_dir + grid_name) is False:
            continue

        grid = Grid()
        grid.load(base_dir + grid_name)

        bmap_rgb = bmap_to_categorical(grid.dmap)

        bmap_title = data_name + ' - ' + title + ' Boundary Map'
        plot_grid(base_dest + fig_name + '.pdf', bmap_rgb, bmap_title)

        if num_misclf > 0:
            misclf_path = base_dest + fig_name + '_misclf.pdf'
            plot_misclf(misclf_path, bmap_rgb, grid.X_2d[misclf_idx],
                        title + ' Misclassifications')
        if i == 0:
            title = data_name + ' tSNE Projection'
            path = base_dest + 'tsne_proj.pdf'
            plot_projection(path, grid.X_2d, y_pred, title, labels)
            # TODO: plot legend
        elif i == 1:
            title = data_name + ' UMAP Projection'
            path = base_dest + 'umap_proj.pdf'
            plot_projection(path, grid.X_2d, y_pred, title, labels)


# FASHION MNIST
labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plot('data/fm_tsne_umap/', 'figs/fm_syn_', 'Fashion MNIST', labels)

# MNIST
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
plot('data/mnist_500/', 'figs/mnist_syn_', 'MNIST', labels)

# BLOBS
labels = ['0', '1', '2', '3', '4']
plot('data/blobs_syn/', 'figs/blobs_syn_', 'BLOBS', labels)
