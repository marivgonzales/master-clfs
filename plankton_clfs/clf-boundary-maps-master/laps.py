from boundarymap import CLF
from boundarymap import Grid
from boundarymap import PlotDenseMap
from boundarymap import PlotProjection

import numpy as np
np.random.seed(0)


def main():
    base_dir = '/home/caio/workspace/laps/'
    proj_tsne = np.load(base_dir + 'laps_X_proj_tsne.npy')

    min0 = proj_tsne[:, 0].min()
    min1 = proj_tsne[:, 1].min()
    max0 = proj_tsne[:, 0].max()
    max1 = proj_tsne[:, 1].max()
    proj_tsne[:, 0] = (proj_tsne[:, 0] - min0)/(max0 - min0)
    proj_tsne[:, 1] = (proj_tsne[:, 1] - min1)/(max1 - min1)

    X_train = np.load(base_dir + 'laps_X_train.npy')
    # y_train = np.load('data/fm/y_proj.npy')

    clf1 = CLF()
    clf1.LoadSKLearn(base_dir + 'lr_clf.joblib', "LR")
    # Plots the projected points coulored according to the label assigned by
    # the classifier.
    # As it is the first projection plotted, the legend is also save into a
    # separate file
    y_pred = clf1.Predict(X_train)
    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    path = base_dir + "projection_clf1_tsne.pdf"
    path_leg = base_dir + "projection_leg.pdf"
    title = "t-SNE Projection"
    PlotProjection(proj_tsne, y_pred, path, title, path_leg, labels)

    # Run boundary map construction function
    R = 200
    N = [5]
    grid1 = Grid(proj_tsne, R)

    _, dmap = grid1.BoundaryMap(X_train, N[0], clf1)

    np.save(base_dir + 'dmap_tsne_lr_R_50_N_10.npy', dmap)


if __name__ == "__main__":
    main()
