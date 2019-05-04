import numpy as np
import joblib
from matplotlib.colors import hsv_to_rgb


# Linear interpolation between a and b
def Lerp(a, b, t):
    return (1.0 - t)*a + t*b


def TransferFunc(hsv, k):
    a = 0.0
    b = 1.0
    new_img = np.copy(hsv)
    new_img[:, :, 2] = (a + b*new_img[:, :, 2])**k
    return new_img


def SampleSquare(num_samples, limits):
    pts = []
    for i in range(num_samples):
        x = np.random.uniform(low=limits[0], high=limits[2])
        y = np.random.uniform(low=limits[1], high=limits[3])
        pts.append([x, y])
    return pts


# samples num_samples x num_samples equally spaced in a square
def SampleSquareUniform(num_samples, limits):
    pts = []
    x_pts = np.linspace(limits[0], limits[2], num_samples)
    y_pts = np.linspace(limits[1], limits[3], num_samples)

    for y in y_pts[::-1]:
        for x in x_pts:
            pts.append([x, y])
    return pts


def LoadProjection(path):
    if path.split('.')[-1] == 'npy':
        proj = np.load(path)
    else:
        proj_dict = joblib.load(open(path, 'rb'))
        key = next(iter(proj_dict))
        proj = proj_dict[key]['X']
    return proj


def HSV2RGB(dmap_hsv):
    tmp_dense = TransferFunc(dmap_hsv, 0.7)
    rgb_img = hsv_to_rgb(tmp_dense)
    return rgb_img


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


def bmap_to_categorical(bmap_hsv):
    # The boundary map stores a separate hue value for each class.
    # Those specific hues will be mapped to categorical colors from color
    # brewer.
    HUES = np.array([216, 18, 126, 306, 270, 90, 198, 342, 54, 162])/360.0

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

    W, H, C = bmap_hsv.shape
    C = 4
    new_bmap = np.zeros((W, H, C))
    for i in range(W):
        for j in range(H):
            c_idx = np.where(HUES == bmap_hsv[i, j, 0])
            new_bmap[i, j] = COLORS[c_idx]

    return new_bmap


def bmap_to_categorical_base(bmap_base):
    # The boundary map stores a separate hue value for each class.
    # Those specific hues will be mapped to categorical colors from color
    # brewer.
    COLORS = np.array([[166, 206, 227, 255],
                       [31,  120, 180, 255],
                       [178, 223, 138, 255],
                       [51,  160,  44, 255],
                       [251, 154, 153, 255],
                       [227, 26,   28, 255],
                       [253, 191, 111, 255],
                       [255, 127,   0, 255],
                       [202, 178, 214, 255],
                       [106, 61,  154, 255],
                       [127, 127, 127, 127]])/255.0

    W, H, C = bmap_base.shape
    C = 4
    new_bmap = np.zeros((W, H, C))
    for i in range(W):
        for j in range(H):
            c_idx = int(bmap_base[i, j, 0])
            if c_idx > 10:
                print('diff idx: ', c_idx, i, j)
                c_idx = 10
            new_bmap[i, j] = COLORS[c_idx]

    return new_bmap

