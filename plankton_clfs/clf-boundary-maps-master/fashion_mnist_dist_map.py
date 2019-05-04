import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.externals import joblib
import dist_maps
import os.path


def plot_dense_map(dmap_hsv):
    dmap_rgb = hsv_to_rgb(dmap_hsv)
    plt.xticks([])
    plt.yticks([])

    extent = [0, dmap_rgb.shape[1], 0, dmap_rgb.shape[0]]
    plt.imshow(dmap_rgb, interpolation='none', origin='lower', extent=extent)
    plt.show()


# def main():
dmap = np.load('data/fm/dmap_tsne_lr.npy')
H, W, _ = dmap.shape
X_train = np.load('data/fm/X_proj.npy')
X_proj = np.load('data/fm/X_tsne_norm.npy')
GRID_SIZE = dmap.shape[0]
clf = joblib.load('data/fm/logistic_reg.joblib')


# check if file exists
dist_nd_path = "data/fm/dist_nd_fixed_new.npy"
if os.path.isfile(dist_nd_path):
    dist_nd = np.load(dist_nd_path)
else:
    # create the dist map if it doesn't exist
    dist_nd = dist_maps.dist_nd(dmap, X_train, X_proj, clf=clf)
    np.save(dist_nd_path, dist_nd)


# same thing for the nd dist map
dist_nd_2_path = "data/fm/dist_nd_2_fixed_new.npy"
if os.path.isfile(dist_nd_2_path):
    dist_nd_2 = np.load(dist_nd_2_path)
else:
    # create the dist map if it doesn't exist
    dist_nd_2 = dist_maps.distance_nd_2(X_train, X_proj, clf, GRID_SIZE)
    np.save(dist_nd_2_path, dist_nd_2)

dist_nd /= dist_nd.max()
dist_nd = 1.0 - dist_nd
dist_nd_2 /= dist_nd_2.max()
dist_nd_2 = 1.0 - dist_nd_2

dmap_new = np.copy(dmap)

print("max saturation: ", dmap_new[:, :, 1].max())
print("min saturation: ", dmap_new[:, :, 1].min())

# dmap_new[:, :, 2] = dist_nd
dmap_new[:, :, 2] = 0.2 + 0.8*dist_nd
dmap_new[:, :, 2] = 0.2 + 0.8*(dist_nd**(1/0.3))

# first attempt
# dmap_new[:, :, 1] *= dist_nd

# second attempt
# dmap_new[:, :, 1] *= dist_nd
# dmap_new[:, :, 1] = np.maximum(0.3, dmap_new[:, :, 1])

# third attempt
# dmap_new[:, :, 1] = 0.2 + 0.8*dist_nd

# fourth attempt
dmap_new[:, :, 1] = (1.0 - dist_nd)*dmap_new[:, :, 1] + dist_nd*(np.maximum(dmap_new[:, :, 1] - 0.2, 0.0))

plt.imshow(hsv_to_rgb(dmap_new))
plt.savefig('dist_map_new.pdf')

dmap_new_2 = np.copy(dmap)
# dmap_new_2[:, :, 2] = 0.2 + 0.8*dist_nd_2
dmap_new_2[:, :, 2] = 0.2 + 0.8*(dist_nd_2**(1/0.3))

# first attempt
# dmap_new_2[:, :, 1] *= dist_nd_2

# second attempt
# dmap_new_2[:, :, 1] *= dist_nd_2
# dmap_new_2[:, :, 1] = np.maximum(0.3, dmap_new_2[:, :, 1])


# third attempt
# dmap_new_2[:, :, 1] = 0.2 + 0.8*dist_nd_2

# fourth attempt
dmap_new_2[:, :, 1] = (1.0 - dist_nd_2)*dmap_new_2[:, :, 1] + dist_nd_2*(np.maximum(dmap_new_2[:, :, 1] - 0.2, 0.0))

plt.imshow(hsv_to_rgb(dmap_new_2))
plt.savefig('dist_map_2_new.pdf')

plt.imshow(hsv_to_rgb(dmap))
plt.savefig('densemap_orig.pdf')
