import matplotlib as mpl
mpl.use('Agg')

from keras.applications import VGG16
from vis.utils import utils
from keras import activations

from vis.visualization import visualize_saliency, overlay

import tensorflow as tf 
print(tf.__version__)
# Build the VGG16 network with ImageNet weights
model = VGG16(weights='imagenet', include_top=True)

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'predictions')

# Swap softmax with linear
model.layers[layer_idx].activation = activations.linear
model = utils.apply_modifications(model)


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (18, 6)

img1 = utils.load_img('images/36693.jpg', target_size=(95, 95))
img2 = utils.load_img('images/51085.jpg', target_size=(95, 95))
"""
f, ax = plt.subplots(1, 2)
ax[0].imshow(img1)
ax[1].imshow(img2)
"""

# Utility to search for layer index by name. 
# Alternatively we can specify this as -1 since it corresponds to the last layer.
layer_idx = utils.find_layer_idx(model, 'predictions')

f, ax = plt.subplots(1, 2)
for i, img in enumerate([img1, img2]):    
    # 20 is the imagenet index corresponding to `ouzel`
    grads = visualize_saliency(model, layer_idx, filter_indices=20, seed_input=img)
    
    # visualize grads as heatmap
    ax[i].imshow(grads, cmap='jet')

f.savefig('att.png')
