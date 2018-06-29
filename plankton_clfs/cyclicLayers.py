from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

def array_tf_0(arr):
    return arr

def array_tf_90(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None), slice(None, None, -1)]
    return arr[tuple(slices)].transpose(axes_order)

def array_tf_180(arr):
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None, None, -1)]
    return arr[tuple(slices)]

def array_tf_270(arr):
    axes_order = range(arr.ndim - 2) + [arr.ndim - 1, arr.ndim - 2]
    slices = [slice(None) for _ in range(arr.ndim - 2)] + [slice(None, None, -1), slice(None)]
    return arr[tuple(slices)].transpose(axes_order)


class CyclicSliceLayer(Layer):
    """
    This layer stacks rotations of 0, 90, 180, and 270 degrees of the input
    along the batch dimension.

    If the input has shape (batch_size, num_channels, r, c),
    then the output will have shape (4 * batch_size, num_channels, r, c).

    Note that the stacking happens on axis 0, so a reshape to
    (4, batch_size, num_channels, r, c) will separate the slice axis.
    """

    def __init__(self, in_shape, **kwargs):
        self.output_dim = (4*in_shape[0],) + in_shape[1:]
        super(CyclicSliceLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=False)
        super(CyclicSliceLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.concatenate([
                array_tf_0(x),
                array_tf_90(x),
                array_tf_180(x),
                array_tf_270(x),
            ], axis=0)

    def compute_output_shape(self, input_shape):
        return (4*input_shape[0],) + input_shape[1:]
        
class CyclicRollLayer(Layer):
    """
    This layer turns (n_views * batch_size, num_features) into
    (n_views * batch_size, n_views * num_features) by rolling
    and concatenating feature maps.
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CyclicRollLayer, self).__init__(**kwargs)
        self.compute_permutation_matrix()

    def compute_permutation_matrix(self):
        map_identity = np.arange(4)
        map_rot90 = np.array([1, 2, 3, 0])

        valid_maps = []
        current_map = map_identity
        for k in xrange(4):
            valid_maps.append(current_map)
            current_map = current_map[map_rot90]

        self.perm_matrix = np.array(valid_maps)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(CyclicRollLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        s = x.shape
        input_unfolded = x.reshape((4, s[0] // 4, s[1]))
        permuted_inputs = []
        for p in self.perm_matrix:
            input_permuted = input_unfolded[p].reshape(s)
            permuted_inputs.append(input_permuted)
        return K.concatenate(permuted_inputs, axis=0)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4*input_shape[1])
        
class CyclicConvRollLayer(CyclicRollLayer):
    """
    This layer turns (n_views * batch_size, num_channels, r, c) into
    (n_views * batch_size, n_views * num_channels, r, c) by rolling
    and concatenating feature maps.

    It also applies the correct inverse transforms to the r and c
    dimensions to align the feature maps.
    """

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CyclicConvRollLayer, self).__init__(**kwargs)
        self.inv_tf_funcs = [array_tf_0, array_tf_270, array_tf_180, array_tf_90]

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(CyclicConvRollLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        s = x.shape
        input_unfolded = x.reshape((4, s[0] // 4, s[1], s[2], s[3]))
        permuted_inputs = []
        for p, inv_tf in zip(self.perm_matrix, self.inv_tf_funcs):
            input_permuted = inv_tf(input_unfolded[p].reshape(s))
            permuted_inputs.append(input_permuted)
        return K.concatenate(permuted_inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return ((input_shape[0], 4*input_shape[1]) + input_shape[2:])
        
        
class CyclicPoolLayer(Layer):
    """
    Utility layer that unfolds the viewpoints dimension and pools over it.

    Note that this only makes sense for dense representations, not for
    feature maps (because no inverse transforms are applied to align them).
    """
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(CyclicPoolLayer, self).__init__(**kwargs)
        self.pool_function = K.mean

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(CyclicPoolLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        unfolded_input = input.reshape((4, x.shape[0] // 4, x.shape[1]))
        return self.pool_function(unfolded_input, axis=0)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
