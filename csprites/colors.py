import numpy as np
import matplotlib.pyplot as plt

# COLORS = np.array(
#     [[255, 51, 51],  # red
#      [153, 255, 51],  # green
#      [51, 153, 255],  # blue-teal
#      [255, 255, 51],  # yellow
#      [255, 51, 255],  # pink
#      [255, 153, 51],  # orange
#      [51, 255, 255],  # teal
#      [153, 51, 255],  # purple
#      [51, 255, 153],  # green-teal
#      [51, 51, 255],   # blue
#      [255, 51, 153],  # magenta
#      ], dtype=np.uint8)


def apply_color(mask, color):
    assert len(mask.shape) == 2
    assert color.shape == (3,)
    #
    img = np.stack((mask, mask, mask), axis=2)
    img = np.einsum('ijk,k->ijk', img, color)
    return img


def get_cmap(n_colors, name='Spectral'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
       RGB color;
       the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n_colors)


def get_colors(n_colors, cmap="nipy_spectral"):
    cmap = get_cmap(n_colors, name=cmap)
    colors = [cmap(i) for i in range(n_colors)]
    colors = [(np.array(c[:3]) * 255).astype(np.uint8) for c in colors]
    return colors
