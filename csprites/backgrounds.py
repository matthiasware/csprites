import random
from itertools import cycle
#
import numpy as np
import matplotlib.pyplot as plt
#
import torchvision.transforms as T
from PIL import Image


def rand_func_img(shape):
    h, w, _ = shape
    dX, dY = w, h
    xArray = np.linspace(0.0, 1.0, dX).reshape((1, dX, 1))
    yArray = np.linspace(0.0, 1.0, dY).reshape((dY, 1, 1))

    def randColor():
        return np.random.random((1, 1, 3))

    def getX():
        return xArray

    def getY():
        return yArray

    def safeDivide(a, b):
        return np.divide(a, np.maximum(b, 0.001))

    functions = [(0, randColor),
                 (0, getX),
                 (0, getY),
                 (1, np.sin),
                 (1, np.cos),
                 (2, np.add),
                 (2, np.subtract),
                 (2, np.multiply),
                 (2, safeDivide)]
    depthMin = 1
    depthMax = 20

    def buildImg(depth=0):
        funcs = [f for f in functions if
                 (f[0] > 0 and depth < depthMax) or
                 (f[0] == 0 and depth >= depthMin)]
        nArgs, func = random.choice(funcs)
        args = [buildImg(depth + 1) for n in range(nArgs)]
        return func(*args)

    img = buildImg()

    # Ensure it has the right dimensions, dX by dY by 3
    img = np.tile(img, (dY // img.shape[0],
                        dX // img.shape[1], 3 // img.shape[2]))

    img = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))
    return img


def color_img(shape, color=(0, 0, 0)):
    img = np.ones(shape, dtype=np.uint8)
    img = np.einsum('ijk,k->ijk', img, color)
    img = img.astype(np.uint8)
    return img


def rand_img(shape):
    img = np.random.random(shape) * 255
    img = img.astype(np.uint8)
    return img


def get_single_bg(shape, style, **kwargs):
    if style == "constant_color":
        return color_img(shape, **kwargs)
    elif style == "random_pixel":
        return rand_img(shape)
    elif style == "random_function":
        return rand_func_img(shape)
    else:
        raise Exception("Unkown style {}".format(style))


def get_cmap(n_colors, name='Spectral'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
       RGB color;
       the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n_colors)


def get_multiple_bg(shape, style, n_bg=8):
    if style == "constant_color":
        cmap = get_cmap(n_bg)
        colors = [cmap(i) for i in range(n_bg)]
        colors = [(np.array(c[:3]) * 255).astype(np.uint8) for c in colors]
        return [color_img(shape, color) for color in colors]
    elif style == "random_pixel":
        return [rand_img(shape) for _ in range(n_bg)]
    elif style == "random_function":
        return [rand_func_img(shape) for _ in range(n_bg)]
    else:
        raise Exception("Unkown style {}".format(style))


def get_infinite_bg(shape, style):
    if style == "constant_color":
        min_val = 0
        max_val = 1000000
        cmap = get_cmap(max_val)

        def func():
            idx = np.random.randint(min_val, max_val)
            c = cmap(idx)
            c = (np.array(c[:3]) * 255).astype(np.uint8)
            return color_img(shape, c)
        return func
    elif style == "random_pixel":
        def func():
            return rand_img(shape)
        return func
    elif style == "random_function":
        def func():
            return rand_func_img(shape)
        return func
    else:
        raise Exception("Unkown style {}".format(style))


def get_bg_func(shape, n_bg, bg_style):
    if n_bg == 1:
        bg = get_single_bg(shape, bg_style)

        def func():
            return np.copy(bg)
    elif n_bg > 1 and n_bg != np.inf:
        bgs = get_multiple_bg(shape, bg_style, n_bg)
        bgs = cycle(bgs)

        def func():
            return np.copy(next(bgs))
    elif n_bg == np.inf:
        func = get_infinite_bg(shape, bg_style)
    else:
        raise Exception("BLA")
    return func


def safe_divide(a, b):
    return np.divide(a, np.maximum(b, 0.001))


def scale(x):
    return safe_divide(x - x.min(), x.max() - x.min())


def rand_sin(x):
    if np.random.random() > 0.5:
        freq = np.random.uniform(-8, 8)
    else:
        freq = np.random.uniform(-1, 1)
    add = np.random.uniform(0, 2 * np.pi)
    return np.sin(freq * x + add) * np.random.uniform(-10, 10)


def randColor():
    return np.random.random((1, 1, 3))


def getX(dX):
    if np.random.random() > 0.5:
        return np.linspace(0.0, 1.0, dX).reshape((1, dX, 1))
    else:
        return np.linspace(1.0, 0.0, dX).reshape((1, dX, 1))


def getY(dY):
    if np.random.random() > 0.5:
        return np.linspace(0.0, 1.0, dY).reshape((dY, 1, 1))
    else:
        return np.linspace(1.0, 0.0, dY).reshape((dY, 1, 1))


def rand_init_y(dY):
    ax = getY(dY)
    c = randColor()

    img = np.add(ax, c)
    return img


def rand_init_x(dX):
    ax = getX(dX)
    c = randColor()

    img = np.add(ax, c)
    return img


def basic_img(dX, dY):
    imgx = rand_init_x(dX)
    imgy = rand_init_y(dY)
    #
    f = random.choice([np.add, np.subtract, np.multiply, safe_divide])
    img = f(imgx, imgy)

    img = rand_sin(img)
    return img


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def rand_function_img(shape, n_min=3, n_max=3, add_min=-0.5, add_max=0.5):
    h, w, _ = shape
    dX, dY = w, h
    #
    n = np.random.randint(n_min, n_max + 1)
    #
    f = random.choice([np.prod, np.mean, np.sum])
    img = f([basic_img(dX, dY) for _ in range(n)], axis=0)
    img = scale(img)
    if np.random.random() > 0.5:
        add = np.random.uniform(add_min, add_max, 3)
        img = img + add
        img = img.clip(0, 1)
    elif np.random.random() > 0.5:
        add = np.random.uniform(0.1, 2, 3)
        img = img * add
        img = img.clip(0, 1)

    img = np.tile(img, (dY // img.shape[0],
                        dX // img.shape[1], 3 // img.shape[2]))
    img = np.uint8(np.rint(img.clip(0.0, 1.0) * 255.0))

    if img.max() - img.min() < 20:
        img = rand_function_img(shape)
    jitter = T.ColorJitter(brightness=(
        0.9, 1.15), hue=(-0.5, 0.5), saturation=(0.4, 1.3), contrast=(.5, 2))
    img = np.array(jitter(Image.fromarray(img)))
    return img
