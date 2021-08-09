import numpy as np
from skimage.draw import disk, ellipse


class RectangleMask(object):
    """
      Create mask for rectangle of aspect ratio
       height : width = 2 : 1

      Note that:
      if fixed aspect ratio of e.g 2:1:
        h  w
        -----
        3  1.5 -> 1 -> 3:1
        5  2.5 -> 2 -> 5:2
        7  3.5 -> 3 -> 7:3

      Therefor scale up lower dimension by +2
      and calculate other dimension from that

        w  h
        ----
        3  7
        5  11
        7  15
        -> this converges to certain aspect ratio!

        Also note that we always want a discrete center point
        and therefore the dimensions of the mask are uneven
        This allows even padding, that is necessary for scaling.

        Also note that in order to allow a smooth roation
        the shape mask must be quadrativ
    """
    min_size = 5
    min_step = 2
    rot_variance = (0, 180)
    name = "rectangle"

    @staticmethod
    def create_rectangle(height, width):
        assert width % 2 == 1
        assert height % 2 == 1

        transpose = False
        if height < width:
            height, width = width, height
            transpose = True

        # minimum dimension of mask to ensure smooth roations
        diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))
        #
        # add zero colums on top and bottom plus
        # extra col if diagonal is even, which we dont want!
        mask_height = diagonal + 2 + (1 - diagonal % 2)
        mask_width = mask_height
        #
        img = np.zeros((mask_height, mask_width), dtype=np.uint8)

        diff_y = mask_height - height
        diff_x = mask_width - width
        #
        assert diff_y % 2 == 0
        assert diff_x % 2 == 0
        #
        x_from = (diff_x // 2)
        x_to = x_from + width
        y_from = (diff_y // 2)
        y_to = y_from + height
        #
        img[y_from:y_to, x_from:x_to] = 1
        #
        if transpose:
            img = img.T101
        return img

    @staticmethod
    def create(width):
        assert width % 2 == 1
        height = width * 2 + 1
        mask = RectangleMask.create_rectangle(height, width)
        return mask


class SquareMask(object):
    min_size = 9
    min_step = 2
    rot_variance = (0, 90)
    name = "square"

    @staticmethod
    def create(width):
        return RectangleMask.create_rectangle(width, width)


class CircleMask(object):
    min_size = 11
    min_step = 2
    rot_variance = (0, 0)
    name = "circle"

    @staticmethod
    def create(diameter):
        assert diameter % 2 == 1
        #
        radius = diameter // 2 + 1
        # plus 2 px at each side to ensure smooth rotation
        d_img = diameter + 4

        center = (d_img // 2, d_img // 2)
        rr, cc = disk(center, radius)
        #
        img = np.zeros((d_img, d_img), dtype=np.uint8)
        img[rr, cc] = 1

        assert img.shape[0] == img.shape[1]
        return img


class EllipseMask(object):
    min_size = 5
    min_step = 2
    rot_variance = (0, 180)
    name = "ellipse"

    @staticmethod
    def create_ellipse(diameter_x, diameter_y):

        radius_x = diameter_x // 2 + 1
        radius_y = diameter_y // 2 + 1

        # plus 2 px at each side to ensure smooth rotation
        d_img = max(diameter_x, diameter_y) + 4

        # center = (d_img // 2, d_img // 2)
        rr, cc = ellipse(d_img // 2, d_img // 2, radius_y, radius_x)
        #
        img = np.zeros((d_img, d_img), dtype=np.uint8)
        img[rr, cc] = 1

        assert img.shape[0] == img.shape[1]
        return img

    @staticmethod
    def create(diameter):
        assert diameter % 2 == 1

        diameter_x = diameter
        diameter_y = diameter_x * 2 + 1
        return EllipseMask.create_ellipse(diameter_x, diameter_y)


class MoonMask(object):
    min_size = 7
    min_step = 2
    rot_variance = (0, 360)
    name = "moon"

    @staticmethod
    def create(radius):
        assert radius % 2 == 1

        diameter = radius * 2 - 1

        # get circle, offset = 2px
        m_circle = CircleMask.create(diameter)

        x_from = 2
        x_to = x_from + diameter

        # ensure our half circle has uneven height
        if radius % 2 == 0:
            y_from = 2 + radius
        else:
            y_from = 2 + radius - 1
        y_to = 2 + diameter
        #
        m_half_circle = m_circle[y_from:y_to, x_from:x_to]
        #
        assert m_half_circle.shape == (radius - (1 - radius % 2), diameter)
        #
        height, width = m_half_circle.shape
        diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))
        #
        # (height = width = dim)
        mask_dim = diagonal + 2 + (1 - diagonal % 2)

        # new mask
        mask = np.zeros((mask_dim, mask_dim), dtype=np.uint8)
        #
        #
        diff_y = mask_dim - height
        diff_x = mask_dim - width
        #
        assert diff_y % 2 == 0
        assert diff_x % 2 == 0
        #
        x_from = (diff_x // 2)
        x_to = x_from + width
        y_from = (diff_y // 2)
        y_to = y_from + height
        #
        mask[y_from:y_to, x_from:x_to] = m_half_circle
        assert mask.shape[0] == mask.shape[1]
        return mask


class GelatoMask(object):
    min_size = 11
    min_step = 2
    rot_variane = (0, 360)
    name = "gelato"

    @staticmethod
    def create(diameter):
        mask = CircleMask.create(diameter)
        radius = diameter // 2 + 1

        # delete half of the circle
        mask[2 + radius:, :] = 0

        pyramid_height = radius
        pyramid_width = 2 * pyramid_height - 1

        for step in range(pyramid_height):
            row_idx = 2 + radius - 1 + step
            x_from = 2 + step
            x_to = x_from + pyramid_width - (2 * step)
            mask[row_idx, x_from: x_to] = 1
        return mask


class PyramidMask(object):
    min_size = 7
    min_step = 2
    rot_variance = (0, 360)
    name = "pyramid"

    @staticmethod
    def create(height):
        assert height % 2 == 1
        width = 2 * height - 1
        #
        diagonal = int(np.ceil(np.sqrt(height**2 + width**2)))
        # add zero colums on top and bottom
        # plus extra col if diagonal is even, which we dont want!
        mask_height = diagonal + 2 + (1 - diagonal % 2)
        mask_width = mask_height
        #
        mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
        diff_y = mask_height - height
        diff_x = mask_width - width
        #
        assert diff_y % 2 == 0
        assert diff_x % 2 == 0
        #
        x_offset = (diff_x // 2)
        y_offset = (diff_y // 2)
        #
        for step in range(height):
            row_idx = y_offset + step
            x_from = x_offset + step
            x_to = x_from + width - (2 * step)
            mask[row_idx, x_from: x_to] = 1
        return mask


SHAPES = [RectangleMask, SquareMask, CircleMask,
          GelatoMask, PyramidMask, EllipseMask,
          MoonMask]


def get_shapes(shape_names=None):
    if shape_names is None:
        return SHAPES
    else:
        shape_names = set(shape_names)
        return [s for s in SHAPES if s.name in shape_names]
