import numpy as np
from itertools import count


def get_shape_sizes(shapes: list, min_area: int, max_area: int):
    """
        Get shape sizes, such that the related mask
        takes smaller surface than max_area
    """
    sm_sizes = {}
    max_mask_size = 0
    min_mask_size = 0
    for shape in shapes:
        sm_sizes[shape] = {"shape_sizes": [], "mask_dims": []}
        shape_min_mask_size = np.inf
        for size in count(start=shape.min_size, step=shape.min_step):
            mask = shape.create(size)
            area = np.prod(mask.shape)
            if area < min_area:
                continue
            if area > max_area:
                break
            sm_sizes[shape]["shape_sizes"].append(size)
            sm_sizes[shape]["mask_dims"].append(mask.shape[0])
            #
            max_mask_size = max(max_mask_size, mask.shape[0])
            shape_min_mask_size = min(shape_min_mask_size, mask.shape[0])

        min_mask_size = max(min_mask_size, shape_min_mask_size)
    return sm_sizes, max_mask_size, min_mask_size

def get_shape_sizes_evenly_scaled(shapes, min_area, max_area):

    # select shape sizes, s.t mask area < max area
    sm_sizes, max_mask_size, min_mask_size = get_shape_sizes(
        shapes, min_area, max_area)

    # check if all shapes can be crated
    if min_mask_size == np.inf:
        shapes = [shape.name for shape,
                  dct in sm_sizes.items() if len(dct['shape_sizes']) == 0]
        raise Exception(
            "Cannot create masks for shapes {} \
             for given fill rates!".format(shapes))

    # select shape sizes with related mask_size > min_mask_size
    # = sort out shapes that are too small
    all_shape_sizes = {}
    for shape, sm in sm_sizes.items():
        shape_sizes = sm["shape_sizes"]
        mask_dims = sm["mask_dims"]
        #
        sizes = [shape_sizes[idx] for idx in range(
            len(shape_sizes)) if mask_dims[idx] >= min_mask_size]

        # if we could not match, just take the previous found sizes
        # not ideal but rather an edge case, so who cares ;)
        if len(sizes) == 0:
            sizes = shape_sizes
        all_shape_sizes[shape] = sizes

    # equalize number of sizes for the shapes,
    # sucht that scaling happens evenly and smoothly
    num_sizes = min(len(s) for s in all_shape_sizes.values())
    final_sizes = {}
    for shape, sizes in all_shape_sizes.items():
        step = len(sizes) // num_sizes
        sizes = [sizes[idx] for idx in range(0, num_sizes, 1)]
        final_sizes[shape] = sizes
    return final_sizes, max_mask_size
