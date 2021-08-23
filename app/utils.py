'''
Utility Functions
'''


import numpy as np
import pickle
import os
from dotted_dict import DottedDict
import json
from itertools import count

import sys
sys.path.insert(0,'..')
from csprites.scales import get_shape_sizes_evenly_scaled
from csprites.positions import get_max_positions
from csprites.masks import SHAPES


def create_dataset(p):
    conf = create_config_dict(p)
    with open('current_config.json', 'w') as fp:
        json.dump(conf, fp, indent=4)

def create_config_dict(p):
    # TODO: translate process params into config dict
    ret = DottedDict()

    if p.nmb_backgrounds == 0:
        ret['n_bg'] = np.inf
    else:
        ret['n_bg'] = p.nmb_backgrounds
    ret['bg_style'] = p.bg_mode
    ret['shapes'] = p.selected_shapes
    ret["n_scales"] = p.scales
    ret["n_angles"] = p.nmb_rotations
    ret["n_colors"] = p.colors
    ret["n_positions"] = p.positions
    ret["img_size"] = int(2**p.img_size)

    n_samples = p.n_samples
    test_perc = p.test_perc
    if test_perc>0 and test_perc<1:
        generate_subset = True
        n_valid = int(test_perc*n_samples)
        n_train = n_samples-n_valid
    else:
        generate_subset = False
        if test_perc==1:
            n_valid = n_samples
            n_train = 0
        else:
            n_valid = 0
            n_train = n_samples
    ret["n_train"] = n_train
    ret["n_valid"] = n_valid
    ret["subset"] = generate_subset

    target_bbox = p.target_bbox
    target_segm = p.target_seg
    ret["target_bbox"] = target_bbox
    ret["target_segm"] = target_segm

    ret["min_mask_fill_rate"] = p.min_fillrate 
    ret["max_mask_fill_rate"] = p.max_fillrate  
    return ret

def recalculate_params(p, init=False):
    # init number of scales
    min_mask_area = (2**p.img_size)**2 * p.min_fillrate
    max_mask_area = (2**p.img_size)**2 * p.max_fillrate
    shapes = [shape_dict[i] for i in p.selected_shapes]
    try:
        shape_sizes, max_mask_size = get_shape_sizes_evenly_scaled(shapes, min_mask_area, max_mask_area)
        n_scales = len(list(shape_sizes.values())[0])
        p['max_scales'] = n_scales
        if init:
            p['scales'] = p.max_scales
        else:
            p['scales'] = min(p['scales'], p.max_scales)
    except:
        p['max_scales'] = 0
        p['scales'] = p.max_scales
    
    # init number of positions
    try:
        p.max_positions = get_max_positions((2**p.img_size), max_mask_size)
        if init:
            p['positions'] = p.max_positions
        else:
            p['positions'] = min(p['positions'], p.max_positions)
    except:
        p.max_positions = 0
        p['positions'] = p.max_positions

    # init number of states, masks, memory usage
    p.n_masks = len(shapes) * p.colors * p.nmb_rotations * p.scales
    p.n_states = p.n_masks * p.positions**2
    if init:
        p.n_samples = p.n_states
    else:
        p.n_samples = min(p.n_samples, p.n_states)
    p.mem_usage = round((2**p.img_size)**2 * 3 * p.n_samples * 1e-9, 2)
    return p

def get_scales(p):
    shapes = shape_dict.keys()
    max_size = (2**p.max_img_size)**2
    res = {}
    for shape in shapes:
        res[shape] = {  'area': [],
                        'mask_size' : []}
        s = shape_dict[shape]
        for size in count(start=s.min_size, step=s.min_step):
            mask = s.create(size)
            area = int(np.prod(mask.shape))
            if area>max_size:
                break
            else:
                res[shape]['area'].append(area)
                res[shape]['mask_size'].append(mask.shape[0])
    return res


'''
Initial Values
'''


with open('static/config.json', 'r') as fp:
    config = json.load(fp)


shape_dict = {n : SHAPES[i] for i,n in enumerate(config['all_shape_names'])}
'''
shape_dict = { 
    'Rectangle' : SHAPES[0],
    'Square' : SHAPES[1],
    'Circle' : SHAPES[2],
    'Gelato' : SHAPES[3],
    'Pyramid' : SHAPES[4],
    'Ellipse' : SHAPES[5],
    'Moon' : SHAPES[6]
}
'''

process_params = DottedDict()

# Size of image
process_params['img_sizes'] =  [int(np.log2(i)) for i in config['img_sizes']]
process_params['min_img_size'] = min(process_params.img_sizes)
process_params['max_img_size'] = max(process_params.img_sizes)
process_params['img_size'] =  process_params.img_sizes[0]

process_params['max_colors'] = config['n_colors_max']
process_params['colors'] = min(process_params.max_colors, 10)

# Shapes
process_params['shapes'] = shape_dict.keys()
process_params['selected_shapes'] = shape_dict.keys()

# Fill rate of shapes
process_params['min_fillrate'] = 0.1
process_params['max_fillrate'] = 0.9

# Angles of rotation
process_params['min_rotations'] = config['n_angles_min']
process_params['max_rotations'] = config['n_angles_max']
process_params['nmb_rotations'] = process_params.min_rotations

# Background Mode
process_params['bg_modes'] = config['bg_styles']
process_params['bg_mode'] = process_params.bg_modes[0]

# Number of Backgrounds (0 means random generation)
process_params['nmb_backgrounds'] = 10

# Positions
process_params['min_positions'] = config['n_positions_min']

# Scales
process_params['min_scales'] = config['n_scales_min']

# Percentage of Dataset
process_params['ds_perc'] = 100
process_params['test_perc'] = 0
process_params['max_mem_usage'] = config['mem_usage_gb_max']

# Targets
process_params['target_bbox'] = False
process_params['target_seg'] = False


process_params = recalculate_params(process_params, init=True)
possible_scales = get_scales(process_params)






