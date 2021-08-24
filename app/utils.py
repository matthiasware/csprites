'''
Utility Functions
'''


import numpy as np
import pickle
import os
from dotted_dict import DottedDict
import json
from itertools import count
import time
import timeit
from PIL import Image
from tqdm import tqdm
import shutil

import sys
sys.path.insert(0,'..')
# Local
from csprites.masks import SHAPES
from csprites.configs import be_config
from csprites.masks import get_shapes
from csprites.colors import get_colors, apply_color
from csprites.scales import get_shape_sizes_evenly_scaled
from csprites.angles import get_angles, apply_angle
from csprites.positions import get_max_positions, centered_position_idcs, get_position_idcs_from_center
from csprites.backgrounds import get_bg_func
from csprites.utils import MeanStdTracker, shape_sizes_to_html, masks_to_html_animation


with open('/home/kahlmeyer94/csprites/app/static/base_config.json', 'r') as fp:
    base_config = json.load(fp)
base_config = DottedDict(base_config)
shape_dict = {n : SHAPES[i] for i,n in enumerate(base_config['all_shape_names'])}


class DatasetCreator():
    def __init__(self):
        self.status = 'Finished'

    def create_dataset(self, p, request_id):
        self.status = 'Initialization'

        conf = create_config_dict(p)
        p_dir = f'/home/kahlmeyer94/csprites/app/static/requests/{request_id}'

        if not os.path.exists(p_dir):
            os.makedirs(p_dir)

        # valid parameters? copied as it is from notebook
        self.status = 'Validating Parameters'
        config = conf
        be_config = base_config
        try:
            img_size = config["img_size"]
            assert img_size in be_config.img_sizes

            # shapes
            shape_names = config['shapes']
            for name in shape_names:
                assert name in be_config.all_shape_names
            n_shapes = len(shape_names)

            # colors
            n_colors = config['n_colors']
            assert be_config.n_colors_min <= n_colors
            assert n_colors <= be_config.n_colors_max

            # angles
            n_angles = config['n_angles']
            assert be_config.n_angles_min <= n_angles
            assert n_angles <= be_config.n_angles_max

            # backgrounds
            n_bg = config['n_bg']
            if n_bg != np.inf:
                assert be_config.n_bgs_min <= n_bg
                assert n_bg <= be_config.n_bgs_max

            bg_style = config["bg_style"]
            assert bg_style in be_config.bg_styles

            target_bbox = config["target_bbox"]
            target_segm = config["target_segm"]

            # scales
            n_scales = config["n_scales"]
            assert be_config.n_scales_min <= n_scales
            assert n_scales <= be_config.n_scales_max

            # positions
            n_positions = config["n_positions"]
            assert be_config.n_positions_min <= n_positions
            assert n_positions <= be_config.n_positions_max

            # fill rates
            min_mask_fill_rate = config["min_mask_fill_rate"]
            assert 0 < min_mask_fill_rate < 1

            max_mask_fill_rate = config["max_mask_fill_rate"]
            assert min_mask_fill_rate <= max_mask_fill_rate
            assert max_mask_fill_rate < 1

            # further checks
            shapes = [shape_dict[i] for i in shape_names]
            #
            # shape_sizes
            min_mask_area = img_size** 2 * min_mask_fill_rate
            max_mask_area = img_size** 2 * max_mask_fill_rate
            #
            shape_sizes, max_mask_size = get_shape_sizes_evenly_scaled(shapes, min_mask_area, max_mask_area)

            # n_scales_max
            n_scales_max = len(list(shape_sizes.values())[0])

            # n_positions_max
            n_positions_max = get_max_positions(img_size, max_mask_size)
            #
            assert n_scales <= n_scales_max
            assert n_positions <= n_positions_max

            # subset
            n_train = config["n_train"]
            n_valid = config["n_valid"]
            n_samples = n_train + n_valid

            n_masks = n_shapes * n_colors * n_angles * n_scales
            n_states = n_masks * n_positions**2
            generate_subset = n_samples<n_states

            assert be_config.n_samples_min <= n_samples
            assert n_samples < be_config.n_samples_max
            train_rate = n_train / n_samples

            n_masks = n_shapes * n_colors * n_angles * n_scales
            n_states = n_masks * n_positions**2
            assert n_states >= n_samples

            sampling_rate = n_samples / n_states
            mem_usage_bytes = img_size**2 * 3 * n_samples
            assert mem_usage_bytes * 1e-9 <= be_config.mem_usage_gb_max
        except AssertionError:
            self.status = -1

        colors = get_colors(n_colors, cmap=be_config.cmap_colors)
        angles = get_angles(n_angles)
        positions = centered_position_idcs(n_positions, n_positions_max, max_mask_size)
        #
        bg_shape = (img_size, img_size, 3)
        bg_func = get_bg_func(bg_shape, n_bg, bg_style)
        #
        shape_map = {idx: shape.name for idx, shape in enumerate(shapes)}
        angle_map = {idx: angle for idx, angle in enumerate(angles)}
        color_map = {idx: list(color) for idx, color in enumerate(colors)}
        sizes_map = {key.name : value for key,value in shape_sizes.items()}
        posit_map = {idx: pos for idx,pos in enumerate(positions)}

        self.status = 'Creating Configurations'
        if generate_subset:
            # Generate Params for Subset [good for large state space]
            s_shapes = np.random.choice(a=n_shapes, size=n_samples)
            s_scales = np.random.choice(a=n_scales, size=n_samples)
            s_colors = np.random.choice(a=n_colors, size=n_samples)
            s_angles = np.random.choice(a=n_angles, size=n_samples)
            s_px = np.random.choice(a=n_positions, size=n_samples)
            s_py = np.random.choice(a=n_positions, size=n_samples)
            #
            class_targets = np.stack([s_shapes, s_scales, s_colors, s_angles, s_py, s_px]).T
        else:
            # Generate Params for whole Space [fine for small state space]
            class_targets = []
            for shape_idx in range(n_shapes):
                for scale_idx in range(n_scales):
                    for angle_idx in range(n_angles):
                        for color_idx in range(n_colors):
                            for py_idx in range(n_positions):
                                for px_idx in range(n_positions):
                                    classes = (shape_idx, scale_idx, color_idx, angle_idx, py_idx, px_idx)
                                    class_targets.append(classes)
            class_targets = np.array(class_targets)
        unique_classes = np.unique(class_targets, axis=0).shape[0]

        self.status = 'Creating Paths'
        # Paths
        csprices_type = config['nmb_instances']
        ds_name = be_config.ds_name_tmp.format(
            csprices_type,img_size,img_size,n_shapes, n_colors,
            n_angles, n_positions, n_scales, n_bg, bg_style, n_samples)

        #p_data = be_config.p_base / ds_name
        #p_data.mkdir(exist_ok=True, parents=True)
        p_data = os.path.join(p_dir, ds_name)
        if not os.path.exists(p_data):
            os.mkdir(p_data)

        # save config as json
        json_p = os.path.join(p_data, f'config.json')
        with open(json_p, 'w') as fp:
            json.dump(conf, fp, indent=4)

        #
        p_X_train = os.path.join(p_data, be_config["p_X_train"])
        p_Y_train_clas = os.path.join(p_data, be_config["p_Y_train_clas"])
        p_Y_train_segm = os.path.join(p_data, be_config["p_Y_train_segm"])
        p_Y_train_bbox = os.path.join(p_data, be_config["p_Y_train_bbox"])
        #
        p_X_valid = os.path.join(p_data, be_config["p_X_valid"])
        p_Y_valid_clas = os.path.join(p_data, be_config["p_Y_valid_clas"])
        p_Y_valid_segm = os.path.join(p_data, be_config["p_Y_valid_segm"])
        p_Y_valid_bbox = os.path.join(p_data, be_config["p_Y_valid_bbox"])

        #
        p_config = os.path.join(p_data, be_config["p_config"])
        #
        p_imgs = os.path.join(p_data, be_config["p_imgs"])
        if not os.path.exists(p_imgs):
            os.mkdir(p_imgs)

        if target_segm:
            p_segs = os.path.join(p_data, be_config["p_segs"])
            if not os.path.exists(p_segs):
                os.mkdir(p_segs)

        # Create Dataset
        self.status = 'Creating Dataset : 0.00%'

        debug = False
        n_debug = 10
        imgs = []
        #
        targets_bbox = []
        targets_segm = []
        #
        tracker = MeanStdTracker()
        #
        start = timeit.default_timer()
        for sample_idx, (shape_idx, scale_idx, color_idx, angle_idx, py_idx, px_idx) in enumerate(class_targets):
            self.status = f'Creating Dataset : {round(sample_idx/len(class_targets),2)}%'
            shape = shapes[shape_idx]
            size = shape_sizes[shape][scale_idx]
            angle = angles[angle_idx]
            color = colors[color_idx]
            px = positions[px_idx]  # center position width
            py = positions[py_idx]  # center position height
            #
            mask = shape.create(size)
            mask = apply_angle(mask, angle)
            #
            h_mask, w_mask = mask.shape
            #
            assert h_mask % 2 == 1
            assert w_mask % 2 == 1

            #mask = pad_mask(mask, max_mask_size)
            if target_bbox:
                # corners: (upper left, lower right)
                w_shape = (mask.sum(axis=0) > 0).sum()
                h_shape = (mask.sum(axis=1) > 0).sum()
                #
                if w_shape % 2 == 0:
                    w_shape+=1
                if h_shape % 2 == 0:
                    h_shape+=1
                #
                assert w_shape % 2 == 1
                assert h_shape % 2 == 1

                y0 = max(0, py - h_shape // 2 - 1)
                x0 = max(0, px - w_shape // 2 - 1)
                y1 = min(py + h_shape // 2 + 1, img_size)
                x1 = min(px + w_shape // 2 + 1, img_size)
                targets_bbox.append((y0, x0 , y1, x1))

            x0, y0, x1, y1 = get_position_idcs_from_center(h_mask, w_mask, px, py)
            #
            if target_segm:
                seg_map = np.zeros((img_size, img_size)).astype(np.uint8)
                seg_map[y0: y1, x0: x1] = mask
                seg_name = be_config["seg_name"].format(sample_idx)
                p_seg = os.path.join(p_segs, seg_name)
                #
                Image.fromarray(seg_map*255).save(p_seg)
                targets_segm.append(seg_name)
            #
            mask_wo_color = np.copy(mask)
            mask = apply_color(mask, color)
            #
            img = bg_func()
            img[y0: y1, x0: x1,:][mask_wo_color > 0] = mask[mask_wo_color > 0]
            #
            img_name = be_config["img_name"].format(sample_idx)
            p_img = os.path.join(p_imgs, img_name)
            Image.fromarray(img).save(p_img)
            tracker.add(img/255)
            imgs.append(img_name)

            if debug and sample_idx > n_debug:
                break

        elapsed  = timeit.default_timer() - start
        print("{:.3f}".format(elapsed))

        # Train/Testsplit
        self.status = 'Creating Train- and Testsplit'
        X = np.array(imgs)
        assert n_samples == X.shape[0]

        Y_clas = class_targets
        if targets_bbox:
            Y_bbox = np.array(targets_bbox)
        if targets_segm:
            Y_segm = np.array(targets_segm)
        idcs = np.arange(n_samples)
        np.random.shuffle(idcs)
        #
        train_idcs = idcs[:n_train]
        valid_idcs = idcs[n_train:]
        #
        X_train = X[train_idcs]
        X_valid = X[valid_idcs]
        #
        Y_train_clas = Y_clas[train_idcs]
        Y_valid_clas = Y_clas[valid_idcs]

        if target_bbox:
            Y_train_bbox = Y_bbox[train_idcs]
            Y_valid_bbox = Y_bbox[valid_idcs]
            #
            np.save(p_Y_train_bbox, Y_train_bbox)
            np.save(p_Y_valid_bbox, Y_valid_bbox)

        if target_segm:
            Y_train_segm = Y_segm[train_idcs]
            Y_valid_segm = Y_segm[valid_idcs]
            #
            np.save(p_Y_train_segm, Y_train_segm)
            np.save(p_Y_valid_segm, Y_valid_segm)
        #
        np.save(p_Y_train_clas, Y_train_clas)
        np.save(p_Y_valid_clas, Y_valid_clas)
        np.save(p_X_train, X_train)
        np.save(p_X_valid, X_valid)

        # zip the dataset
        self.status = 'Creating zip archive'
        shutil.make_archive(os.path.join(p_dir, ds_name), 'zip', p_data)

        # keep only zip
        self.status = 'Cleaning up'
        shutil.rmtree(p_data)

        self.status = 'Finished'


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

    # TODO: Single/Multiple instance
    ret['nmb_instances'] = 'single'
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

def get_initial_params():
    process_params = DottedDict()

    # TODO: Single/Multiple instance

    # Size of image
    process_params['img_sizes'] =  [int(np.log2(i)) for i in base_config['img_sizes']]
    process_params['min_img_size'] = min(process_params.img_sizes)
    process_params['max_img_size'] = max(process_params.img_sizes)
    process_params['img_size'] =  process_params.img_sizes[0]

    process_params['max_colors'] = base_config['n_colors_max']
    process_params['colors'] = min(process_params.max_colors, 10)

    # Shapes
    process_params['shapes'] = list(shape_dict.keys())
    process_params['selected_shapes'] = list(shape_dict.keys())

    # Fill rate of shapes
    process_params['min_fillrate'] = 0.1
    process_params['max_fillrate'] = 0.9

    # Angles of rotation
    process_params['min_rotations'] = base_config['n_angles_min']
    process_params['max_rotations'] = base_config['n_angles_max']
    process_params['nmb_rotations'] = process_params.min_rotations

    # Background Mode
    process_params['bg_modes'] = base_config['bg_styles']
    process_params['bg_mode'] = process_params.bg_modes[0]

    # Number of Backgrounds (0 means random generation)
    process_params['nmb_backgrounds'] = 10

    # Positions
    process_params['min_positions'] = base_config['n_positions_min']

    # Scales
    process_params['min_scales'] = base_config['n_scales_min']

    # Percentage of Dataset
    process_params['ds_perc'] = 100
    process_params['test_perc'] = 0
    process_params['max_mem_usage'] = base_config['mem_usage_gb_max']

    # Targets
    process_params['target_bbox'] = False
    process_params['target_seg'] = False

    process_params = recalculate_params(process_params, init=True)
    possible_scales = get_scales(process_params)


    return process_params, possible_scales






