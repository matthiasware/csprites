from pathlib import Path
#
from dotted_dict import DottedDict
import numpy as np

be_config = {
    'p_base': Path("/mnt/data/csprites"),
    'img_sizes': [32, 64, 128, 256, 512],
    'all_shape_names': ['rectangle', 'square', 'circle',
                        'gelato', 'pyramid', 'ellipse', 'moon'],
    'n_colors_min': 1,
    'n_colors_max': 8192,
    'n_angles_max': 360,
    'n_angles_min': 1,
    'n_scales_min': 1,
    'n_scales_max': 512,
    'n_positions_min': 1,
    'n_positions_max': 512,
    'n_bgs_min': 1,
    'n_bgs_max': 8192,
    'n_bgs_lim': np.inf,
    'bg_styles': ['constant_color', 'random_pixel', 'random_function'],
    'mem_usage_gb_max': 10,
    'csprices_types': ["single", "multi"],
    'ds_name_tmp': "{}_csprites_{}x{}_n{}_c{}_a{}_p{}_s{}_bg_{}_{}_{}",
    'img_name': 'csprite{}.png',
    'seg_name': 'csprite{}_seg.png',
    'p_X_train': "X_train.npy",
    'p_Y_train_clas': "Y_train_clas.npy",
    'p_Y_train_segm': "Y_train_segm.npy",
    'p_Y_train_bbox': "Y_train_bbox.npy",
    'p_X_valid': "X_valid.npy",
    'p_Y_valid_clas': "Y_valid_clas.npy",
    'p_Y_valid_segm': "Y_valid_segm.npy",
    'p_Y_valid_bbox': "Y_valid_bbox.npy",
    'p_gifs': "gifs",
    'p_imgs': "imgs",
    'p_segs': "segs",
    'p_config': 'config.pkl',
    'n_samples_min': 1,
    'n_samples_max': 1e7,
    'cmap_colors': 'nipy_spectral',
}
be_config = DottedDict(be_config)
