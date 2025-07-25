"""
python config file
"""

import os

from addict import Dict

tab_features = [
    'clin_size_long_diam_mm',
    'tbp_lv_A',
    'tbp_lv_Aext',
    'tbp_lv_B',
    'tbp_lv_Bext',
    'tbp_lv_C',
    'tbp_lv_Cext',
    'tbp_lv_H',
    'tbp_lv_Hext',
    'tbp_lv_L',
    'tbp_lv_Lext',
    'tbp_lv_areaMM2',
    'tbp_lv_area_perim_ratio',
    'tbp_lv_color_std_mean',
    'tbp_lv_deltaA',
    'tbp_lv_deltaB',
    'tbp_lv_deltaL',
    'tbp_lv_deltaLBnorm',
    'tbp_lv_eccentricity',
    'tbp_lv_minorAxisMM',
    'tbp_lv_nevi_confidence',
    'tbp_lv_norm_border',
    'tbp_lv_norm_color',
    'tbp_lv_perimeterMM',
    'tbp_lv_radial_color_std_max',
    'tbp_lv_stdL',
    'tbp_lv_stdLExt',
    'tbp_lv_symm_2axis',
    'tbp_lv_symm_2axis_angle',
    'tbp_lv_x',
    'tbp_lv_y',
    'tbp_lv_z',
]
config = Dict(
    {
        "seed": 1,
        "epochs": 1,
        # "img_size": 384,
        "tab_features": tab_features,
        "model": "efficientnet_b0",
        "pretrained_weights_type": "torchvision_id",
        "pretrained_weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        # "pretrained_weights" : None,
        # "/kaggle/input/efficientnet_b0/pytorch/default/1/efficientnet_b0_rwightman-7f5810bc.pth",
        "transfer_learning_layer": ['classifier', 1],
        "transfer_learning_layer_spec": [64, 2],
        "train_batch_size": 32,
        "val_batch_size": 64,
        "num_workers": int(os.cpu_count() - 1),
        "train_sample_count": 2**14,
        "train_pos_ratio": 0.5,
        "learning_rate": 1e-3,
        "finetune_learning_rate": 1e-4,
        "optimizer": "torch.optim.Adam",
        "optimizer_params": {},
        "lr_sceduler": "torch.optim.lr_scheduler.ReduceLROnPlateau",
        "lr_sceduler_params": {"mode": 'min', "factor": 0.1, "patience": 5},
        "n_fold": 2,
        "data_dir": "/workspace/data/",
        "checkpoints_dir": "/workspace/working/checkpoints/",
        "log_dir": "/workspace/working/logs/",
        "pos_class_weight": 2.0,
        "roc_pos_class_weight": 1000,
    }
)
