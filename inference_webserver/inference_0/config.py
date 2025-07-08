import os
from addict import Dict
from torchvision.transforms import v2 as torchvision_v2
import torch

tab_features=['clin_size_long_diam_mm',
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
                     'tbp_lv_z']
config = Dict({
        "seed": 1,
        "epochs": 1,
        #"img_size": 384,

        "tab_features":tab_features,
        
        "model": "efficientnet_b0",
        "pretrained_weights_type": "torchvision_id",
        "pretrained_weights" : "EfficientNet_B0_Weights.IMAGENET1K_V1",
        #"pretrained_weights" : None,#"/kaggle/input/efficientnet_b0/pytorch/default/1/efficientnet_b0_rwightman-7f5810bc.pth",
        
        "transfer_learning_layer":['classifier',1],
        "transfer_learning_layer_spec":[64,2],
        
        "train_batch_size": 32,
        "val_batch_size": 64,
        "num_workers":int(os.cpu_count() - 1),

        "train_sample_count":2**14,
        "train_pos_ratio":0.5,
        
        "learning_rate": 1e-3,
        "finetune_learning_rate":1e-4,
        "optimizer":"torch.optim.Adam",
        "optimizer_params":{},
        "lr_sceduler":"torch.optim.lr_scheduler.ReduceLROnPlateau",
        "lr_sceduler_params":{"mode":'min',
                              "factor":0.1,
                              "patience":5},
        "n_fold": 2,
        "data_dir":"/workspace/data/",
        "checkpoints_dir":"/workspace/working/checkpoints/",
        "log_dir":"/workspace/working/logs/",
        "pos_class_weight":2.0,
        
        "roc_pos_class_weight":1000
    })

train_transforms = torchvision_v2.Compose([
            #torchvision_v2.PILToTensor(),
            #torchvision_v2.Resize(size=(256,256),interpolation = torchvision_v2.InterpolationMode.BICUBIC ),
            #torchvision_v2.CenterCrop(size=(224, 224)),
            torchvision_v2.RandomHorizontalFlip(p=0.5),
            torchvision_v2.RandomVerticalFlip(p=0.5),
            #torchvision_v2.RandomAffine(degrees=45, translate=(0.1,0.3), scale=(0.75,1.0)),
            #torchvision_v2.ColorJitter(brightness=.5, hue=.3),
            #torchvision_v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
            #torchvision_v2.RandomInvert(),
            #torchvision_v2.RandomPosterize(bits=2),
            #torchvision_v2.RandomAdjustSharpness(sharpness_factor=2),
            #torchvision_v2.RandAugment(),
            torchvision_v2.AutoAugment(torchvision_v2.AutoAugmentPolicy.IMAGENET),
            #torchvision_v2.RandomEqualize(),
            torchvision_v2.AugMix(),
            torchvision_v2.ToDtype(torch.float32, scale=True),
            torchvision_v2.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
    ])
    
val_transforms = torchvision_v2.Compose([
            #torchvision_v2.PILToTensor(),
            #torchvision_v2.Resize(size=(256,256),interpolation = torchvision_v2.InterpolationMode.BICUBIC ),
            #torchvision_v2.CenterCrop(size=(224, 224)),
            torchvision_v2.ToDtype(torch.float32, scale=True),
            torchvision_v2.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]),
    ])