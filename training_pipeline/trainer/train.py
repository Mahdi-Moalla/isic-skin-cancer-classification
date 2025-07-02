import os
import os.path as osp
import shutil
import warnings

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
warnings.filterwarnings("ignore")

from addict import Dict

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from torchvision.transforms import v2 as torchvision_v2


#from torcheval.metrics.functional import binary_auroc
import pytorch_lightning as pl   ########
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


from sklearn.model_selection import StratifiedKFold

from isic_datset import isic_skin_cancer_datset,  isic_train_sampler

from isic_model import isic_classifier

from scorer import binary_auroc_scorer


def prepare_data(config, 
                 X_train, 
                 X_val,
                 train_transforms,
                 val_transforms):
    


    train_dataset=isic_skin_cancer_datset(osp.join(config.data_dir,'train-metadata.csv'),
                                          osp.join(config.data_dir,'train-image.hdf5'),
                                          isic_ids=X_train,
                                          transform=train_transforms,
                                          tab_features=config.tab_features)
    
    
    val_dataset=isic_skin_cancer_datset(osp.join(config.data_dir,'train-metadata.csv'),
                                        osp.join(config.data_dir,'train-image.hdf5'),
                                        isic_ids=X_val,
                                        transform=val_transforms,
                                          tab_features=config.tab_features)
    
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=config.train_batch_size, 
                                  #shuffle=True,
                                  sampler=isic_train_sampler( train_dataset, 
                                                             total_size=config.train_sample_count,
                                                             pos_ratio=config.train_pos_ratio),
                                  num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config.val_batch_size, 
                                shuffle=False, 
                                num_workers=config.num_workers)

    return train_dataloader, val_dataloader


def fold_train(config,
               fold_i,
               train_dataloader, 
               val_dataloader,
               logger):
    
    model = isic_classifier(config)

    checkpoint_callback = ModelCheckpoint(dirpath=config.checkpoints_dir,
                                      filename=f'best_{fold_i}',
                                      save_top_k=1,
                                      monitor="val_binary_auc")

    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        deterministic=True,
        callbacks=[LearningRateMonitor(logging_interval='step'),
                  binary_auroc_scorer(config),
                  checkpoint_callback],
    )

    trainer.fold_i=fold_i
    
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    best_model_path=osp.join(config.checkpoints_dir,f'best_{fold_i}.ckpt')
    
    print('%%%%%%%%%%% finetuning.....')
    
    model = isic_classifier.load_from_checkpoint(best_model_path,
                                                 config=config, 
                                                 train_mode='finetuning')
    
    
    checkpoint_callback = ModelCheckpoint(dirpath=config.checkpoints_dir,
                                          filename=f'best_finetune_{fold_i}',
                                          save_top_k=1,
                                          monitor="val_binary_auc")
    
    
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        deterministic=True,
        callbacks=[LearningRateMonitor(logging_interval='step'),
                  binary_auroc_scorer(config),
                  checkpoint_callback],
    )

    trainer.fold_i=fold_i
    
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
    

def infer_test_data(config, val_transforms):

    test_dataset=isic_skin_cancer_datset(osp.join(config.data_dir,'test-metadata.csv'),
                                        osp.join(config.data_dir,'test-image.hdf5'),
                                        transform=val_transforms,
                                        tab_features=config.tab_features)

    test_dataloader = DataLoader(test_dataset,
                                batch_size=config.val_batch_size, 
                                shuffle=False, 
                                num_workers=config.num_workers)
    
    test_preds=[]

    for fold_i in range(config.n_fold):

        best_model_path=osp.join(config.checkpoints_dir,
                                 f'best_finetune_{fold_i}.ckpt')
        model = isic_classifier.load_from_checkpoint(best_model_path,
                                                    config=config, 
                                                    train_mode='finetuning')
        
        trainer=pl.Trainer(
            accelerator="auto",
            devices=1,
            deterministic=True,
        )

        preds=trainer.predict(model, test_dataloader, return_predictions=True)
        
        test_preds.append(np.concatenate([t.cpu().numpy() for t in preds]))

    test_metadata=pd.read_csv(osp.join(config.data_dir,'test-metadata.csv'))
    test_metadata['target']=np.mean(test_preds,axis=0)
    test_metadata[ ['isic_id','target'] ].to_csv('submission.csv', index=False)



def main(config,
         train_transforms,
         val_transforms):
    
    pl.seed_everything(config.seed)

    
    #test_metadata=pd.read_csv(osp.join(config.data_dir,'test-metadata.csv'))

    shutil.rmtree(config.checkpoints_dir, ignore_errors=True)
    shutil.rmtree(config.log_dir, ignore_errors=True)

    logger = CSVLogger(save_dir=config.log_dir)
    

    train_metadata=pd.read_csv(osp.join(config.data_dir,'train-metadata.csv'))

    X=train_metadata['isic_id']
    y=train_metadata['target']
    
    print('########## StratifiedKFold ##########')
    skf = StratifiedKFold(n_splits=config.n_fold)
    for i, (train_index, val_index) in enumerate(skf.split(X,y)):
        print('#########################################################################')
        print('#########################################################################')
        print(f'fold {i}: {len(val_index)/len(y)}  {sum(y[val_index])/sum(y)}')
        X_train=train_metadata.loc[train_index, 'isic_id']
        X_val=train_metadata.loc[val_index, 'isic_id']
    
        # reduce val dataset
        val_pos=train_metadata.loc[ (train_metadata['isic_id'].isin(X_val)) & \
                                (train_metadata['target']==1) ]
        val_neg=train_metadata.loc[ (train_metadata['isic_id'].isin(X_val)) & \
                                    (train_metadata['target']==0) ]
        
        
        reduced_X_val=pd.concat([val_pos['isic_id'],val_neg.loc[::50,'isic_id']])
        #reduced_X_val=pd.concat([val_pos['isic_id'],
        #                         val_neg.sample(frac=0.2)['isic_id']])
        #reduced_X_val.shape
    
        train_dataloader, val_dataloader = prepare_data(config, 
                                                        X_train, 
                                                        reduced_X_val,
                                                        train_transforms,
                                                        val_transforms)
    
        fold_train(config, i, train_dataloader, val_dataloader, logger )


    
    logs=pd.read_csv(osp.join(config.log_dir,'lightning_logs/version_0/metrics.csv'))
    
    
    logs=logs.loc[logs['log_val']==1,['epoch','fold_i','train_mode','val_binary_auc','val_binary_auc_tpr.8']]
    
    logs.epoch=logs.epoch.astype('int')
    logs.fold_i=logs.fold_i.astype('int')
    logs.train_mode=logs.train_mode.astype('int')
    
    logs.train_mode=logs.train_mode.map(lambda x:'finetune' if x==1 else 'transfer-learning')
    
    print('####################################')
    print(logs.loc[logs['train_mode']=='finetune'])
    print('####################################')
    print(logs.loc[logs['train_mode']=='finetune','val_binary_auc_tpr.8'].mean())

    infer_test_data(config, val_transforms)


#if __name__=='__main__':
if True:
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


    main(config,
         train_transforms,
         val_transforms)

    #infer_test_data(config, val_transforms)
