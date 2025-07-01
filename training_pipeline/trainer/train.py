import os
import os.path as osp
import shutil
import warnings

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
warnings.filterwarnings("ignore")

from addict import Dict   ########

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, Sampler
#from torch.cuda import amp

import torchvision
from torchvision.models import get_model
from torchvision.transforms import v2 as torchvision_v2


#from torcheval.metrics.functional import binary_auroc
import pytorch_lightning as pl   ########
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor

#import albumentations as A

import h5py   ########

from PIL import Image
import io
import matplotlib.pyplot as plt

from tqdm.auto import tqdm



from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, auc


class  isic_skin_cancer_datset(Dataset):
    def __init__(self, 
                 metadata_file_path,
                 h5file_path,
                 isic_ids=None,
                 transform=None,
                 tab_features=[]):
        super().__init__()
        
        assert osp.isfile(metadata_file_path)
        assert osp.isfile(h5file_path)
        
        self.metadata_file_path=metadata_file_path
        self.h5file_path=h5file_path

        self.metadata=pd.read_csv(self.metadata_file_path)
        self.h5data=h5py.File(self.h5file_path, 'r')

        if isic_ids is not None:
            assert isic_ids.isin(self.metadata['isic_id']).sum()==len(isic_ids)
            self.isic_ids=isic_ids.tolist()
        else:
            self.isic_ids = self.metadata['isic_id'].tolist()

        self.isic_id_lut={ idx:k  for k,idx in enumerate(self.metadata['isic_id'])  }

        self.transform=transform

        self.tab_features=tab_features

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self,idx):
        isic_id=self.isic_ids[idx]
        #pil_image=Image.open( io.BytesIO( self.h5data[isic_id][()] ) )
        image=torch.tensor(self.h5data[isic_id][...])#permute(2,0,1)
        #print(image.shape)
        label=self.metadata.loc[self.isic_id_lut[isic_id],'target']\
                if 'target' in self.metadata.columns else -1
        return self.transform(image),\
            torch.tensor(self.metadata.loc[self.isic_id_lut[isic_id],self.tab_features],  dtype=torch.float),\
            label
                

class isic_train_sampler(Sampler):

    def __init__(self, 
                 isic_dataset,
                 pos_ratio=0.5,
                 total_size=2**16):
        self.pos_idxs=[]
        self.neg_idxs=[]
        
        for i,isic_id in enumerate(isic_dataset.isic_ids):
            pd_idx=isic_dataset.isic_id_lut[isic_id]
            if isic_dataset.metadata.loc[pd_idx,'target']==1:
                self.pos_idxs.append(i)
            else:
                self.neg_idxs.append(i)

        self.pos_i=0
        self.neg_i=0

        self.pos_idxs=np.array(self.pos_idxs)
        self.neg_idxs=np.array(self.neg_idxs)

        

        self.pos_ratio=pos_ratio
        self.total_size=total_size


    def __len__(self):
        return self.total_size

    def __iter__(self):
        np.random.shuffle(self.pos_idxs)
        np.random.shuffle(self.neg_idxs)
        
        new_idxs=[]
        for _ in range(self.total_size):
            if np.random.uniform()<=self.pos_ratio:
                idx=self.pos_idxs[ self.pos_i ]
                self.pos_i = (self.pos_i + 1) % len(self.pos_idxs)
            else:
                idx=self.neg_idxs[ self.neg_i ]
                self.neg_i = (self.neg_i + 1) % len(self.neg_idxs)
    
            new_idxs.append(idx)
        yield from new_idxs

def dynamic_import(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

class isic_classifier(pl.LightningModule):

    output_features=2

    def __init__(self,
                 config,
                 train_mode='transfer-learning'):

        super().__init__()
        
        self.config=config
        if self.config.pretrained_weights_type=="torchvision_id":
            self.model=torchvision.models.get_model(self.config.model,
                                                    weights=self.config.pretrained_weights)
        elif self.config.pretrained_weights_type=="path":
            self.model=torchvision.models.get_model(self.config.model, weights=None)
            if  self.config.pretrained_weights is not  None:
                self.model.load_state_dict(torch.load(self.config.pretrained_weights))
        else:
            raise Exception(f'unknown pretrained_weights_type {self.config.pretrained_weights_type}')

        self.train_mode=train_mode
        
        #######################################################
        # generic update of classification layer
        
        transfer_learning_layer=self.config.transfer_learning_layer

        ## finding last-1 layer  
        layer=self.model
        
        for layer_id in transfer_learning_layer[:-1]:
            if  type(layer_id)  is str:
                layer=getattr(layer,layer_id)
            elif type(layer_id)  is int:
                layer=layer[layer_id]
            else:
                raise Exception(f"unknown layer_id type: {type(layer_id)} of  {layer_id}")


        ## updating last layer

        #self.model.new_fc_layer=nn.Sequential(*[nn.LazyLinear(out_feats) \
        #                    for  out_feats in self.config.transfer_learning_layer_spec])

        self.model.new_fc_layer=nn.ModuleDict({'transfer_learning':
                nn.Sequential(*[nn.LazyLinear(out_feats) \
                            for  out_feats in self.config.transfer_learning_layer_spec[:-1]]),
                                              'merger':nn.LazyLinear(
                                                self.config.transfer_learning_layer_spec[-1])})

        layer_id=transfer_learning_layer[-1]
        if  type(layer_id)  is str:
            
            curr_last_layer  =  getattr(layer,layer_id) 
            assert type( curr_last_layer ) is nn.Linear
            in_features = curr_last_layer.in_features
            #self.new_fc_layer=nn.Linear(in_features , self.output_features)
            
            #layers_specs=[in_features]+self.config.transfer_learning_layer_spec
            #self.model.new_fc_layer=nn.Sequential(*[nn.Linear(in_feats, out_feats) \
            #        for  in_feats, out_feats in zip(layers_specs[:-1],layers_specs[1:])])
            
            setattr(layer,layer_id, self.model.new_fc_layer['transfer_learning'] )
            
        elif type(layer_id)  is int:
            curr_last_layer  =  layer[layer_id] 
            assert type( curr_last_layer ) is nn.Linear
            in_features = curr_last_layer.in_features
            #self.new_fc_layer=nn.Linear(in_features , self.output_features )
            
            #layers_specs=[in_features]+self.config.transfer_learning_layer_spec
            #self.model.new_fc_layer=nn.Sequential(*[nn.Linear(in_feats, out_feats) \
            #        for  in_feats, out_feats in zip(layers_specs[:-1],layers_specs[1:])])
            
            layer[layer_id] =  self.model.new_fc_layer['transfer_learning']
        else:
            raise Exception(f"unknown layer_id type: {type(layer_id)} of  {layer_id}")

        

    def forward(self, x, tab_feats):
        x= self.model(x)
        #import  pdb; pdb.set_trace()
        x=self.model.new_fc_layer['merger'](torch.cat([x,tab_feats],dim=1))
        #import  pdb; pdb.set_trace()
        return x

    def training_step(self, batch, batch_idx):
        #import  pdb; pdb.set_trace()
        x, tab_feats, y = batch
        #import  pdb; pdb.set_trace()
        logits = F.log_softmax(self(x,tab_feats ), dim=1)
        
        loss = F.nll_loss(logits, y,
                          weight = torch.tensor([1.0,self.config.pos_class_weight], 
                                                device=self.device))
        self.log("train_loss", loss,  logger=None)
        return loss

    def evaluate(self, batch, stage=None):
        #import  pdb; pdb.set_trace()
        x, tab_feats, y = batch
        #import  pdb; pdb.set_trace()
        logits = F.log_softmax(self(x,tab_feats ), dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.validation_targets.extend(y.tolist())
        self.validation_scores.extend(F.softmax(logits)[:,1].tolist())
        
        if stage:
            self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):

        x, tab_feats,_=batch
        return F.softmax(self(x, tab_feats), dim=1)[...,1]

    def configure_optimizers(self):

        try:
            optimizer_cls=dynamic_import(self.config.optimizer)
        except:
            print('failed to import optimizer')
            exit(1)

        try:
            lr_scheduler_cls=dynamic_import(self.config.lr_sceduler)
        except:
            print('failed to import lr scheduler')
            exit(1)
        
        
        if self.train_mode=='transfer-learning':
            optimizer = optimizer_cls(self.model.new_fc_layer.parameters(),
                                      lr=self.config.learning_rate,
                                  **self.config.optimizer_params)
        elif self.train_mode=='finetuning':
            optimizer = optimizer_cls(self.parameters(),
                                      lr=self.config.finetune_learning_rate,
                                      **self.config.optimizer_params)
        else:
            raise Exception('unknown train mode')
        
        return {
                "optimizer":optimizer,
                "lr_scheduler": {"scheduler":lr_scheduler_cls(optimizer,
                                                        **self.config.lr_sceduler_params),
                                 "monitor": "train_loss"}
        }



class binary_auroc_validator(Callback):
    def __init__(self, config):
        self.config=config
        self.sanity_check=False
        os.makedirs(osp.join(config.log_dir,'aucs'), exist_ok=True)
        self.train_mode_lut={'transfer-learning':0,
                             'finetuning':1}

    def on_fit_start(self, trainer, pl_module):
        self.val_binary_auc=[]
        self.val_binary_auc_tpr08=[]
        

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.sanity_check:
            self.sanity_check=True
            return
        
        val_scores_np=np.array(pl_module.validation_scores)
        val_targets_np=np.array(pl_module.validation_targets)
        val_weights_np=np.array([ self.config.roc_pos_class_weight if x==1  else 1.0\
                                    for x in pl_module.validation_targets ])

        fpr, tpr, thresholds = roc_curve(val_targets_np, val_scores_np)
        fpr_1=fpr[tpr>=0.8]
        tpr_1=tpr[tpr>=0.8]
        
        if len(fpr_1)==0 or len(tpr_1)==0:
            return
            
        #print(f'tpr_1[0]={tpr_1[0]}')
        fpr_1-=fpr_1[0]
        tpr_1-=tpr_1[0]
        
        
        
        score_08=auc(fpr_1,tpr_1)
        #import pdb; pdb.set_trace()
        score=roc_auc_score(val_targets_np,
                            val_scores_np,
                            sample_weight=val_weights_np)

        self.val_binary_auc.append(score)
        self.val_binary_auc_tpr08.append(score_08)
                           
        print(f'@@@@@@@ binary au_roc_tpr.8 = {score_08} @@@@@@@')
        print(f'@@@@@@@ binary au_roc       = {score} @@@@@@@')
        
        
        pl_module.log("val_binary_auc_tpr.8", 
                      score_08, 
                      on_epoch =True, 
                      prog_bar =True,
                      logger=True)
        pl_module.log("val_binary_auc", 
                      score, 
                      on_epoch =True, 
                      prog_bar =True,
                      logger=True)
        pl_module.log("fold_i", 
                      trainer.fold_i, 
                      on_epoch =True, 
                      #prog_bar =True,
                      logger=True)
        pl_module.log("train_mode", 
                      self.train_mode_lut[pl_module.train_mode], 
                      on_epoch =True, 
                      #prog_bar =True,
                      logger=True)
        pl_module.log("log_val", 
                      1, 
                      on_epoch =True, 
                      #prog_bar =True,
                      logger=True)

        #fpr, tpr, _ = roc_curve(val_targets_np, val_scores_np)

        #plt.figure()
        #plt.plot(fpr, tpr,color='b')
        #plt.plot([0,1], [.8, .8],color='r')
        #plt.title(f'roc at epoch {trainer.current_epoch}')
        #plt.savefig(osp.join(config.log_dir,'rocs',
        #        f'fold_{trainer.fold_i}_mode_{pl_module.train_mode}_epoch_{trainer.current_epoch}.png'))
        #plt.show()

        plt.figure()

        plt.plot(self.val_binary_auc, color='b', marker="*")
        plt.title(f'aucs up to {trainer.current_epoch}')
        plt.show()

    def on_fit_end(self, trainer, pl_module):
        
        plt.figure()
        plt.plot(self.val_binary_auc, color='b', marker="*")
        plt.title(f'aucs')
        plt.savefig(osp.join(config.log_dir,'aucs',
                f'auc_fold_{trainer.fold_i}_mode_{self.train_mode_lut[pl_module.train_mode]}.png'))
        plt.show()

        plt.figure()
        plt.plot(self.val_binary_auc_tpr08, color='r', marker="*")
        plt.title(f'aucs tpr=.0')
        plt.savefig(osp.join(config.log_dir,'aucs',
                f'aucs08_fold_{trainer.fold_i}_mode_{self.train_mode_lut[pl_module.train_mode]}.png'))
        plt.show()
        
                
    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.validation_targets=[]
        pl_module.validation_scores=[]




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
                  binary_auroc_validator(config),
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
                  binary_auroc_validator(config),
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
        "epochs": 10,
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
        "train_pos_ratio":0.75,
        
        "learning_rate": 1e-3,
        "finetune_learning_rate":1e-4,
        "optimizer":"torch.optim.Adam",
        "optimizer_params":{},
        "lr_sceduler":"torch.optim.lr_scheduler.ReduceLROnPlateau",
        "lr_sceduler_params":{"mode":'min',
                              "factor":0.1,
                              "patience":5},
        "n_fold": 5,
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
