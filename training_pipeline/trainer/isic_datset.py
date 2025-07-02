import os
import os.path as osp

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

import numpy as np

import  pandas  as pd

import h5py

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
