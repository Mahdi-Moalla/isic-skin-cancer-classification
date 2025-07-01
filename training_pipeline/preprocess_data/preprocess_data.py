import os
import os.path as osp
import shutil

import h5py

import fire

import numpy as np

from PIL import Image
import io
import albumentations as A
import cv2

from tqdm import tqdm

preprocess_transform=A.Compose([
    A.Resize(
            height=256,
            width=256,
            interpolation=cv2.INTER_CUBIC,
            p=1.0),
    A.CenterCrop(height=224, width=224)
])


def preprocess_data(input_data_path='../../project_data_prepare/split_dataset/',
                    preprocessed_data_path='./preprocessed_data/'):

    os.makedirs(preprocessed_data_path, exist_ok=True)

    shutil.copy(osp.join(input_data_path,'train_metadata.csv'),
                preprocessed_data_path)
    shutil.copy(osp.join(input_data_path,'test_metadata.csv'),
                preprocessed_data_path)
    

    for split  in ['train','test']:
        with h5py.File(osp.join(input_data_path,f'{split}_image.hdf5'),'r') as f_in:
            with h5py.File(osp.join(preprocessed_data_path,f'{split}_image.hdf5'),'w') as f_out:
                for isic_id in tqdm(f_in.keys()):
                    pil_image=Image.open( io.BytesIO( f_in[isic_id][()]))
                    image_np=np.array(pil_image)
                    preprocessed_image_np=preprocess_transform(image=image_np)['image']
                    f_out[isic_id]=preprocessed_image_np[...]

    #from IPython import embed; embed(colors='Linux')

if __name__=='__main__':
    
    fire.Fire(preprocess_data)
