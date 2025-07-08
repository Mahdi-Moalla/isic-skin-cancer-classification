import  fire

import pandas as pd

import numpy  as np

import h5py

from PIL import Image
import io

import torch
import torch.nn.functional as F

import common

from preprocess_data import preprocess_transform
from config import (config,
                    tab_features,
                    val_transforms)

from onnx_trt import build_engine_onnx

from  isic_model import isic_classifier

def trt_inference(onnx_model_path,
                  tab_feats_torch,
                  image_torch):

    print('################## trt ####################')
    

    engine = build_engine_onnx(onnx_model_path)

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
    context = engine.create_execution_context()

    tab_feats_np=tab_feats_torch.numpy()
    image_np = image_torch.numpy()
    np.copyto(inputs[0].host, image_np.ravel())
    np.copyto(inputs[1].host, tab_feats_np.ravel())

    trt_outputs = common.do_inference(
            context,
            engine=engine,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )
        # We use the highest probability as our prediction. Its index corresponds to the predicted label.
    print(trt_outputs[0])
    
    common.free_buffers(inputs, outputs, stream)
    


def pytorch_inference(pytorch_model_path,
                      tab_feats,
                      image_torch):

    print('################## pytorch ####################')
    

    with torch.no_grad():
        logits=model(image_torch[None,...].cuda(),
                    tab_feats[None,...].cuda())
    
    print(logits)

    #scores = F.softmax(logits, dim=1)

    #print(scores)

    #from IPython import embed; embed(colors='Linux')

def main(pytorch_model_path='./best_finetune_0.ckpt',
         onnx_model_path='./model.onnx',
         metadata='/dataset/train-metadata.csv',
         images_hdf5='/dataset/train-image.hdf5'):
    
    
    metadata_df = pd.read_csv(metadata)

    idx=np.random.randint(len(metadata_df))

    isic_id=metadata_df.loc[idx, 'isic_id']

    print(f'isic_id: {isic_id}')

    tab_feats=metadata_df.loc[idx,tab_features]

    tab_feats=tab_feats.to_numpy().astype(np.float32)


    with h5py.File(images_hdf5,'r') as f:
        image_bytes=f[isic_id][()]
    
    pil_image=Image.open( io.BytesIO( image_bytes ))

    image_np=np.array(pil_image)
    
    preprocessed_image_np=preprocess_transform(image=image_np)['image'].transpose(2,0,1)

    image_torch = torch.tensor(preprocessed_image_np)


    image_torch = val_transforms(image_torch)

    print(image_torch.shape)
    #print(img_torch.dtype)

    tab_feats_torch=torch.tensor(tab_feats)

    print(tab_feats_torch.shape)

    pytorch_inference(pytorch_model_path,
                      tab_feats_torch,
                      image_torch)
    
    trt_inference(onnx_model_path,
                  tab_feats_torch,
                  image_torch)
                        

def main_fast(pytorch_model_path='./best_finetune_0.ckpt',
         onnx_model_path='./model.onnx',
         metadata='/dataset/train-metadata.csv',
         images_hdf5='/dataset/train-image.hdf5'):
    

    engine = build_engine_onnx(onnx_model_path)

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
    context = engine.create_execution_context()

    model_torch=isic_classifier.load_from_checkpoint(pytorch_model_path,
                                               config=config)
    
    metadata_df = pd.read_csv(metadata)

    for _ in range(10):
        idx=np.random.randint(len(metadata_df))

        isic_id=metadata_df.loc[idx, 'isic_id']

        print(f'isic_id: {isic_id}')

        tab_feats=metadata_df.loc[idx,tab_features]

        tab_feats=tab_feats.to_numpy().astype(np.float32)


        with h5py.File(images_hdf5,'r') as f:
            image_bytes=f[isic_id][()]
        
        pil_image=Image.open( io.BytesIO( image_bytes ))

        image_np=np.array(pil_image)
        
        preprocessed_image_np=preprocess_transform(image=image_np)['image'].transpose(2,0,1)

        image_torch = torch.tensor(preprocessed_image_np)


        image_torch = val_transforms(image_torch)

        #print(image_torch.shape)
        #print(img_torch.dtype)

        tab_feats_torch=torch.tensor(tab_feats)

        #print(tab_feats_torch.shape)

        with torch.no_grad():
            logits=model_torch(image_torch[None,...].cuda(),
                        tab_feats_torch[None,...].cuda())
        
        print("torch: ",logits)


        tab_feats_np=tab_feats_torch.numpy()
        image_np = image_torch.numpy()
        np.copyto(inputs[0].host, image_np.ravel())
        np.copyto(inputs[1].host, tab_feats_np.ravel())

        trt_outputs = common.do_inference(
                context,
                engine=engine,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
        print("trt:",trt_outputs[0])
        
    common.free_buffers(inputs, outputs, stream)
    


        




if __name__=='__main__':
    fire.Fire(main_fast)