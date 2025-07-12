import mlflow
import fire
import torch
from isic_model import isic_classifier
from config import config

def generate_onnx(img_shape=(3,224,224),
                  output_file='./model.onnx'):

    model=isic_classifier.load_from_checkpoint('model.ckpt',
                                               config=config)
    
    input_img=torch.zeros(1,*img_shape, dtype=torch.float )

    input_tab_feats=torch.zeros(1,len(config.tab_features), 
                                dtype=torch.float)
    
    inputs=(input_img,input_tab_feats)

    model.to_onnx(output_file, inputs, export_params=True)


if __name__=='__main__':
    fire.Fire(generate_onnx)