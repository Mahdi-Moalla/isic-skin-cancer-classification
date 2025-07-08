import mlflow
import fire
import torch

def generate_onnx(mlflow_server_url="http://192.168.1.8:5000",
                  mlflow_experiment_name="isic-skin-cancer-classification",
                  model_checkpoint_uri='mlflow-artifacts:/1/9fc00423c95b4d84844bcb5fa05e83fd/artifacts/best_finetune/best_finetune_0.ckpt',
                  model_def_uri='mlflow-artifacts:/1/f5c4f8f93d1c4baaae6a0b746252db60/artifacts/trainer/isic_model.py',
                  model_config_uri='mlflow-artifacts:/1/f5c4f8f93d1c4baaae6a0b746252db60/artifacts/trainer/config.py',
                  img_shape=(3,224,224),
                  output_file='./model.onnx'):
    
    mlflow.set_tracking_uri(uri=mlflow_server_url)
    mlflow.set_experiment(mlflow_experiment_name)

    checkpoint_path=mlflow.artifacts.download_artifacts(artifact_uri=model_checkpoint_uri,
                                        dst_path ='.')
    
    mlflow.artifacts.download_artifacts(artifact_uri=model_def_uri,
                                        dst_path ='.')
    
    mlflow.artifacts.download_artifacts(artifact_uri=model_config_uri,
                                        dst_path ='.')
    

    from isic_model import isic_classifier
    from config import config

    print(checkpoint_path)
    model=isic_classifier.load_from_checkpoint(checkpoint_path,
                                               config=config)
    
    input_img=torch.zeros(1,*img_shape, dtype=torch.float )

    input_tab_feats=torch.zeros(1,len(config.tab_features), 
                                dtype=torch.float)
    
    inputs=(input_img,input_tab_feats)

    model.to_onnx(output_file, inputs, export_params=True)


if __name__=='__main__':
    fire.Fire(generate_onnx)