import  os

import mlflow

if __name__=='__main__':

    mlflow_server_url=os.getenv('mlflow_server_url')
    mlflow_experiment_name=os.getenv('mlflow_experiment_name')
    
    model_checkpoint_uri=os.getenv('model_checkpoint_uri')
    model_def_uri=os.getenv('model_def_uri')
    model_config_uri=os.getenv('model_config_uri')
    preprocessor_def_uri=os.getenv('preprocessor_def_uri')
    
    mlflow.set_tracking_uri(uri=mlflow_server_url)
    mlflow.set_experiment(mlflow_experiment_name)

    model_checkpoint_file=mlflow.artifacts.download_artifacts(artifact_uri=model_checkpoint_uri,
                                        dst_path ='.')
    
    os.rename(model_checkpoint_file, 'model.ckpt')

    mlflow.artifacts.download_artifacts(artifact_uri=model_def_uri,
                                        dst_path ='.')
    
    mlflow.artifacts.download_artifacts(artifact_uri=model_config_uri,
                                        dst_path ='.')
    mlflow.artifacts.download_artifacts(artifact_uri=preprocessor_def_uri,
                                        dst_path ='.')