import  os

import mlflow



if __name__=='__main__':

    py_artifacts=['trainer/isic_model.py',
                  'trainer/config.py',
                  'trainer/transforms.py',
                  'data_preprocessor/preprocess_data.py']

    mlflow_server_url=os.getenv('mlflow_server_url')
    mlflow_experiment_name=os.getenv('mlflow_experiment_name')
    
    registry_model_name=os.getenv('registry_model_name')
    registry_model_version=os.getenv('registry_model_version')
    py_artifacts_run_id=os.getenv('py_artifacts_run_id')
    


    mlflow.set_tracking_uri(uri=mlflow_server_url)
    experiment=mlflow.set_experiment(mlflow_experiment_name)
    experiment_id=experiment.experiment_id


    model_uri=f'models:/{registry_model_name}/{registry_model_version}/data/model.pth'

    model_checkpoint_file=mlflow.artifacts.download_artifacts(artifact_uri=model_uri,
                                        dst_path ='.')
    
    os.rename(model_checkpoint_file, 'model.pth')


    for  artifact_path in py_artifacts:
        artifact_uri=f"mlflow-artifacts:/{experiment_id}/{py_artifacts_run_id}/artifacts/{artifact_path}"

        mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            dst_path ='.')
    
    