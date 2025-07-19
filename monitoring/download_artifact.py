import  os

import mlflow

if __name__=='__main__':

    mlflow_server_url='http://legion-pro-7-16arx8h:5000/'
    mlflow_experiment_name='isic-skin-cancer-classification'
    
    artifact_uri='mlflow-artifacts:/1/b07ab1407928453e8af1b9530083be95/artifacts/trainer/config.py'
    
    mlflow.set_tracking_uri(uri=mlflow_server_url)
    experiment=mlflow.set_experiment(mlflow_experiment_name)

    print(experiment.experiment_id)

    #from IPython import embed as idbg; idbg(colors='Linux')

    mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri,
                                        dst_path ='.')
    
    model='models:/isic-skin-cancer-classification-model/1/data/model.pth'

    mlflow.artifacts.download_artifacts(artifact_uri=model,
                                        dst_path ='model')
    
    

    