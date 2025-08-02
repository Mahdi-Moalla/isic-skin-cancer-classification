"""
download model files from mlflow
"""

import os

import mlflow

if __name__ == '__main__':

    py_artifacts = [
        'trainer_config.json',
        'trainer/isic_model.py',
        'trainer/val_transform.json',
        'data_preprocessor/preprocess_transform.json',
    ]

    mlflow_server_url = os.getenv('mlflow_server_url')
    mlflow_experiment_name = os.getenv('mlflow_experiment_name')

    registry_model_name = os.getenv('registry_model_name')
    registry_model_version = os.getenv('registry_model_version')
    py_artifacts_run_id = os.getenv('py_artifacts_run_id')

    mlflow.set_tracking_uri(uri=mlflow_server_url)
    experiment = mlflow.set_experiment(mlflow_experiment_name)
    experiment_id = experiment.experiment_id

    MODEL_URI = f'models:/{registry_model_name}/{registry_model_version}/data/model.pth'

    model_checkpoint_file = mlflow.artifacts.download_artifacts(
        artifact_uri=MODEL_URI, dst_path='.'
    )

    os.rename(model_checkpoint_file, 'model.pth')

    for artifact_path in py_artifacts:
        artifact_uri = \
            f"mlflow-artifacts:/{experiment_id}"\
            + f"/{py_artifacts_run_id}/artifacts/{artifact_path}" # fmt: skip
        print(artifact_uri)
        mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path='.')
