from kubernetes.client import models as k8s

from airflow import DAG
#from airflow.decorators import task
from airflow.decorators import task_group

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

from airflow.providers.cncf.kubernetes.operators.resource import (
    KubernetesCreateResourceOperator,
    KubernetesDeleteResourceOperator,
)

from airflow.api.client.local_client import Client

import yaml

from pathlib import Path
import os
import os.path as osp

import datetime

namespace="isic-skin-cancer-classification"


train_pvc_conf = f"""
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: traininig-pipeline-pvc
  namespace: {namespace}
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
"""

train_cfg_map=f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-pipeline-cfgmap
  namespace: {namespace}
data:
  original_dataset_http_server: "http://192.168.1.8:9000"
  code_repo: "https://github.com/Mahdi-Moalla/isic-skin-cancer-classification"
  dataset_folder: "dataset"
  preprocessed_dataset_folder: "preprocessed_dataset"
  mlflow_server_url: "http://mlflow-service:5000"
  mlflow_experiment_name: "isic-skin-cancer-classification"
"""

with DAG(
    dag_id="training_pipeline_cleaner"
) as dag:
    pcv_delete = KubernetesDeleteResourceOperator(
        task_id="delete_training_pipeline_pvc",
        yaml_conf=train_pvc_conf,
    )
    cfgmap_delete = KubernetesDeleteResourceOperator(
        task_id="delete_training_pipeline_cfgmap",
        yaml_conf=train_cfg_map,
    )




def trigger_cleaner_dag(context):
    c = Client(None, None)
    c.trigger_dag(dag_id='training_pipeline_cleaner', 
                  run_id=f'cleaner_run_{str(datetime.datetime.now())}')

with DAG(
    dag_id="training_pipeline"
    #on_failure_callback=trigger_cleaner_dag,
) as dag:
    dag_file_path=str(Path(__file__).parent.resolve())


    @task_group()
    def resources_allocator_group():
        pvc_create = KubernetesCreateResourceOperator(
            task_id="create_training_pipeline_pvc",
            yaml_conf=train_pvc_conf,
        )

        cfgmap_create = KubernetesCreateResourceOperator(
            task_id="create_training_pipeline_cfgmap",
            yaml_conf=train_cfg_map,
        )
    
    @task_group()
    def resources_cleaner_group():
        pvc_delete = KubernetesDeleteResourceOperator(
            task_id="delete_training_pipeline_pvc",
            yaml_conf=train_pvc_conf,
        )

        cfgmap_delete = KubernetesDeleteResourceOperator(
            task_id="delete_training_pipeline_cfgmap",
            yaml_conf=train_cfg_map,
        )
    

    data_downloader = KubernetesPodOperator(
        namespace=namespace,
        name="data_downloader",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(
                dag_file_path,'data_download_pod.yml')).read_text()),
        task_id="data-downloader",
        get_logs=True
    )

    init_mlflow_run = KubernetesPodOperator(
        namespace=namespace,
        name="init_mlflow_run",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(
                dag_file_path,'init_mlflow_tracking_pod.yml')).read_text()),
        task_id="init-mlflow-run",
        do_xcom_push=True,
        get_logs=True
    )
    

    data_preprocessor = KubernetesPodOperator(
        namespace=namespace,
        name="data_preprocessor",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(
                dag_file_path,'data_preprocess_pod.yml')).read_text()),
        task_id="data-preprocessor",
        do_xcom_push=True,
        env_vars=[k8s.V1EnvVar(name="run_context",
                                  value="{{ task_instance.xcom_pull('init-mlflow-run') }}")],
        get_logs=True
    )

    trainer = KubernetesPodOperator(
        namespace=namespace,
        name="trainer",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(
                dag_file_path,'trainer_pod.yml')).read_text()),
        task_id="trainer",
        on_finish_action="keep_pod",
        do_xcom_push=True,
        env_vars=[k8s.V1EnvVar(name="run_context",
                                value="{{ task_instance.xcom_pull('data-preprocessor') }}")],
        get_logs=True
    )


    resources_allocator_group() >> data_downloader
    data_downloader >> init_mlflow_run >> data_preprocessor >> trainer  
    trainer >> resources_cleaner_group()
    
