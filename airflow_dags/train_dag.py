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
  dataset_http_server: "192.168.1.8"
  dataset_http_server_port: "9000"
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
    dag_id="training_pipeline"#,
    #on_failure_callback=trigger_cleaner_dag,
) as dag:
    script_path=str(Path(__file__).parent.resolve())


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
    
    

    data_downloader = KubernetesPodOperator(
        namespace=namespace,
        name="data_downloader",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(
                script_path,'data_download_pod.yml')).read_text()),
        task_id="data-downloader",
        get_logs=True
    )

    '''
    data_downloader = KubernetesPodOperator(
        namespace=namespace,
        name="data_downloader",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(
                script_path,'data_download_pod.yml')).read_text()),
        task_id="data-downloader",
        get_logs=True
    )
    '''

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

    resources_allocator_group() >> data_downloader
    data_downloader  >>  resources_cleaner_group()
    

    





