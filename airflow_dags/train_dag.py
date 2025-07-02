from kubernetes.client import models as k8s

from airflow import DAG
from airflow.decorators import task

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

from airflow.providers.cncf.kubernetes.operators.resource import (
    KubernetesCreateResourceOperator,
    KubernetesDeleteResourceOperator,
)

import yaml

from pathlib import Path
import os
import os.path as osp


namespace="isic-skin-cancer-classification"


train_pvc_conf = f"""
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: traininig-pipeline-pvc
  namespace: {namespace}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
"""


with DAG(
    dag_id="training_pipeline_cleaner"
) as dag:
    pcv_delete = KubernetesDeleteResourceOperator(
        task_id="delete_training_pipeline_pvc",
        yaml_conf=train_pvc_conf,
    )

with DAG(
    dag_id="training_pipeline"
) as dag:
    script_path=str(Path(__file__).parent.resolve())

    pcv_create = KubernetesCreateResourceOperator(
        task_id="create_training_pipeline_pvc",
        yaml_conf=train_pvc_conf,
    )

    data_downloader = KubernetesPodOperator(
        namespace=namespace,
        name="data_downloader",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(
                script_path,'data_download_pod.yml')).read_text()),
        task_id="data-downloader",
    )

    pcv_delete = KubernetesDeleteResourceOperator(
        task_id="delete_training_pipeline_pvc",
        yaml_conf=train_pvc_conf,
    )

    pcv_create >> data_downloader >> pcv_delete





