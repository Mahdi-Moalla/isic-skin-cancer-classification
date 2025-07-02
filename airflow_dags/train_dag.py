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



train_pvc_conf = """
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: traininig-pipeline-pvc
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
    pcv_create = KubernetesCreateResourceOperator(
        task_id="create_training_pipeline_pvc",
        yaml_conf=train_pvc_conf,
    )

