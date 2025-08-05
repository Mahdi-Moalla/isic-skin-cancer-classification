"""
Airflow training pipeline dag
"""

# pylint: disable=import-error
import os.path as osp
from pathlib import Path

import yaml
from airflow import DAG

# from airflow.decorators import task
from airflow.decorators import task_group
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.operators.resource import (
    KubernetesCreateResourceOperator,
    KubernetesDeleteResourceOperator,
)
from kubernetes.client import models as k8s

NAMESPACE = "isic-skin-cancer-classification"


TRAIN_PVC_CONF = f"""
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: traininig-pipeline-pvc
  namespace: {NAMESPACE}
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
"""

TRAIN_CFG_MAP = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: training-pipeline-cfgmap
  namespace: {NAMESPACE}
data:
  code_repo: "https://github.com/Mahdi-Moalla/isic-skin-cancer-classification"
  dataset_folder: "dataset"
  preprocessed_dataset_folder: "preprocessed_dataset"
  mlflow_server_url: "http://mlflow-service:5000"
"""
#  mlflow_experiment_name: "isic-skin-cancer-classification"

with DAG(dag_id="training_pipeline_cleaner") as dag:
    pvc_delete = KubernetesDeleteResourceOperator(
        task_id="delete_training_pipeline_pvc",
        yaml_conf=TRAIN_PVC_CONF,
    )
    cfgmap_delete = KubernetesDeleteResourceOperator(
        task_id="delete_training_pipeline_cfgmap",
        yaml_conf=TRAIN_CFG_MAP,
    )


with DAG(
    dag_id="training_pipeline"
    # on_failure_callback=trigger_cleaner_dag,
) as dag:
    DAG_FILE_PATH = str(Path(__file__).parent.resolve())

    @task_group()
    def resources_allocator_group():
        """
        task group for applying pvc and configmap
        """
        pvc_create = KubernetesCreateResourceOperator(  # pylint: disable=unused-variable
            task_id="create_training_pipeline_pvc",
            yaml_conf=TRAIN_PVC_CONF,
        )

        cfgmap_create = KubernetesCreateResourceOperator(  # pylint: disable=unused-variable
            task_id="create_training_pipeline_cfgmap",
            yaml_conf=TRAIN_CFG_MAP,
        )

    @task_group()
    def resources_cleaner_group():
        """
        task group to delete pvc and configmap
        """
        pvc_delete = KubernetesDeleteResourceOperator(  # pylint: disable=unused-variable, redefined-outer-name
            task_id="delete_training_pipeline_pvc",
            yaml_conf=TRAIN_PVC_CONF,
        )

        cfgmap_delete = KubernetesDeleteResourceOperator(  # pylint: disable=unused-variable, redefined-outer-name
            task_id="delete_training_pipeline_cfgmap",
            yaml_conf=TRAIN_CFG_MAP,
        )

    data_downloader = KubernetesPodOperator(
        namespace=NAMESPACE,
        name="data_downloader",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(DAG_FILE_PATH, 'data_download_pod.yml')).read_text(encoding="utf-8")
        ),
        task_id="data-downloader",
        get_logs=True,
    )

    init_mlflow_run = KubernetesPodOperator(
        namespace=NAMESPACE,
        name="init_mlflow_run",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(DAG_FILE_PATH, 'init_mlflow_tracking_pod.yml')).read_text(
                encoding="utf-8"
            )
        ),
        task_id="init-mlflow-run",
        do_xcom_push=True,
        get_logs=True,
    )

    data_preprocessor = KubernetesPodOperator(
        namespace=NAMESPACE,
        name="data_preprocessor",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(DAG_FILE_PATH, 'data_preprocess_pod.yml')).read_text(encoding="utf-8")
        ),
        task_id="data-preprocessor",
        do_xcom_push=True,
        env_vars=[
            k8s.V1EnvVar(
                name="run_context",
                value="{{ task_instance.xcom_pull('init-mlflow-run') }}",
            )
        ],
        get_logs=True,
    )

    trainer = KubernetesPodOperator(
        namespace=NAMESPACE,
        name="trainer",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(DAG_FILE_PATH, 'trainer_pod.yml')).read_text(encoding="utf-8")
        ),
        task_id="trainer",
        do_xcom_push=True,
        env_vars=[
            k8s.V1EnvVar(
                name="run_context",
                value="{{ task_instance.xcom_pull('data-preprocessor') }}",
            )
        ],
        get_logs=True,
    )

    # pylint: disable=pointless-statement, expression-not-assigned
    resources_allocator_group() >> data_downloader
    data_downloader >> init_mlflow_run >> data_preprocessor >> trainer
    trainer >> resources_cleaner_group()
