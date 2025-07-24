# pylint: disable=import-error
from kubernetes.client import models as k8s 

from airflow import DAG
from airflow.models.param import Param, ParamsDict
from airflow.decorators import task

from airflow.decorators import task_group

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator

from airflow.providers.cncf.kubernetes.operators.resource import (
    KubernetesCreateResourceOperator,
    KubernetesDeleteResourceOperator,
)

from airflow.api.client.local_client import Client

import yaml
import  json
from pathlib import Path
import os
import os.path as osp

from datetime import datetime

namespace="isic-skin-cancer-classification"


monitoring_cfg_map=f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: monitoring-cfgmap
  namespace: {namespace}
data:
  mlflow_server_url: "http://mlflow-service:5000"
  mlflow_experiment_name: "isic-skin-cancer-classification"
  inference_webserver_host: "http://gloo-proxy-inference-webservice:8080"
  postgres_server: "airflow-service-postgresql"
  postgres_port: "5432"
  postgres_db_name: "monitoring_db"
"""

with DAG(
    dag_id="monitoring_cfgmap_cleaner"
    
) as dag:
    cfgmap_delete = KubernetesDeleteResourceOperator(
            task_id="delete_monitoring_cfgmap",
            yaml_conf=monitoring_cfg_map,
    )

with DAG(
    dag_id="monitoring_dag",
    params={
        "run_type": Param(
            "single",
            type="string",
            title="run_type",
            enum=["single","range"],
        ),
        "start_date": Param(
            f"2025-01-01",
            type="string",
            format="date",
            title="start_date",
            description="Please select a date and time, use the button on the left for a pop-up calendar.",
        ),
        "end_date":  Param(
            f"2025-01-01",
            type="string",
            format="date",
            title="end_date",
            description="Please select a date and time, use the button on the left for a pop-up calendar.",
        ),
        "fold_run_id": Param("",
            # In this example we have no default value
            # Form will enforce a value supplied by users to be able to trigger
            type="string",
            title="fold_run_id",
            description="This field is required. You can not submit without having value in here.",
        ),
    }
) as dag:
    dag_file_path=str(Path(__file__).parent.resolve())

    cfgmap_create = KubernetesCreateResourceOperator(
            task_id="create_monitoring_cfgmap",
            yaml_conf=monitoring_cfg_map,
        )
    
    monitoring_task = KubernetesPodOperator(
        namespace=namespace,
        name="monitoring-pod",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(
                dag_file_path,'monitoring_pod.yml')).read_text()),
        #on_finish_action="keep_pod",
        task_id="monitoring-task",
        env_vars=[k8s.V1EnvVar(name="dag_params",value="{{ params }}")],
        get_logs=True
    )

    cfgmap_delete = KubernetesDeleteResourceOperator(
            task_id="delete_monitoring_cfgmap",
            yaml_conf=monitoring_cfg_map,
    )

    cfgmap_create >> monitoring_task >> cfgmap_delete
    
