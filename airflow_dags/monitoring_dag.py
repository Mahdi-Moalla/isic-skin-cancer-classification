"""
Airflow monitoring dag
"""

# pylint: disable=import-error
import os.path as osp
from pathlib import Path

import yaml
from airflow import DAG
from airflow.models.param import Param
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.operators.resource import (
    KubernetesCreateResourceOperator,
    KubernetesDeleteResourceOperator,
)
from kubernetes.client import models as k8s

NAMESPACE = "isic-skin-cancer-classification"


MONITORING_CFG_MAP = f"""
apiVersion: v1
kind: ConfigMap
metadata:
  name: monitoring-cfgmap
  namespace: {NAMESPACE}
data:
  mlflow_server_url: "http://mlflow-service:5000"
  mlflow_experiment_name: "isic-skin-cancer-classification"
  inference_webserver_host: "http://gloo-proxy-inference-webservice:8080"
  postgres_server: "airflow-service-postgresql"
  postgres_port: "5432"
  postgres_db_name: "monitoring_db"
"""

with DAG(dag_id="monitoring_cleaner") as dag:
    cfgmap_delete = KubernetesDeleteResourceOperator(
        task_id="delete_monitoring_cfgmap",
        yaml_conf=MONITORING_CFG_MAP,
    )

with DAG(
    dag_id="monitoring_dag",
    params={
        "run_type": Param(
            "single",
            type="string",
            title="run_type",
            enum=["single", "range"],
        ),
        "start_date": Param(
            "2025-01-01",
            type="string",
            format="date",
            title="start_date",
            description="""Please select a date and time,
              use the button on the left for
              a pop-up calendar.""",
        ),
        "end_date": Param(
            "2025-01-01",
            type="string",
            format="date",
            title="end_date",
            description="""Please select a date and time,
              use the button on the left for
              a pop-up calendar.""",
        ),
        "run_id": Param(
            "",
            # In this example we have no default value
            # Form will enforce a value supplied by users to be able to trigger
            type="string",
            title="run_id",
            description="This field is required. You can not submit without having value in here.",
        ),
    },
) as dag:
    DAG_FILE_PATH = str(Path(__file__).parent.resolve())

    cfgmap_create = KubernetesCreateResourceOperator(
        task_id="create_monitoring_cfgmap",
        yaml_conf=MONITORING_CFG_MAP,
    )

    monitoring_task = KubernetesPodOperator(
        namespace=NAMESPACE,
        name="monitoring-pod",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(DAG_FILE_PATH, 'monitoring_pod.yml')).read_text(encoding="utf-8")
        ),
        # on_finish_action="keep_pod",
        task_id="monitoring-task",
        env_vars=[
            k8s.V1EnvVar(name=k, value="{{ params[\"" + k + "\"] }}") for k in dag.params.keys()
        ],
        get_logs=True,
    )

    cfgmap_delete = KubernetesDeleteResourceOperator(
        task_id="delete_monitoring_cfgmap",
        yaml_conf=MONITORING_CFG_MAP,
    )

    # pylint: disable=pointless-statement
    cfgmap_create >> monitoring_task >> cfgmap_delete
