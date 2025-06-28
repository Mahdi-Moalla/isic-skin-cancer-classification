
from kubernetes.client import models as k8s

from airflow import DAG
from airflow.decorators import task

from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from airflow.providers.cncf.kubernetes.secret import Secret
from airflow.operators.bash import BashOperator

from airflow.providers.cncf.kubernetes.operators.resource import (
    KubernetesCreateResourceOperator,
    KubernetesDeleteResourceOperator,
)

import yaml

from pathlib import Path
import os
import os.path as osp

# docker run -it --runtime=nvidia --gpus all\
#      --ipc=host --ulimit memlock=-1\
#      --ulimit stack=67108864 -p 8888:8888\
#       nvcr.io/nvidia/pytorch:25.05-py3 python -m http.server --directory /
# nvidia-smi -l


pod_yaml_def="""
apiVersion: v1
kind: Pod
metadata:
  name: cuda-samples
  namespace: isic-skin-cancer-classification
spec:
  restartPolicy: OnFailure
  containers:
    - name: base     ## main container name must be base
      image: ubuntu
      command: ["nvidia-smi","-l"]
      resources:
        limits:
          nvidia.com/gpu: 1
"""




with DAG(
    dag_id="gpu_dag_example"
) as dag:
    script_path=str(Path(__file__).parent.resolve())
    k = KubernetesPodOperator(
        namespace="isic-skin-cancer-classification",
        name="gpu-pod",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(
                script_path,'gpu_pod.yml')).read_text()),
        task_id="task",
    )

    

    # https://github.com/apache/airflow/discussions/34033
    write_xcom_2 = KubernetesPodOperator(
        namespace="isic-skin-cancer-classification",
        image="python:3.9-slim-buster",
        cmds=["python", 
              "-c", 
              "import os; import time; import json; os.makedirs('/airflow/xcom/',exist_ok=True); "+
              "data={'v1':1,'v2':[1,2,3,4]}; "+
              "f = open('/airflow/xcom/return.json','w'); json.dump(data,f); f.close(); "+
              "print('done')"],
        name="write-xcom-2",
        do_xcom_push=True,
        task_id="write-xcom-2",
        get_logs=True,
    )
    
    @task.kubernetes(
        name="k8s_test_pull",
        image="python:3.9-slim-buster",
        task_id="k8s_xcom_pull_task",
        namespace="isic-skin-cancer-classification",
        env_vars=[k8s.V1EnvVar(name="XCOM_DATA",
                                  value="{{ task_instance.xcom_pull('write-xcom-2') }}")],
        get_logs=True,
    )
    def execute_in_k8s_pod_pull():
        import time
        import  json
        import os

        print(os.getenv('XCOM_DATA'))

    


    execute_in_k8s_pod_pull_instance = execute_in_k8s_pod_pull()
    write_xcom_2 >> execute_in_k8s_pod_pull_instance



    write_xcom = KubernetesPodOperator(
        namespace="isic-skin-cancer-classification",
        image="alpine",
        cmds=["sh", "-c", "mkdir -p /airflow/xcom/;echo '[1,2,3,4]' > /airflow/xcom/return.json"],
        name="write-xcom",
        do_xcom_push=True,
        on_finish_action="delete_pod",
        in_cluster=True,
        task_id="write-xcom",
        get_logs=True,
    )

    pod_task_xcom_result = BashOperator(
        bash_command="echo \"{{ task_instance.xcom_pull('write-xcom')[0] }}\"",
        task_id="pod_task_xcom_result",
    )

    write_xcom >> pod_task_xcom_result

    
    
