# pylint: disable=all
import os
import os.path as osp
from pathlib import Path

import yaml
from airflow import DAG
from airflow.decorators import task
from airflow.providers.cncf.kubernetes.operators.pod import \
    KubernetesPodOperator
from airflow.providers.cncf.kubernetes.operators.resource import (
    KubernetesCreateResourceOperator, KubernetesDeleteResourceOperator)
from kubernetes.client import models as k8s

pvc_conf = """
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: test-volume
  namespace: isic-skin-cancer-classification
spec:
  accessModes: [ReadWriteOnce]
  resources: { requests: { storage: 1Gi } }
"""


python_pod = """
apiVersion: v1
kind: Pod
metadata:
  name: python-pod
  namespace: isic-skin-cancer-classification
spec:
  restartPolicy: OnFailure
  containers:
    - name: base
      image: python:3.9-slim-buster
      env:
      - name: envv1
        value: "envv1"
      command: ["python"]
      args:
        - -c
        - >-
            import  json;
            import os;
            print(os.environ);
            print('xcom push');
            f_xcom=open('/airflow/xcom/return.json','w');
            data={'v':1,
                  'x':'hello',
                  'z':[1,2,3,4,5]};
            json.dump(data,f_xcom);
            f_xcom.close();
"""


with DAG(
    dag_id="gpu_pod_xcom_dag_example",
    params={"param1": 1, "param2": "abcdefgh"},
) as dag:
    script_path = str(Path(__file__).parent.resolve())
    '''
    k = KubernetesPodOperator(
        namespace="isic-skin-cancer-classification",
        name="gpu-pod",
        pod_template_dict=yaml.safe_load(
            Path(osp.join(
                script_path,'gpu_pod_example.yml')).read_text()),
        task_id="task",
    )

    t1 = KubernetesCreateResourceOperator(
        task_id="create_pvc",
        yaml_conf=pvc_conf,
    )

    t2 = KubernetesDeleteResourceOperator(
        task_id="delete_pvc",
        yaml_conf=pvc_conf,
    )

    t1 >> t2
    '''

    python_task = KubernetesPodOperator(
        namespace="isic-skin-cancer-classification",
        name="python-pod",
        pod_template_dict=yaml.safe_load(python_pod),
        task_id="python_pod",
        env_vars=[k8s.V1EnvVar(name="envv2", value="envv2")],
        do_xcom_push=True,
        get_logs=True,
    )

    # https://github.com/apache/airflow/discussions/34033
    write_xcom_2 = KubernetesPodOperator(
        namespace="isic-skin-cancer-classification",
        image="python:3.9-slim-buster",
        cmds=[
            "python",
            "-c",
            "import os; import time; import json; os.makedirs('/airflow/xcom/',exist_ok=True); "
            + "data={'v1':1,'v2':[1,2,3,4]}; "
            + "f = open('/airflow/xcom/return.json','w'); json.dump(data,f); f.close(); "
            + "print('done')",
        ],
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
        env_vars=[
            k8s.V1EnvVar(
                name="XCOM_DATA1", value="{{ task_instance.xcom_pull('write-xcom-2') }}"
            ),
            k8s.V1EnvVar(
                name="XCOM_DATA2", value="{{ task_instance.xcom_pull('python_pod') }}"
            ),
        ],
        get_logs=True,
    )
    def execute_in_k8s_pod_pull():
        import json
        import os
        import time

        print(os.getenv('XCOM_DATA1'))
        print(os.getenv('XCOM_DATA2'))

    execute_in_k8s_pod_pull_instance = execute_in_k8s_pod_pull()
    write_xcom_2 >> execute_in_k8s_pod_pull_instance
    python_task >> execute_in_k8s_pod_pull_instance
