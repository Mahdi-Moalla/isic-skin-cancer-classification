K8S_NAMESPACE:=isic-skin-cancer-classification
CLUSTER_NAME:=isic-cluster
AIRFLOW_NAME:=airflow-service

create-cluster:
	kind create cluster --name  ${CLUSTER_NAME}\
	  --image kindest/node:v1.30.13\
	  --config kubernetes_files/cluster_config.yaml

delete-cluster:
	kind delete clusters ${CLUSTER_NAME}

init-namespace: create-cluster
	kubectl create ns ${K8S_NAMESPACE}
set-context: init-namespace
	kubectl config set-context --current --namespace=${K8S_NAMESPACE}

init-airflow:
	docker pull apache/airflow:2.10.5
	docker pull  docker.io/bitnami/postgresql:16.1.0-debian-11-r15
	kind load docker-image apache/airflow:2.10.5\
			docker.io/bitnami/postgresql:16.1.0-debian-11-r15\
			 --name ${CLUSTER_NAME}
	helm repo add apache-airflow https://airflow.apache.org
	helm install ${AIRFLOW_NAME} apache-airflow/airflow \
		 --namespace ${K8S_NAMESPACE}\
		 --version 1.16.0\
		 -f kubernetes_files/airflow_values.yaml\
		 --debug

redirect-airflow:
	kubectl port-forward --address 0.0.0.0 svc/${AIRFLOW_NAME}-webserver 8080:8080 --namespace ${K8S_NAMESPACE} &

remove-airflow:
	helm uninstall ${AIRFLOW_NAME}


