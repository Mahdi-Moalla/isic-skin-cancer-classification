K8S_NAMESPACE:=isic-skin-cancer-classification
AIRFLOW_NAME:=AIRFLOW_SERVICE

init-namespace:
	kubectl create ns ${K8S_NAMESPACE}
	kubectl config set-context --current --namespace=${K8S_NAMESPACE}

init-airflow:
	helm install ${AIRFLOW_NAME} apache-airflow/airflow\
		 --namespace ${K8S_NAMESPACE}\
		 -f kubernetes_files/airflow_values.yaml

remove-airflow:
	helm uninstall ${AIRFLOW_NAME}


