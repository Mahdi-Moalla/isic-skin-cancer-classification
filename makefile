SHELL := /bin/bash

K8S_NAMESPACE:=isic-skin-cancer-classification

AIRFLOW_NAME:=airflow-service
MLFLOW_NAME:=mlflow-service
ADMINER_NAME:=adminer-service
KAFKA_NAME:=kafka-service
INFERENCE_WEBSERVER_NAME:=inference-webservice
GLOO_GATEWAY:=gloo-gateway
GRAFANA_NAME=grafana-service

kube_config = create

trainer_docker_image:=nvcr_pytorch_tensorrt_mod:latest
preprocessor_docker_image:=isic_preprocessor:latest
ubuntu_toolset_docker_image:=ubuntu_toolset:latest
webserver_docker_image:=webserver:latest
monitoring_docker_image:=monitoring_img:latest

mlflow_db_name:=mlflow_db
mlflow_db_secret_name:=mlflow-db-secret

monitoring_db_name:=monitoring_db
monitoring_db_secret_name:=monitoring-db-secret


inference_webserver_db_name:=inference_webserver_db
inference_webserver_db_secret_name:=inference-webserver-db-secret


init-microk8s:
	sudo snap install microk8s --classic --channel=1.32
	sudo usermod -a -G microk8s ${USER}
	mkdir -p ~/.kube
	chmod 0700 ~/.kube
	newgrp microk8s


init-kube-config:
ifeq ($(kube_config), update)
	microk8s config | tail -n +3  >> ~/.kube/config
else ifeq ($(kube_config), create)
	microk8s config > ~/.kube/config
endif

init-cluster:
	microk8s start
	bash utils/k8s_init/k8s_init.sh
	microk8s enable hostpath-storage
	microk8s  enable dns
	microk8s enable nvidia


#  https://discuss.kubernetes.io/t/microk8s-images-prune-utility-for-production-servers/15874
prune-storage:
	crictl -r unix:///var/snap/microk8s/common/run/containerd.sock rmi --prune

check-gpu-support:	
	microk8s kubectl logs -n gpu-operator-resources -lapp=nvidia-operator-validator -c nvidia-operator-validator

init-namespace:
	kubectl create ns ${K8S_NAMESPACE}
	kubectl config set-context --current --namespace=${K8S_NAMESPACE}

expose-dashboard:
	microk8s kubectl create token default
	microk8s kubectl port-forward -n kube-system service/kubernetes-dashboard 10443:443


delete-cluster:
	sudo microk8s reset --destroy-storage
	microk8s stop


init-images:
	#bash monitoring/build_docker_image.sh ${monitoring_docker_image}
	docker save -o monitoring_docker_image.tar  ${monitoring_docker_image} 
	microk8s images import < monitoring_docker_image.tar
	rm monitoring_docker_image.tar

	#bash utils/ubuntu_toolset/build_image.sh ${ubuntu_toolset_docker_image}
	docker save -o ubuntu_toolset_docker_image.tar  ${ubuntu_toolset_docker_image} 
	microk8s images import < ubuntu_toolset_docker_image.tar
	rm ubuntu_toolset_docker_image.tar

	#bash inference_webserver/webserver_docker_img/build_webserver_dockerfile.sh ${webserver_docker_image}
	docker save -o webserver_docker_image.tar  ${webserver_docker_image} 
	microk8s images import < webserver_docker_image.tar
	rm webserver_docker_image.tar


	#bash training_pipeline/preprocess_data/build_preprocessor_image.sh ${preprocessor_docker_image}
	docker save -o preprocessor_docker_image.tar  ${preprocessor_docker_image} 
	microk8s images import < preprocessor_docker_image.tar
	rm preprocessor_docker_image.tar

	#docker pull nvcr.io/nvidia/pytorch:25.05-py3
	#bash training_pipeline/trainer/build_trainer_image.sh ${trainer_docker_image}
	docker save -o trainer_docker_image.tar  ${trainer_docker_image} 
	microk8s images import < trainer_docker_image.tar
	rm trainer_docker_image.tar

	docker pull apache/airflow:2.10.5
	docker pull burakince/mlflow:3.1.1
	docker pull docker.io/bitnami/postgresql:16.1.0-debian-11-r15
	docker pull kafkace/kafka:v3.7.1-63ba8d2
	docker pull provectuslabs/kafka-ui:v0.7.2
	docker pull docker.io/adminer:4.8.1-standalone
	docker pull quay.io/solo-io/gloo:1.19.3
	docker pull docker.io/grafana/grafana:12.0.2

	docker save apache/airflow:2.10.5 > airflow.tar
	docker save docker.io/bitnami/postgresql:16.1.0-debian-11-r15 > postgresql.tar
	docker save burakince/mlflow:3.1.1 > mlflow.tar
	docker save kafkace/kafka:v3.7.1-63ba8d2 > kafka.tar
	docker save provectuslabs/kafka-ui:v0.7.2 > kafka_ui.tar
	docker save docker.io/adminer:4.8.1-standalone > adminer.tar
	docker save quay.io/solo-io/gloo:1.19.3 > gloo.tar
	docker save docker.io/grafana/grafana:12.0.2 > grafana.tar

	microk8s images import < airflow.tar
	microk8s images import < postgresql.tar
	microk8s images import < mlflow.tar
	microk8s images import < kafka.tar
	microk8s images import < kafka_ui.tar
	microk8s images import < adminer.tar
	microk8s images import < gloo.tar
	microk8s images import < grafana.tar

	rm airflow.tar postgresql.tar mlflow.tar kafka.tar kafka_ui.tar adminer.tar gloo.tar grafana.tar

init-airflow:
	helm repo add apache-airflow https://airflow.apache.org
	helm install ${AIRFLOW_NAME} apache-airflow/airflow \
		 --namespace ${K8S_NAMESPACE}\
		 --version 1.16.0\
		 -f kubernetes_files/airflow_values.yaml\
		 --debug

expose-airflow:
	kubectl port-forward --address 0.0.0.0 svc/${AIRFLOW_NAME}-webserver 8888:8080 --namespace ${K8S_NAMESPACE} &

expose-postgres:
	kubectl port-forward --address 0.0.0.0 svc/${AIRFLOW_NAME}-postgresql 5432:5432 --namespace ${K8S_NAMESPACE} &

remove-airflow:
	helm uninstall ${AIRFLOW_NAME}
	ps -C "kubectl port-forward --address 0.0.0.0 svc/airflow" -o pid= | xargs kill -9

init-dbs:
	kubectl apply -f kubernetes_files/mlflow_db_creds.yml  -n ${K8S_NAMESPACE}
	bash utils/postgres_db/create_postgres_db_psql.sh\
	 ${K8S_NAMESPACE} ${mlflow_db_name} ${mlflow_db_secret_name}
	kubectl apply -f kubernetes_files/monitoring_db_creds.yml\
	  -n ${K8S_NAMESPACE}
	bash utils/postgres_db/create_postgres_db_psql.sh ${K8S_NAMESPACE}\
		${monitoring_db_name} ${monitoring_db_secret_name}
	kubectl apply -f kubernetes_files/inference_webserver_db_creds.yml\
	  -n ${K8S_NAMESPACE}
	bash utils/postgres_db/create_postgres_db_psql.sh ${K8S_NAMESPACE}\
		${inference_webserver_db_name} ${inference_webserver_db_secret_name}
	bash utils/postgres_db/create_grafana_user.sh ${K8S_NAMESPACE} 



init-mlflow:
	kubectl apply -f kubernetes_files/mlflow_pvcs.yml -n ${K8S_NAMESPACE}
	helm repo add community-charts https://community-charts.github.io/helm-charts
	helm install ${MLFLOW_NAME} community-charts/mlflow\
	 --namespace ${K8S_NAMESPACE}\
	 -f kubernetes_files/mlflow_values.yaml\
	 --set backendStore.postgres.database=${mlflow_db_name}\
	 --set backendStore.existingDatabaseSecret.name=${mlflow_db_secret_name}\
	 --version 1.3.0

expose-mlflow:
	kubectl port-forward --address 0.0.0.0 svc/${MLFLOW_NAME} 5000:5000 --namespace ${K8S_NAMESPACE} &

remove-mlflow:
	helm uninstall ${MLFLOW_NAME}
	ps -C "kubectl port-forward --address 0.0.0.0 svc/mlflow" -o pid= | xargs kill -9
	kubectl delete -f  kubernetes_files/mlflow_pvcs.yml

init-dataset-http-server:
	bash utils/project_data_prepare/http_serve.sh 9000 split_dataset/ &


init-adminer:
	helm repo add startechnica https://startechnica.github.io/apps
	helm install ${ADMINER_NAME} startechnica/adminer\
		--namespace ${K8S_NAMESPACE}\
		--version 0.1.8

expose-adminer:
	kubectl port-forward --address 0.0.0.0 svc/${ADMINER_NAME} 8880:8080 --namespace ${K8S_NAMESPACE} &

init-kafka:
	helm repo add kafka https://helm-charts.itboon.top/kafka/
	helm install ${KAFKA_NAME} kafka/kafka\
	 --namespace ${K8S_NAMESPACE}\
	 -f kubernetes_files/kafka_values.yaml\
	 --version 18.0.1

expose-kafka-ui:
	kubectl port-forward --address 0.0.0.0 svc/${KAFKA_NAME}-ui 8088:8080 --namespace ${K8S_NAMESPACE} &


remove-kafka:
	helm uninstall ${KAFKA_NAME}
	ps -C "kubectl port-forward --address 0.0.0.0 svc/${KAFKA_NAME}" -o pid= | xargs kill -9

init-inference-webserver:
	helm upgrade -i ${INFERENCE_WEBSERVER_NAME}\
	 inference_webserver/helm_chart/inference_webserver\
	 --namespace ${K8S_NAMESPACE}\
	 -f inference_webserver/helm_chart/inference_webserver/values.yaml\
	 --set postgres_db.db_name=${inference_webserver_db_name}\
	 --set postgres_db.existing_db_secret.name=${inference_webserver_db_secret_name}\
	 -f kubernetes_files/inference_webserver_mlflow_artifacts.yml
	 

expose-inference-webserver:
	kubectl port-forward --address 0.0.0.0 svc/gloo-proxy-${INFERENCE_WEBSERVER_NAME}\
	 8080:8080 --namespace ${K8S_NAMESPACE} &


remove-inference-webservice:
	helm uninstall ${INFERENCE_WEBSERVER_NAME}
	ps -C "kubectl port-forward --address 0.0.0.0 svc/gloo-proxy-${INFERENCE_WEBSERVER_NAME}" -o pid= | xargs kill -9
	

init-gloo-gateway:
	kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.1/standard-install.yaml
	helm repo add gloo https://storage.googleapis.com/solo-public-helm
	helm install ${GLOO_GATEWAY} gloo/gloo \
	 --namespace ${K8S_NAMESPACE}\
	 --version 1.19.3 \
	 -f kubernetes_files/gloo_values.yml

remove-gloo-gateway:
	helm uninstall ${GLOO_GATEWAY}


init-grafana:
	helm repo add grafana https://grafana.github.io/helm-charts
	helm install ${GRAFANA_NAME} grafana/grafana\
	 --namespace ${K8S_NAMESPACE}\
	 -f kubernetes_files/grafana_values.yml\
	 --version 9.2.10

remove-grafana:
	helm uninstall ${GRAFANA_NAME}
	ps -C "kubectl port-forward --address 0.0.0.0\
	 svc/${GRAFANA_NAME}" -o pid= | xargs kill -9
	

expose-grafana:
	kubectl port-forward --address 0.0.0.0 svc/${GRAFANA_NAME}\
	 8008:80 --namespace ${K8S_NAMESPACE} &


init-apps: init-airflow init-dataset-http-server init-dbs init-adminer init-mlflow init-kafka init-gloo-gateway expose-airflow expose-adminer expose-mlflow expose-kafka-ui
	kubectl apply -f kubernetes_files/system_cfgmap.yml -n ${K8S_NAMESPACE}
	echo "apps installed"


expose-all: expose-airflow expose-mlflow expose-adminer expose-kafka-ui expose-postgres
	echo "all services exposed"