K8S_NAMESPACE:=isic-skin-cancer-classification
CLUSTER_NAME:=isic-cluster
AIRFLOW_NAME:=airflow-service
kube_config = create

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

create-cluster:
	microk8s start
	microk8s status --wait-ready
	microk8s enable hostpath-storage

enable-gpu-support: create-cluster
	microk8s enable nvidia
	microk8s kubectl logs -n gpu-operator-resources -lapp=nvidia-operator-validator -c nvidia-operator-validator
	
init-namespace: enable-gpu-support
	kubectl create ns ${K8S_NAMESPACE}
set-context: init-namespace
	kubectl config set-context --current --namespace=${K8S_NAMESPACE}

setup-cluster: set-context
	echo "cluster setup finished"

delete-cluster:
	sudo microk8s reset --destroy-storage
	microk8s stop

check-gpu-support:	
	microk8s kubectl logs -n gpu-operator-resources -lapp=nvidia-operator-validator -c nvidia-operator-validator


microk8s-init-images:
	docker pull apache/airflow:2.10.5
	docker pull docker.io/bitnami/postgresql:16.1.0-debian-11-r15
	docker save apache/airflow:2.10.5 > airflow.tar
	docker save docker.io/bitnami/postgresql:16.1.0-debian-11-r15 > postgresql.tar
	microk8s images import < airflow.tar
	microk8s images import < postgresql.tar
	rm airflow.tar postgresql.tar

init-airflow:
	helm repo add apache-airflow https://airflow.apache.org
	helm install ${AIRFLOW_NAME} apache-airflow/airflow \
		 --namespace ${K8S_NAMESPACE}\
		 --version 1.16.0\
		 -f kubernetes_files/airflow_values.yaml\
		 --debug

expose-airflow:
	kubectl port-forward --address 0.0.0.0 svc/${AIRFLOW_NAME}-webserver 8080:8080 --namespace ${K8S_NAMESPACE} &

remove-airflow:
	helm uninstall ${AIRFLOW_NAME}
	ps -C "kubectl port-forward --address 0.0.0.0 svc/airflow" -o pid= | xargs kill -9




init-nvidia-plugin:
	docker pull nvcr.io/nvidia/k8s-device-plugin:v0.17.1
	kind load docker-image nvcr.io/nvidia/k8s-device-plugin:v0.17.1\
			 --name ${CLUSTER_NAME}
	kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.1/deployments/static/nvidia-device-plugin.yml
	
init-nvidia-plugin-helm:
	helm repo add nvdp https://nvidia.github.io/k8s-device-plugin
	helm repo update
	helm upgrade -i nvdp nvdp/nvidia-device-plugin \
		--namespace nvidia-device-plugin \
		--create-namespace \
		--version 0.17.1
