K8S_NAMESPACE:=isic-skin-cancer-classification
CLUSTER_NAME:=isic-cluster
AIRFLOW_NAME:=airflow-service
MLFLOW_NAME:=mlflow-service
kube_config = create

trainer_docker_image:=nvcr_pytorch_tensorrt_mod:latest
preprocessor_docker_image:=isic_preprocessor:latest
ubuntu_toolset_docker_image:=localhost:32000/ubuntu_toolset:latest

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

setup-cluster:
	microk8s start
	microk8s status --wait-ready
	microk8s enable hostpath-storage
	microk8s enable nvidia
	microk8s enable registry:size=50Gi

check-gpu-support:	
	microk8s kubectl logs -n gpu-operator-resources -lapp=nvidia-operator-validator -c nvidia-operator-validator

init-namespace:
	kubectl create ns ${K8S_NAMESPACE}
	kubectl config set-context --current --namespace=${K8S_NAMESPACE}

delete-cluster:
	sudo microk8s reset --destroy-storage
	microk8s stop

microk8s-init-images-1:

	bash utils/ubuntu_toolset/build_image.sh ${ubuntu_toolset_docker_image}
	docker push ${ubuntu_toolset_docker_image}
	microk8s ctr images pull ${ubuntu_toolset_docker_image} 
	#docker save -o ubuntu_toolset_docker_image.tar ${ubuntu_toolset_docker_image} 
	#rm ubuntu_toolset_docker_image.tar

	#bash training_pipeline/preprocess_data/build_preprocessor_image.sh  ${preprocessor_docker_image}
	#docker save ${preprocessor_docker_image} > preprocessor_docker_image.tar
	#microk8s images import < preprocessor_docker_image.tar
	#rm preprocessor_docker_image.tar
	
	#docker pull nvcr.io/nvidia/pytorch:25.05-py3
	#bash training_pipeline/trainer/build_trainer_image.sh  ${trainer_docker_image}
	#docker save ${trainer_docker_image} > trainer_docker_image.tar
	#microk8s images import < trainer_docker_image.tar
	#rm trainer_docker_image.tar


	docker pull apache/airflow:2.10.5
	docker pull burakince/mlflow:3.1.1
	docker pull docker.io/bitnami/postgresql:16.1.0-debian-11-r15

	docker save apache/airflow:2.10.5 > airflow.tar
	docker save docker.io/bitnami/postgresql:16.1.0-debian-11-r15 > postgresql.tar
	docker save burakince/mlflow:3.1.1 > mlflow.tar

	microk8s images import < airflow.tar
	microk8s images import < postgresql.tar
	microk8s images import < mlflow.tar

	rm airflow.tar postgresql.tar mlflow.tar

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


init-mlflow:
	#bash create_db.sh ${K8S_NAMESPACE}	
	kubectl apply -f kubernetes_files/mlflow_pvcs.yml
	helm repo add community-charts https://community-charts.github.io/helm-charts
	helm install ${MLFLOW_NAME} community-charts/mlflow\
	 --namespace ${K8S_NAMESPACE}\
	 -f kubernetes_files/mlflow_values.yaml\
	 --version 1.3.0

expose-mlflow:
	kubectl port-forward --address 0.0.0.0 svc/${MLFLOW_NAME} 5000:5000 --namespace ${K8S_NAMESPACE} &

remove-mlflow:
	helm uninstall ${MLFLOW_NAME}
	ps -C "kubectl port-forward --address 0.0.0.0 svc/mlflow" -o pid= | xargs kill -9
	kubectl delete -f  kubernetes_files/mlflow_pvcs.yml


