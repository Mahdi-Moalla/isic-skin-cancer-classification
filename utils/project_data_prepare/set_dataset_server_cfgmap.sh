#!/usr/bin/env bash
control_node_ip=$(microk8s config | grep "server: https://" | tr -d "//"  | cut -d ":" -f3)
echo $control_node_ip
kubectl create configmap dataset-server-cfgmap \
 --from-literal=dataset_server=http://$control_node_ip:$2 \
 -n $1