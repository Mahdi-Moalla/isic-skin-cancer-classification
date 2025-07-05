#!/usr/bin/env bash

postgres_pod_name=$(kubectl get pod -n $1 | grep postgres|  cut -d' '  -f1)
echo $postgres_pod_name

postgres_user=postgres
postgres_password=$(kubectl get secret -n $1 airflow-service-postgresql -o jsonpath="{.data.postgres-password}" | base64 -d)

db_user=$(kubectl get secret -n $1 $3 -o jsonpath="{.data.username}" | base64 -d)
db_password=$(kubectl get secret -n $1 $3 -o jsonpath="{.data.password}" | base64 -d)


echo "$db_password
$db_password
$postgres_password" | kubectl exec --stdin --tty $postgres_pod_name\
 -- createuser -U $postgres_user $db_user -S -D -R -P

echo $postgres_password | kubectl exec --stdin --tty $postgres_pod_name\
 -- createdb -U $postgres_user $2  -O $db_user