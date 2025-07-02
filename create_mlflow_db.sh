#!/usr/bin/env bash
postgres_pod_name=$(kubectl get pod -n $1 | grep postgres|  cut -d' '  -f1)
echo $postgres_pod_name

echo "mlflow_pass
mlflow_pass
postgres" | kubectl exec --stdin --tty $postgres_pod_name\
 -- createuser -U postgres mlflow_user -S -D -R -P

 echo "postgres" | kubectl exec --stdin --tty $postgres_pod_name\
 -- createdb -U postgres mlflow_db  -O mlflow_user