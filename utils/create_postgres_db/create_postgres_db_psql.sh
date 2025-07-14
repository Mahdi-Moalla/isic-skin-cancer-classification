#!/usr/bin/env bash
cd "$(dirname "$0")"

postgres_pod_name=$(kubectl get pod -n $1 | grep postgres|  cut -d' '  -f1)
echo $postgres_pod_name

postgres_user=postgres
postgres_password=$(kubectl get secret -n $1 airflow-service-postgresql -o jsonpath="{.data.postgres-password}" | base64 -d)


user_name=$(kubectl get secret -n $1 $3 -o jsonpath="{.data.username}" | base64 -d)
user_password=$(kubectl get secret -n $1 $3 -o jsonpath="{.data.password}" | base64 -d)


kubectl exec --stdin $postgres_pod_name\
 -- /bin/bash -c "PGPASSWORD=$postgres_password psql -U postgres" <<EOF

SELECT 'CREATE USER $user_name LOGIN PASSWORD ''$user_password'' '
WHERE NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '$user_name')\gexec

SELECT 'CREATE DATABASE $2 WITH OWNER $user_name'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$2')\gexec
\q
EOF
