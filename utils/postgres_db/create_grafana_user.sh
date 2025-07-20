#!/usr/bin/env bash
cd "$(dirname "$0")"

postgres_pod_name=$(kubectl get pod -n $1 | grep postgres|  cut -d' '  -f1)
echo $postgres_pod_name

postgres_user=postgres
postgres_password=$(kubectl get secret -n $1 airflow-service-postgresql -o jsonpath="{.data.postgres-password}" | base64 -d)


kubectl exec --stdin $postgres_pod_name\
 -- /bin/bash -c "PGPASSWORD=$postgres_password psql -U postgres -d monitoring_db" <<EOF

SELECT 'CREATE USER grafanareader LOGIN PASSWORD ''password'' '
WHERE NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'grafanareader')\gexec

GRANT USAGE ON SCHEMA public TO grafanareader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafanareader;
\q
EOF
