#psql  -U postgres -d monitoring_db
ALTER SCHEMA "public" RENAME TO "monitoring_db"

CREATE USER grafanareader WITH PASSWORD 'password';
GRANT USAGE ON SCHEMA public TO grafanareader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafanareader;