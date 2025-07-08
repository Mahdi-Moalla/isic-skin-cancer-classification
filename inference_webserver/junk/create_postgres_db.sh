#!/usr/bin/env bash


postgres_user=postgres
postgres_password=postgres

postgres_container=$(docker container ls | grep postgres | cut -d ' ' -f 1)

echo "$3
$3
$postgres_password" | docker exec -it $postgres_container\
    createuser -U $postgres_user $2 -S -D -R -P

echo $postgres_password | docker exec -it $postgres_container\
    createdb -U $postgres_user $1  -O $2