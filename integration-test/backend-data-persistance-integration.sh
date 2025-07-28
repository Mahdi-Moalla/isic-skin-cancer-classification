#!/usr/bin/env bash
cd "$(dirname "$0")"

export parent_dir=$( dirname $(pwd) )
echo $parent_dir

docker pull traefik:v3.5

docker build -t integration_test_image:latest  .

docker compose -f \
    backend-data-persistance-docker-compose.yml \
    up --force-recreate --remove-orphans -d --wait # --no-attach kafka  

sleep 1
# curl http://localhost:8080/v1/data-persistance/test
# echo ""
# curl http://localhost:8080/v1/backend/test
# echo ""

docker run --rm -v $(pwd):/home\
 --network integration-public-net\
 -e webserver_uri=traefik:8080 \
 -w /home\
  integration_test_image:latest python integration_test.py

error_code=$?

# docker compose -f \
#     backend-data-persistance-docker-compose.yml \
#     logs --no-color > logs.txt

docker compose -f \
    backend-data-persistance-docker-compose.yml \
    down

exit $error_code