#!/usr/bin/env bash
cd "$(dirname "$0")"
# docker container prune -f && docker  compose up --force-recreate --remove-orphans
docker pull busybox:glibc
docker run --rm -it -v $(pwd):/home\
 --network host\
 -w /home\
 busybox:glibc wget -nc \
  http://localhost:9000/test-image.hdf5\
  http://localhost:9000/test-metadata.csv

docker run --rm -it -v $(pwd):/home\
 --network host\
 -e metadata_file=test-metadata.csv\
 -e image_file=test-image.hdf5\
 -e webserver_uri=http://localhost:8080\
 -w /home\
  webserver:latest python web_client.py upload_month_data