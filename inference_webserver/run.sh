#!/usr/bin/env bash
cd "$(dirname "$0")"
# docker container prune -f && docker  compose up --force-recreate --remove-orphans
docker run -it -v $(pwd):/home\
 --network inference_webserver_default\
 -e metadata_file=test-metadata.csv\
 -e image_file=test-image.hdf5\
 -e webserver_uri=http://inference-webserver:8080\
  mini_webserver bash