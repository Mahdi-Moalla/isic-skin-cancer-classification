#!/usr/bin/env bash
cd "$(dirname "$0")"

docker run -it -v $(pwd):/home\
 --network inference_webserver_default\
 mini_webserver bash