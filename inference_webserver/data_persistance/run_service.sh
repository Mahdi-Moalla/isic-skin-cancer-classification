#!/usr/bin/env bash
cd "$(dirname "$0")"
flask --app service run\
 --host=0.0.0.0 --port 8080 --debug