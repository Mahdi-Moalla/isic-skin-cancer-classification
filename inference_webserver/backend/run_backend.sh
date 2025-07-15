#!/usr/bin/env bash
cd "$(dirname "$0")"
python topic_creator.py
flask --app backend run\
 --host=0.0.0.0 --port 8080 --debug
