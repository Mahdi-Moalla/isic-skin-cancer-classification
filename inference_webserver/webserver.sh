#!/usr/bin/env bash
cd "$(dirname "$0")"
python init_db.py
python topic_creator.py
python data_persistance.py &
flask --app webserver run\
 --host=0.0.0.0 --port $1 --debug
