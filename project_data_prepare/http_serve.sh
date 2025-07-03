#!/usr/bin/env bash
cd "$(dirname "$0")"
python -m http.server $1 --directory $2