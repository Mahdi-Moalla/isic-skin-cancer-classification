"""
data persistance service
answers queries on the received data
"""

import glob
import importlib
import logging
import os
import os.path as osp
import sys
from datetime import datetime

from addict import Dict
from flask import Flask, jsonify, request, send_file

logging.basicConfig(
    level=logging.INFO,
    format="DATA-PERSISTANCE-WEBSERVICE: %(asctime)s [%(levelname)s]: %(message)s",
)

sys.path.insert(0, '../db_interface')

db_interface = importlib.import_module('db_interface')
db_connector = db_interface.db_connector
db_isic_data = db_interface.db_isic_data


def init_config():
    """
    generate config from environment variables
    """
    data_persistance_config = Dict()  # pylint: disable=redefined-outer-name

    data_persistance_config.db.postgres_server = os.getenv("postgres_server", "postgres-db")
    data_persistance_config.db.postgres_port = os.getenv("postgres_port", "5432")

    data_persistance_config.db.postgres_db_name = os.getenv("postgres_db_name", "isic_db")
    data_persistance_config.db.postgres_db_user = os.getenv("postgres_db_user", "isic_db_user")
    data_persistance_config.db.postgres_db_password = os.getenv(
        "postgres_db_password", "isic_db_password"
    )

    psycopg_conn_str = f"""
    host={data_persistance_config.db.postgres_server}
    port={data_persistance_config.db.postgres_port}
    dbname={data_persistance_config.db.postgres_db_name}
    user={data_persistance_config.db.postgres_db_user}
    password={data_persistance_config.db.postgres_db_password}"""

    data_persistance_config.db.psycopg_conn_str = psycopg_conn_str

    data_persistance_config.images_folder = os.getenv("images_folder", "/home/images")

    return data_persistance_config


data_persistance_config = init_config()

logging.info(str(data_persistance_config))

app = Flask(__name__)

db_isic_data_interface = db_isic_data(db_connector(data_persistance_config.db.psycopg_conn_str))


@app.route('/v1/data-persistance/query', methods=['GET'])
def query_single_record():
    """
    web service to query a single record by isic_id
    """
    isic_id = request.args.get('isic_id')
    if isic_id is None:
        return jsonify({"error": "isic_id is not provided"})
    res = db_isic_data_interface.query_single_record(isic_id)
    return jsonify({"result": res})


@app.route('/v1/data-persistance/multiquery', methods=['GET'])
def query_multi_records():
    """
    web service to query multiple record based on multiple isic_id
    """
    isic_ids = request.args.getlist('isic_ids')
    if isic_ids is None:
        return jsonify({"error": "isic_id is not provided"})
    res = db_isic_data_interface.query_multi_records(isic_ids)
    return jsonify({"result": res})


@app.route('/v1/data-persistance/getimage', methods=['GET'])
def get_image():
    """
    web service download  a record image
    """
    isic_id = request.args.get('isic_id')
    if isic_id is None:
        return jsonify({"error": "isic_id is not provided"})
    res = db_isic_data_interface.query_single_record(isic_id)
    if res is None:
        return jsonify({"error": "record does not exist"})
    isic_id_str = Dict(res).record.isic_id
    print(data_persistance_config.images_folder)
    file_path = glob.glob(osp.join(data_persistance_config.images_folder, f'{isic_id_str}*'))
    if len(file_path) == 0:
        return jsonify({"error": "image not found"}), 500
    return send_file(file_path[0], as_attachment=True, download_name=file_path[0].split('/')[-1])


@app.route('/v1/data-persistance/dayquery', methods=['GET'])
def query_records_by_day():
    """
    web service to query records based on a date
    """
    day_str = request.args.get('day')
    get_score = request.args.get('score')
    if day_str is None:
        return jsonify({"error": "day is not provided"})

    try:
        day = datetime.strptime(str(day_str), "%Y%m%d")
    except ValueError:
        return jsonify({"error": "invalid date format"})
    if get_score is None:
        res = db_isic_data_interface.query_records_by_day(day)
    else:
        res = db_isic_data_interface.query_records_with_score_by_day(day)
    return jsonify({"result": res})


@app.route('/v1/data-persistance/test', methods=['GET'])
def display_msg():
    """
    test web service
    """
    return "welcome to isic data-persistance"
