
import logging

logging.basicConfig(level=logging.INFO, format="INFERENCE-WEBSERVICE: %(asctime)s [%(levelname)s]: %(message)s")


from flask import (Flask,
                   jsonify, 
                   flash, 
                   request, 
                   redirect, 
                   url_for)

from addict import Dict

import os


import sys
sys.path.insert(0, '../db_interface')

from db_interface import (db_connector,
                          db_isic_inference)

def init_config():

    inference_config=Dict()

    inference_config.db.postgres_server=os.getenv("postgres_server",
                              "postgres-db")
    inference_config.db.postgres_port=os.getenv("postgres_port",
                            5432)
    
    inference_config.db.postgres_db_name=os.getenv("postgres_db_name",
                               "isic_db")
    inference_config.db.postgres_db_user=os.getenv("postgres_db_user",
                               "isic_db_user")
    inference_config.db.postgres_db_password=os.getenv("postgres_db_password",
                                   "isic_db_password")
    
    psycopg_conn_str=f"""
    host={inference_config.db.postgres_server} 
    port={inference_config.db.postgres_port} 
    dbname={inference_config.db.postgres_db_name} 
    user={inference_config.db.postgres_db_user} 
    password={inference_config.db.postgres_db_password}"""

    
    

    inference_config.db.psycopg_conn_str=psycopg_conn_str

    return inference_config


inference_config = init_config()

logging.info(str(inference_config))

app = Flask(__name__)

db_isic_inference_interface= db_isic_inference(db_connector(inference_config.db.psycopg_conn_str))

@app.route('/v1/inference/query', methods=['GET'])
def query_single_record():
    isic_id = request.args.get('isic_id')
    if isic_id is None:
        return jsonify({"error":"isic_id is not provided"})
    res=db_isic_inference_interface.query_single_record(isic_id)
    return jsonify({
        "result": res
    })

@app.route('/v1/inference/multiquery', methods=['GET'])
def query_multi_records():
    isic_ids = request.args.getlist('isic_ids')
    if isic_ids is None:
        return jsonify({"error":"isic_id is not provided"})
    res=db_isic_inference_interface.query_multi_records(isic_ids)
    return jsonify({
        "result": res
    })



@app.route('/v1/inference/test', methods=['GET'])
def display_msg():
    return "welcome to isic inference"
