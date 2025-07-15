import os
import os.path as osp
import shutil
import json
import io
import datetime
import  time

import pandas as pd
import psycopg
import logging

from addict import Dict

import fire

import kafka
from kafka import KafkaConsumer

import avro.schema
from avro.datafile import DataFileWriter, DataFileReader
from avro.io import DatumWriter, DatumReader

from PIL import  Image

logging.basicConfig(level=logging.INFO, format="DATA-PERSISTANCE: %(asctime)s [%(levelname)s]: %(message)s")

import sys
from pathlib import Path

import sys
sys.path.insert(0, '../db_interface/')

from db_interface import (db_connector,
                          db_isic_data)


def init_config():

    data_persistance_config=Dict()

    data_persistance_config.db.postgres_server=os.getenv("postgres_server",
                              "postgres-db")
    data_persistance_config.db.postgres_port=os.getenv("postgres_port",
                            5432)
    #postgres_password=os.getenv("postgres_password",
    #                            "postgres")
    
    data_persistance_config.db.postgres_db_name=os.getenv("postgres_db_name",
                               "isic_db")
    data_persistance_config.db.postgres_db_user=os.getenv("postgres_db_user",
                               "isic_db_user")
    data_persistance_config.db.postgres_db_password=os.getenv("postgres_db_password",
                                   "isic_db_password")
    
    psycopg_conn_str=f"""
    host={data_persistance_config.db.postgres_server} 
    port={data_persistance_config.db.postgres_port} 
    dbname={data_persistance_config.db.postgres_db_name} 
    user={data_persistance_config.db.postgres_db_user} 
    password={data_persistance_config.db.postgres_db_password}"""

    
    

    data_persistance_config.db.psycopg_conn_str=psycopg_conn_str

    data_persistance_config.images_folder=os.getenv("images_folder",
                            "images_folder")
    #shutil.rmtree(webserver_config.uploads_folder,  ignore_errors=True)
    #os.makedirs(webserver_config.uploads_folder, exist_ok=True)

    data_persistance_config.kafka.kafka_server=os.getenv("kafka_server",
                        "kafka:9092")

    #data_persistance_config.kafka.isic_schema_file=os.getenv('isic_schema_file',
    #                        'isic_record.avsc')

    #with open(isic_schema_file,'r') as f:
    #    schema = avro.schema.parse(f.read())

    data_persistance_config.kafka.kafka_topic_name=os.getenv('kafka_topic_name',
                            'isic_topic')

    return data_persistance_config


def main():
    data_persistance_config = init_config()

    logging.info(data_persistance_config)

    db_isic_data_interface=db_isic_data(db_connector(data_persistance_config.db.psycopg_conn_str))

    db_isic_data_interface.create_table()

    shutil.rmtree(data_persistance_config.images_folder,  ignore_errors=True)
    os.makedirs(data_persistance_config.images_folder, exist_ok=True)

    #with open(data_persistance_config.kafka.isic_schema_file,'r') as f:
    #   schema = avro.schema.parse(f.read())

    consumer = KafkaConsumer(
        data_persistance_config.kafka.kafka_topic_name,
        bootstrap_servers=[data_persistance_config.kafka.kafka_server],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='data_persistance')


    while(True):
        

        for message in consumer:
            message = message.value
            reader = DataFileReader(io.BytesIO(message), 
                                DatumReader())

            record=next(iter(reader))

            pil_image = Image.open( io.BytesIO(record['image']) )

            json_record=json.loads( record['json_record'] )

            db_isic_data_interface.insert_into_db(json_record)

            image_output_path=osp.join(data_persistance_config.images_folder,
                            f"{json_record['isic_id']}.png")
            #logging.info(image_output_path)
            pil_image.save(image_output_path)
            
            logging.info(f"{json_record['isic_id']} record saved")

            time.sleep(1)



if __name__=='__main__':

    fire.Fire(main)
