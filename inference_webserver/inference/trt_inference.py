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

logging.basicConfig(level=logging.INFO, format="INFERENCE: %(asctime)s [%(levelname)s]: %(message)s")

import fire
import os

# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
import sys

import numpy as np

import tensorrt as trt
from PIL import Image

import common

import  time

import torch
import torch.nn.functional as F

from preprocess_data import preprocess_transform
from config import config,tab_features
from  transforms import val_transforms

import  pandas as pd

import scipy

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


import sys
sys.path.insert(0, '../db_interface')

from db_interface import (db_connector,
                          db_isic_inference)


def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    engine_bytes = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)



def init_config():

    data_persistance_config=Dict()

    data_persistance_config.db.postgres_server=os.getenv("postgres_server",
                              "postgres-db")
    data_persistance_config.db.postgres_port=os.getenv("postgres_port",
                            5432)
    
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

    
    data_persistance_config.kafka.kafka_server=os.getenv("kafka_server",
                        "kafka:9092")

    
    data_persistance_config.kafka.kafka_topic_name=os.getenv('kafka_topic_name',
                            'isic_topic')

    return data_persistance_config


def main(onnx_model_path='./model.onnx'):
    data_persistance_config = init_config()

    logging.info(data_persistance_config)

    db_isic_inference_interface=db_isic_inference(db_connector(data_persistance_config.db.psycopg_conn_str))    

    db_isic_inference_interface.create_table()


    
    consumer = KafkaConsumer(
        data_persistance_config.kafka.kafka_topic_name,
        bootstrap_servers=[data_persistance_config.kafka.kafka_server],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='inference')

    engine = build_engine_onnx(onnx_model_path)

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
    context = engine.create_execution_context()

    while(True):
        

        for message in consumer:
            message = message.value
            reader = DataFileReader(io.BytesIO(message), 
                                DatumReader())

            record=next(iter(reader))

            pil_image = Image.open( io.BytesIO(record['image']) )

            json_record=json.loads( record['json_record'] )


            record_series=  pd.Series( json_record )

            tab_feats=record_series[ tab_features ].to_numpy().astype(np.float32)

            image_np=np.array(pil_image)
    
            preprocessed_image_np=preprocess_transform(image=image_np)['image'].transpose(2,0,1)

            image_torch = torch.tensor(preprocessed_image_np)


            image_torch = val_transforms(image_torch)

    

            image_np = image_torch.numpy()
            np.copyto(inputs[0].host, image_np.ravel())
            np.copyto(inputs[1].host, tab_feats.ravel())

            trt_outputs = common.do_inference(
                context,
                engine=engine,
                bindings=bindings,
                inputs=inputs,
                outputs=outputs,
                stream=stream,
            )
            # We use the highest probability as our prediction. Its index corresponds to the predicted label.
            pred = trt_outputs[0]

            if 'target' in json_record:
                logging.info(f'target: {json_record['target']}')
            logging.info(f'logits: {str(pred)}')
            logging.info(f'type: {type(pred)}')

            score=scipy.special.softmax(pred)[1]

            isic_id=int(json_record['isic_id'].split('_')[1])

            db_isic_inference_interface.insert_into_db(isic_id,
                                                       score)
            

            logging.info(f"{json_record['isic_id']} record saved")

            time.sleep(1)

    
    


if __name__=='__main__':

    fire.Fire(main)
