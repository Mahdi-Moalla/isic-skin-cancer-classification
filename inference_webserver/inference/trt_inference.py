"""
inference component
load data  from kafka + tensorrt inference
"""

# pylint:  disable=import-error

# import importlib
import io
import json
import logging
import os
import sys
import time

import common
import fire
import numpy as np
import pandas as pd
import scipy
import tensorrt as trt
import torch
from addict import Dict
from avro.datafile import DataFileReader
from avro.io import DatumReader
from kafka import KafkaConsumer
from PIL import Image

sys.path.insert(0, '../db_interface')
import db_interface  # pylint: disable=wrong-import-position

sys.path.append('../../')
from utils.python_utils.data_pipeline_util import (  # pylint: disable=wrong-import-position
    create_pipeline,
)

logging.basicConfig(
    level=logging.INFO, format="INFERENCE: %(asctime)s [%(levelname)s]: %(message)s"
)

# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine


TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# db_interface = importlib.import_module("db_interface")
db_connector = db_interface.db_connector  # pylint: disable=invalid-name
db_isic_inference = db_interface.db_isic_inference  # pylint: disable=invalid-name

with open("trainer_config.json", "r", encoding="utf-8") as f:
    config = Dict(json.load(f))


tab_features = config.tab_features

preprocess_transform = create_pipeline("preprocess_transform.json")
val_transforms = create_pipeline("val_transform.json")


def build_engine_onnx(model_file):
    """
    build engine from  onnx file
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    builder_config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    engine_bytes = builder.build_serialized_network(network, builder_config)
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)


# pylint: disable=too-many-locals
def init_config():
    """
    init config
    """
    data_persistance_config = Dict()

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

    data_persistance_config.kafka.kafka_server = os.getenv("kafka_server", "kafka:9092")

    data_persistance_config.kafka.kafka_topic_name = os.getenv('kafka_topic_name', 'isic_topic')

    return data_persistance_config


def main(onnx_model_path='./model.onnx'):
    """
    main kafka consumer + inference
    """
    data_persistance_config = init_config()

    logging.info(data_persistance_config)

    db_isic_inference_interface = db_isic_inference(
        db_connector(data_persistance_config.db.psycopg_conn_str)
    )

    db_isic_inference_interface.create_table()

    consumer = KafkaConsumer(
        data_persistance_config.kafka.kafka_topic_name,
        bootstrap_servers=[data_persistance_config.kafka.kafka_server],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='inference',
    )

    engine = build_engine_onnx(onnx_model_path)

    inputs, outputs, bindings, stream = common.allocate_buffers(engine)

    context = engine.create_execution_context()

    while True:

        for message in consumer:
            message = message.value
            reader = DataFileReader(io.BytesIO(message), DatumReader())

            record = next(iter(reader))

            pil_image = Image.open(io.BytesIO(record['image']))

            json_record = json.loads(record['json_record'])

            record_series = pd.Series(json_record)

            tab_feats = record_series[tab_features].to_numpy().astype(np.float32)

            image_np = np.array(pil_image)

            preprocessed_image_np = preprocess_transform(image=image_np)['image'].transpose(2, 0, 1)

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
            # We use the highest probability as our
            # prediction. Its index corresponds
            # to the predicted label.
            pred = trt_outputs[0]

            # pylint: disable=logging-fstring-interpolation
            if 'target' in json_record:
                logging.info(f'target: {json_record['target']}')
            logging.info(f'logits: {str(pred)}')
            logging.info(f'type: {type(pred)}')

            score = scipy.special.softmax(pred)[1]

            isic_id = int(json_record['isic_id'].split('_')[1])

            db_isic_inference_interface.insert_into_db(isic_id, score)

            logging.info(f"{json_record['isic_id']} record saved")

            time.sleep(1)


if __name__ == '__main__':

    fire.Fire(main)
