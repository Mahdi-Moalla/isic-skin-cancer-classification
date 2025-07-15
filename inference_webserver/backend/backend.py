import os
import json
import io
import datetime
import logging

from PIL import  Image

logging.basicConfig(level=logging.INFO, format="BACKEND: %(asctime)s [%(levelname)s]: %(message)s")

from flask import (Flask,
                   jsonify, 
                   flash, 
                   request, 
                   redirect, 
                   url_for)

import kafka
from kafka import KafkaProducer

import avro.schema
from avro.datafile import DataFileWriter, DataFileReader
from avro.io import DatumWriter, DatumReader


kafka_server=os.getenv("kafka_server",
                        "kafka:9092")

isic_schema_file=os.getenv('isic_schema_file',
                            'isic_record.avsc')

with open(isic_schema_file,'r') as f:
    schema = avro.schema.parse(f.read())

kafka_topic_name=os.getenv('kafka_topic_name',
                            'isic_topic')

producer = KafkaProducer(bootstrap_servers=[kafka_server],
                        acks='all')

app = Flask(__name__)

@app.route('/v1/backend/upload_record', methods=['POST'])
def send_data():
    
    if 'json_record' not in request.files.keys() or\
            'image'  not  in request.files.keys():
        return jsonify({"error":
                "data  incomplete, please  post the json_record and image"}),  404
        
    json_record_str=io.BytesIO()
    request.files['json_record'].save(json_record_str)
    try:
        json_record=json.loads(json_record_str.getvalue())
    except:
        return jsonify({"error":
                "json_record has invalid json data"}), 404
        
    image_buffer=io.BytesIO()
    request.files['image'].save(image_buffer)

    try:
        pil_image=Image.open( image_buffer )
    except:
        return jsonify({"error":
                "image has invalid image data"}), 404
        
    avro_record={
            'image': image_buffer.getvalue(),
            'json_record': json.dumps(json_record).replace('NaN','null')
    }
        
    output_buffer = io.BytesIO()

    writer = DataFileWriter(output_buffer,
                                DatumWriter(), 
                                schema)

    writer.append(avro_record)
    writer.flush()
    output_bytes=output_buffer.getvalue()
    writer.close()

    producer.send(kafka_topic_name,
                        value=output_bytes)

    return jsonify({"message":"data received"})


@app.route('/v1/backend/test', methods=['GET'])
def display_msg():
    return "welcome to isic backend"

#@app.route('/v1/backend/redirect-example', methods=['GET'])
#def display_msg():
#    return redirect()

