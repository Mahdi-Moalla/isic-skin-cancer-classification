"""
backend service
"""

# pylint: disable=import-error
import io
import json
import logging
import os

import avro.schema
import PIL
from avro.datafile import DataFileWriter
from avro.io import DatumWriter
from flask import Flask, jsonify, request
from kafka import KafkaProducer
from PIL import Image

logging.basicConfig(level=logging.INFO, format="BACKEND: %(asctime)s [%(levelname)s]: %(message)s")

kafka_server = os.getenv("kafka_server", "kafka:9092")

isic_schema_file = os.getenv('isic_schema_file', 'isic_record.avsc')

with open(isic_schema_file, 'r', encoding="utf-8") as f:
    schema = avro.schema.parse(f.read())

kafka_topic_name = os.getenv('kafka_topic_name', 'isic_topic')

producer = KafkaProducer(bootstrap_servers=[kafka_server], acks='all')

app = Flask(__name__)


@app.route('/v1/backend/upload_record', methods=['POST'])
def send_data():
    """
    receive  data from http request  and send it to kafka
    """
    if 'json_record' not in request.files.keys() or 'image' not in request.files.keys():
        return (
            jsonify({"error": "data  incomplete, please  post the json_record and image"}),
            404,
        )

    json_record_str = io.BytesIO()
    request.files['json_record'].save(json_record_str)
    try:
        json_record = json.loads(json_record_str.getvalue())
    except json.JSONDecodeError:
        return jsonify({"error": "json_record has invalid json data"}), 404

    image_buffer = io.BytesIO()
    request.files['image'].save(image_buffer)

    try:
        Image.open(image_buffer)
    except PIL.UnidentifiedImageError:
        return jsonify({"error": "image has invalid image data"}), 404

    avro_record = {
        'image': image_buffer.getvalue(),
        'json_record': json.dumps(json_record).replace('NaN', 'null'),
    }

    output_buffer = io.BytesIO()

    writer = DataFileWriter(output_buffer, DatumWriter(), schema)

    writer.append(avro_record)
    writer.flush()
    output_bytes = output_buffer.getvalue()
    writer.close()

    producer.send(kafka_topic_name, value=output_bytes)

    return jsonify({"message": "data received"})


@app.route('/v1/backend/test', methods=['GET'])
def display_msg():
    """
    test web service
    """
    return "welcome to isic backend"


# @app.route('/v1/backend/redirect-example', methods=['GET'])
# def display_msg():
#    return redirect()
