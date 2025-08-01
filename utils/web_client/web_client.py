"""
web client to  send data to the inference web server
"""

# pylint: disable=import-error, logging-fstring-interpolation
import json
import logging
import os
from datetime import datetime, timedelta

import fire
import h5py
import numpy as np
import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def process_record(record, date):
    """
    this function sends data to inference web server
    """
    record = record.to_dict()
    isic_id = record['isic_id']
    record['timestamp'] = str(date)
    logging.info(isic_id)
    logging.info(f'target= {record['target']}')

    with h5py.File(os.getenv('image_file'), 'r') as f:
        image_bytes = f[isic_id][()]

    webserver_uri = os.getenv("webserver_uri", "http://localhost:8080")

    data = {
        'json_record': (
            'json_record',
            json.dumps(record).replace("NaN", "null"),
            'text/plain',
        ),
        'image': ('image', image_bytes, 'application/octet-stream'),
    }

    result = requests.post(webserver_uri + "/v1/backend/upload_record", files=data, timeout=1)
    print(result.text)


def upload_month_data():
    """
    uploada chosen 1-month data
    """
    metadata_df = pd.read_csv('test-metadata.csv', low_memory=False)

    pos_data = metadata_df[metadata_df['target'] == 1]
    neg_data = metadata_df[metadata_df['target'] == 0]

    curr_day = datetime(2025, 1, 1, 0, 0, 0)
    pos_i = 0
    neg_i = 0
    for i in range(31):

        process_record(pos_data.iloc[pos_i], curr_day)
        pos_i += 1
        if i % 2 == 0:
            process_record(pos_data.iloc[pos_i], curr_day)
            pos_i += 1

        n_neg = np.random.randint(low=15, high=20)

        for _ in range(n_neg):
            process_record(neg_data.iloc[neg_i], curr_day)
            neg_i += 1

        curr_day += timedelta(days=1)


def upload_random_data(count):
    """
    upload random data
    """
    metadata_df = pd.read_csv('test-metadata.csv', low_memory=False)

    curr_day = datetime(2025, 3, 1, 0, 0, 0)
    for _ in range(count):
        idx = np.random.randint(len(metadata_df))
        process_record(metadata_df.iloc[idx], curr_day)


if __name__ == '__main__':
    fire.Fire()
