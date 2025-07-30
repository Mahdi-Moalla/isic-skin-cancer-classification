"""
integration test python code
"""

import json
import os
import pickle
import sys
import time

import requests
from deepdiff import DeepDiff

if __name__ == '__main__':

    webserver_uri = os.getenv("webserver_uri")

    with open('example.pkl', 'rb') as f:
        data = pickle.load(f)

    requests.post(f"http://{webserver_uri}/v1/backend/upload_record", files=data, timeout=1)

    time.sleep(1)

    response = requests.get(
        f"http://{webserver_uri}/v1/data-persistance/query", params={"isic_id": 1789180}, timeout=1
    )

    print(response.status_code)
    if response.status_code != 200:
        sys.exit(1)

    reference = json.loads(data['json_record'][1])

    received = response.json()['result']['record']

    result = DeepDiff(reference, received)

    if len(result.keys()) > 0:
        sys.exit(1)
    else:
        sys.exit(0)
