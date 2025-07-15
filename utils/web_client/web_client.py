import os
import json
import logging

import requests
import  pandas as pd
import  numpy as np
import h5py

from datetime import datetime, timedelta

from PIL import Image
import io

import avro.schema
from avro.datafile import DataFileWriter, DataFileReader
from avro.io import DatumWriter, DatumReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def process_record(record,
                   date):
    record=record.to_dict()
    isic_id=record['isic_id']
    record['timestamp']=str( date )
    logging.info(isic_id)
    logging.info(f'target= {record['target']}')

    with h5py.File(os.getenv('image_file'),'r') as f:
        image_bytes=f[isic_id][()]
    
    #pil_image=Image.open( io.BytesIO( image_bytes ))

    webserver_uri=os.getenv("webserver_uri",
                            "http://localhost:8080")
    
    data = {
        'json_record': ('json_record',
                        json.dumps(record).replace("NaN","null"),
                        'text/plain'),
        'image': ('image', 
                  image_bytes, 
                  'application/octet-stream')
    }
    
    result=requests.post(webserver_uri+"/v1/upload_record",
                        files=data)
    print(result.text)


def upload_month_data(metadata_df):
    pos_data=metadata_df[ metadata_df['target']==1  ]
    neg_data=metadata_df[ metadata_df['target']==0  ]

    curr_day=datetime(2025,1,1,0,0,0)
    pos_i=0
    neg_i=0
    for i in range(31):

        process_record(pos_data.iloc[pos_i],
                       curr_day)
        pos_i+=1
        if i%2==0:
            process_record(pos_data.iloc[pos_i],
                       curr_day)
            pos_i+=1

        n_neg=np.random.randint(low=15,high=20)

        for _ in range(n_neg):
            process_record(neg_data.iloc[neg_i],
                           curr_day)
            neg_i+=1
        
        curr_day+=timedelta(days=1)
        

def upload_random_data(metadata_df):

    idx=np.random.randint(len(metadata_df))
    curr_day=datetime(2025,3,1,0,0,0)
    process_record(metadata_df.iloc[idx],
                           curr_day)
            

if __name__=='__main__':

    metadata_df=pd.read_csv('test-metadata.csv',
                            low_memory=False)
    
    upload_random_data(metadata_df)