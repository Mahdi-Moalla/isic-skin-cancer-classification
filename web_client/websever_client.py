import os
import json
import logging

import requests
import  pandas as pd
import  numpy as np
import h5py

from datetime import datetime

from PIL import Image
import io

import avro.schema
from avro.datafile import DataFileWriter, DataFileReader
from avro.io import DatumWriter, DatumReader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def process_record(record):
    isic_id=record['isic_id']
    record['timestamp']=str( datetime(2025,1,1,0,0,0) )
    logging.info(isic_id)

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
    
    result=requests.post(webserver_uri,
                        files=data)
    print(result.text)

    # avro_record={
    #     'image':image_bytes,
    #     'json_record':json.dumps(record).replace("NaN","null")
    # }

    # with open('isic_record.avsc','r') as f:
    #     schema = avro.schema.parse(f.read())
    
    # output_buffer = io.BytesIO()

    # writer = DataFileWriter(output_buffer,
    #                         DatumWriter(), 
    #                         schema)

    # writer.append(avro_record)
    # writer.flush()
    # output_bytes=output_buffer.getvalue()
    # writer.close()



    # print(output_bytes)

    # reader = DataFileReader(io.BytesIO(output_bytes), 
    #                         DatumReader())

    # for record in reader:
    #     print(record.keys())
        
    #     pil_image = Image.open( io.BytesIO(record['image']) )

    #     json_record=json.loads( record['json_record'] )

    #     print(pil_image.size)
    #     print(json_record.keys())



if __name__=='__main__':

    metadata_df=pd.read_csv('test-metadata.csv',
                            low_memory=False)

    idx=np.random.randint(len(metadata_df))

    record=metadata_df.iloc[idx].to_dict()

    process_record(record)