import os
from addict import Dict
from datetime import datetime

import fire
from pprint import pprint

import requests

import io
import numpy as np
from PIL import Image, ImageStat

import pandas as pd


from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import  (ColumnDriftMetric, 
                                DatasetDriftMetric, 
                                DatasetMissingValuesMetric,
                                ColumnQuantileMetric,
                                ColumnDistributionMetric,
                                DatasetCorrelationsMetric,
                                ColumnSummaryMetric,
                                ColumnCorrelationsMetric)


# https://github.com/evidentlyai/evidently/tree/e9c784058e0b7e31a3e03e8849e79dc2e4918092

HIST_BINS=50

def init_config():

    monitoring_config=Dict()

    monitoring_config.postgres_db.server=\
        os.getenv("postgres_server",
                  "legion-pro-7-16arx8h")
    monitoring_config.postgres_db.port=\
        os.getenv("postgres_port",5432)
    
    monitoring_config.postgres_db.db_name=\
        os.getenv("postgres_db_name",
                  "monitoring_db")
    monitoring_config.postgres_db.user_name=\
        os.getenv("postgres_db_user",
                  "monitoring_user")
    monitoring_config.postgres_db.password=\
        os.getenv("postgres_db_password",
                  "monitoring_pass")
    
    psycopg_conn_str=f"""
    host={monitoring_config.postgres_db.server} 
    port={monitoring_config.postgres_db.port} 
    dbname={monitoring_config.postgres_db.db_name} 
    user={monitoring_config.postgres_db.user_name} 
    password={monitoring_config.postgres_db.password}"""

    
    

    monitoring_config.postgres_db.psycopg_conn_str=\
        psycopg_conn_str


    monitoring_config.inference_webserver_host=\
        os.getenv("inference_webserver_host",
                  "http://legion-pro-7-16arx8h:8080")
    

    monitoring_config.date=os.getenv("date",
                                     "2025-01-01")
    
    #monitoring_config.hist_bins=\
    #    int(os.getenv("hist_bins",25))
    
    monitoring_config.monitoring_img_hist_bins=\
        int(os.getenv("monitoring_img_hist_bins",
                        50))

    return monitoring_config



def  main():
    monitoring_config=init_config()
    pprint(monitoring_config.to_dict())

    date=datetime.strptime(monitoring_config.date,
                           "%Y-%m-%d")

    print(date)

    response=requests.get(
        f"{monitoring_config.inference_webserver_host}/v1/data-persistance/dayquery",
        params={
            "day":date.strftime("%Y%m%d"),
            "score":"true"
        })
    
    pprint(response.json())

    data=Dict(response.json())

    bins=np.arange(0,256,
        monitoring_config.monitoring_img_hist_bins)
    bins[-1]=255
    
    data_pd=[]

    for i in range(len(data.result)):
        data.result[i].record.score=\
            data.result[i].score
        data.result[i].record.created_at=\
            data.result[i].created_at
        
        image_response=requests.get(
            f"{monitoring_config.inference_webserver_host}/v1/data-persistance/getimage",
            params={
                "isic_id":data.result[i].isic_id
            }
        )

        img=Image.open( io.BytesIO(image_response.content))
        
        img_np=np.array(img)

        band_count=img_np.size//3

        stats = ImageStat.Stat(img)

        for j, color in enumerate(['r','g','b']):
            data.result[i].record[f"img_{color}_mean"]=stats.mean[j]
            data.result[i].record[f"img_{color}_std"]=stats.stddev[j]

            hist=np.histogram(img_np[...,j], bins)[0]/band_count
            for k in  range(len(hist)):
                data.result[i].record[f"img_{color}_hist_{bins[k]}_{bins[k+1]}"]=hist[k].item()


        
        data_pd.append(data.result[i].record.to_dict())

    data_df=pd.DataFrame(data_pd)

    

    from IPython import embed as idbg; idbg(colors='Linux')




if __name__=='__main__':
    main()
