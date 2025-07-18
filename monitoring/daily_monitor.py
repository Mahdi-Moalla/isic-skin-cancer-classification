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

import mlflow

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import  (ColumnDriftMetric, 
                                DatasetDriftMetric, 
                                DatasetMissingValuesMetric,
                                ColumnQuantileMetric,
                                DatasetSummaryMetric,
                                ColumnDistributionMetric,
                                DatasetCorrelationsMetric,
                                ColumnSummaryMetric,
                                ColumnCorrelationsMetric)


# https://github.com/evidentlyai/evidently/tree/e9c784058e0b7e31a3e03e8849e79dc2e4918092

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
    
    monitoring_config.mlflow_server_url=\
        os.getenv('mlflow_server_url',
                  'http://legion-pro-7-16arx8h:5000/')
    monitoring_config.mlflow_experiment_name=\
        os.getenv('mlflow_experiment_name',
                  'isic-skin-cancer-classification')
    monitoring_config.monitoring_reference_data=\
        os.getenv('monitoring_reference_data',
                  'mlflow-artifacts:/1/0acf699c52604669a8ea2b3682a75e78/artifacts/monitoring_reference_data/monitoring_reference_data.parquet')
    
    monitoring_config.model_config_uri=\
        os.getenv('model_config_uri',
                  'mlflow-artifacts:/1/e00b740317ca4dfd96de57b75e850b2c/artifacts/trainer/config.py')

    return monitoring_config



def  main():

    np.random.seed(1)

    monitoring_config=init_config()
    pprint(monitoring_config.to_dict())

    
    mlflow.set_tracking_uri(uri=
                monitoring_config.mlflow_server_url)
    mlflow.set_experiment(
        monitoring_config.mlflow_experiment_name)
    
    monitoring_reference_data=mlflow.artifacts.download_artifacts(\
        artifact_uri=monitoring_config.monitoring_reference_data,
        dst_path ='.')
    
    ################################################
    #mlflow.artifacts.download_artifacts(artifact_uri=
    #    monitoring_config.model_config_uri,
    #    dst_path ='.')
    
    reference_data_df=pd.read_parquet(monitoring_reference_data)

    pos_ref_data=reference_data_df[ reference_data_df['target']==1 ]
    neg_ref_data=reference_data_df[ reference_data_df['target']==0 ]
    
    npos=len(pos_ref_data)

    new_reference_data_df=pd.concat([pos_ref_data,
                                     neg_ref_data.sample(npos*20,random_state=1)])
    new_reference_data_df=new_reference_data_df.reset_index(drop=True)

    #from IPython import embed as idbg; idbg(colors='Linux')


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

    current_data_df=pd.DataFrame(data_pd)

    from config import tab_features

    img_feats=[]

    for j, color in enumerate(['r','g','b']):
        img_feats.append(f"img_{color}_mean")
        img_feats.append(f"img_{color}_std")
        for k in  range(len(hist)):
            img_feats.append(f"img_{color}_hist_{bins[k]}_{bins[k+1]}")

    img_feats = img_feats
    

    column_mapping = ColumnMapping(
        target='target',
        prediction=None,#'score',
        numerical_features=img_feats + tab_features,
        categorical_features=None
    )


    new_reference_data_df.drop('mel_mitotic_index', axis=1,  inplace=True)
    current_data_df.drop('mel_mitotic_index', axis=1,  inplace=True)

    report = Report(metrics=
        [ ColumnSummaryMetric(c) for c in img_feats+tab_features]
    )
    report.run(reference_data=new_reference_data_df, 
               current_data=current_data_df, 
               column_mapping=column_mapping)

    report_res = Dict(report.as_dict())

    #pprint(resport_res)

    #pprint([
    #    (metric.result.column_name,
    #     metric.result.drift_score) for metric in resport_res.metrics
    #])
    for metric in report_res.metrics:
        output=f"""
        #####################################################
        column  name: {metric.result.column_name}
        ref mean: {metric.result.reference_characteristics.mean}
        ref std: {metric.result.reference_characteristics.std}
        curr mean: {metric.result.current_characteristics.mean}
        curr std: {metric.result.current_characteristics.std}
        """
        print(output)


    report = Report(metrics=
        [ ColumnDriftMetric(c) for c in img_feats+tab_features]
    )
    report.run(reference_data=new_reference_data_df, 
               current_data=current_data_df, 
               column_mapping=column_mapping)

    report_res = Dict(report.as_dict())

    #pprint(report_res)

    for metric in report_res.metrics:
        output=f"""
        #####################################################
        column  name: {metric.result.column_name}
        drift score: {metric.result.drift_score}
        """
        print(output)



    from IPython import embed as idbg; idbg(colors='Linux')




if __name__=='__main__':
    main()
