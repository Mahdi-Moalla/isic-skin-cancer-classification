"""
daily monitoring script
"""

import importlib
# pylint: disable=import-error
import io
import json
import os
from datetime import datetime, timedelta
from pprint import pprint

# import fire
import mlflow
import numpy as np
import pandas as pd
import requests
from addict import Dict
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, ColumnSummaryMetric
from evidently.report import Report
from PIL import Image, ImageStat
from sklearn.metrics import auc, roc_curve
from sqlalchemy import create_engine, text
from tqdm import tqdm

# https://github.com/evidentlyai/evidently/tree/e9c784058e0b7e31a3e03e8849e79dc2e4918092


def init_config():
    """
    generate init config from environment variables
    """
    monitoring_config = Dict()

    monitoring_config.postgres_db.server = os.getenv(
        "postgres_server", "legion-pro-7-16arx8h"
    )
    monitoring_config.postgres_db.port = os.getenv(
        "postgres_port", "5432"
    )  # pylint: disable=invalid-envvar-default

    monitoring_config.postgres_db.db_name = os.getenv(
        "postgres_db_name", "monitoring_db"
    )
    monitoring_config.postgres_db.user_name = os.getenv(
        "postgres_db_user", "monitoring_user"
    )
    monitoring_config.postgres_db.password = os.getenv(
        "postgres_db_password", "monitoring_pass"
    )

    # pylint: disable=consider-using-f-string
    psycopg_conn_str = "postgresql+psycopg://{}:{}@{}:{}/{}".format(
        monitoring_config.postgres_db.user_name,
        monitoring_config.postgres_db.password,
        monitoring_config.postgres_db.server,
        monitoring_config.postgres_db.port,
        monitoring_config.postgres_db.db_name,
    )

    monitoring_config.postgres_db.psycopg_conn_str = psycopg_conn_str

    monitoring_config.inference_webserver_host = os.getenv(
        "inference_webserver_host", "http://legion-pro-7-16arx8h:8080"
    )

    # monitoring_config.date=os.getenv("date",
    #                                 "2025-01-01")

    # monitoring_config.hist_bins=\
    #    int(os.getenv("hist_bins",25))

    monitoring_config.monitoring_img_hist_bins = int(
        os.getenv(  # pylint: disable=invalid-envvar-default
            "monitoring_img_hist_bins", 5
        )
    )

    monitoring_config.mlflow_server_url = os.getenv(
        'mlflow_server_url', 'http://legion-pro-7-16arx8h:5000/'
    )
    monitoring_config.mlflow_experiment_name = os.getenv(
        'mlflow_experiment_name', 'isic-skin-cancer-classification'
    )

    monitoring_config.fold_run_id = os.getenv(
        'fold_run_id', 'f5303275ff1545aeb4d0ec11fe4d7cff'
    )

    return monitoring_config


# pylint: disable=too-many-locals, too-many-statements
def main(date, monitoring_config):
    """
    main function to collect metrics and save them
    to monitoring database
    """
    np.random.seed(1)

    pprint(monitoring_config.to_dict())

    mlflow.set_tracking_uri(uri=monitoring_config.mlflow_server_url)
    experiment = mlflow.set_experiment(monitoring_config.mlflow_experiment_name)

    experiment_id = experiment.experiment_id

    fold_run = mlflow.get_run(run_id=monitoring_config.fold_run_id)

    run_id = fold_run.data.tags['mlflow.parentRunId']

    mlflow.artifacts.download_artifacts(
        artifact_uri=f'mlflow-artifacts:/{experiment_id}/{run_id}/artifacts/trainer/config.py',
        dst_path='.',
    )

    tab_features = importlib.import_module("config").tab_features

    monitoring_reference_data = mlflow.artifacts.download_artifacts(
        artifact_uri=f'mlflow-artifacts:/{experiment_id}/{monitoring_config.fold_run_id}'
        + '/artifacts/monitoring_reference_data/monitoring_reference_data.parquet',
        dst_path='.',
    )

    monitoring_reference_cumul_hist = mlflow.artifacts.download_artifacts(
        artifact_uri=f'mlflow-artifacts:/{experiment_id}/{monitoring_config.fold_run_id}'
        + ''
        '/artifacts/monitoring_reference_cumul_hist/monitoring_reference_cumul_hist.parquet',
        dst_path='.',
    )

    reference_data_df = pd.read_parquet(monitoring_reference_data)
    reference_cumul_hist_df = pd.read_parquet(monitoring_reference_cumul_hist)

    pos_ref_data = reference_data_df[reference_data_df['target'] == 1]
    neg_ref_data = reference_data_df[reference_data_df['target'] == 0]

    npos = len(pos_ref_data)

    new_reference_data_df = pd.concat(
        [pos_ref_data, neg_ref_data.sample(npos * 20, random_state=1)]
    )
    new_reference_data_df = new_reference_data_df.reset_index(drop=True)

    # from IPython import embed as idbg; idbg(colors='Linux')

    response = requests.get(
        f"{monitoring_config.inference_webserver_host}/v1/data-persistance/dayquery",
        params={"day": date.strftime("%Y%m%d"), "score": "true"},
        timeout=1,
    )

    # pprint(response.json())

    data = Dict(response.json())

    bins = np.arange(0, 256, monitoring_config.monitoring_img_hist_bins)
    bins[-1] = 255
    bins_labels = [str(x / 2) for x in (bins[:-1] + bins[1:]).tolist()]

    data_pd = []

    curr_hists_data = []

    scores = []
    targets = []

    alarms = []

    # pylint: disable=consider-using-enumerate
    for i in range(len(data.result)):
        record = data.result[i].record

        record.score = data.result[i].score
        record.created_at = data.result[i].created_at

        scores.append(record.score)
        targets.append(record.target)

        if targets[-1] == 1 and scores[-1] < 0.5:
            alarms.append(
                {"date": date, "isic_id": record.isic_id, "score": record.score}
            )

        image_response = requests.get(
            f"{monitoring_config.inference_webserver_host}/v1/data-persistance/getimage",
            params={"isic_id": data.result[i].isic_id},
            timeout=1,
        )

        img = Image.open(io.BytesIO(image_response.content))

        img_np = np.array(img)

        band_count = img_np.size // 3

        stats = ImageStat.Stat(img)

        for j, color in enumerate(['r', 'g', 'b']):
            record[f"img_{color}_mean"] = stats.mean[j]
            record[f"img_{color}_std"] = stats.stddev[j]
            record["band_count"] = band_count

            hist = np.histogram(img_np[..., j], bins)[0] / band_count
            for k in range(len(hist)):
                curr_hists_data.append(
                    {
                        "date": date,
                        "isic_id": data.result[i].record.isic_id,
                        "color": color,
                        "bin_label": bins_labels[k],
                        "value": hist[k],
                    }
                )

        data_pd.append(record.to_dict())

    img_feats = []

    for j, color in enumerate(['r', 'g', 'b']):
        img_feats.append(f"img_{color}_mean")
        img_feats.append(f"img_{color}_std")

    current_data_df = pd.DataFrame(data_pd)
    curr_hists_data_df = pd.DataFrame(curr_hists_data)

    fpr, tpr, _ = roc_curve(targets, scores)

    roc = []

    for fx, ty in zip(fpr.tolist(), tpr.tolist()):
        roc.append({"date": date, "fx": fx, "ty": ty})

    rocs_df = pd.DataFrame(roc)
    aucs_df = pd.DataFrame([{"date": date, "auc": auc(fpr, tpr)}])

    alarms_df = pd.DataFrame(alarms)

    # from IPython import embed as idbg; idbg(colors='Linux')

    column_mapping = ColumnMapping(
        target='target',
        prediction=None,  #'score',
        numerical_features=img_feats + tab_features,
        categorical_features=None,
    )

    new_reference_data_df.drop('mel_mitotic_index', axis=1, inplace=True)
    current_data_df.drop('mel_mitotic_index', axis=1, inplace=True)

    report = Report(metrics=[ColumnSummaryMetric(c) for c in img_feats + tab_features])
    report.run(
        reference_data=new_reference_data_df,
        current_data=current_data_df,
        column_mapping=column_mapping,
    )

    report_res = Dict(report.as_dict())

    curr_means = Dict({})
    ref_means = Dict({})
    curr_stds = Dict({})
    ref_stds = Dict({})

    for metric in report_res.metrics:
        # output = f"""
        # #####################################################
        # column  name: {metric.result.column_name}
        # ref mean: {metric.result.reference_characteristics.mean}
        # ref std: {metric.result.reference_characteristics.std}
        # curr mean: {metric.result.current_characteristics.mean}
        # curr std: {metric.result.current_characteristics.std}
        # """
        # print(output)
        curr_means[metric.result.column_name] = (
            metric.result.current_characteristics.mean
        )
        ref_means[metric.result.column_name] = (
            metric.result.reference_characteristics.mean
        )
        curr_stds[metric.result.column_name] = metric.result.current_characteristics.std
        ref_stds[metric.result.column_name] = (
            metric.result.reference_characteristics.std
        )

    report = Report(metrics=[ColumnDriftMetric(c) for c in img_feats + tab_features])
    report.run(
        reference_data=new_reference_data_df,
        current_data=current_data_df,
        column_mapping=column_mapping,
    )

    report_res = Dict(report.as_dict())

    # pprint(report_res)

    drift = Dict({})

    for metric in report_res.metrics:
        # output = f"""
        # #####################################################
        # column  name: {metric.result.column_name}
        # drift score: {metric.result.drift_score}
        # """
        # print(output)
        drift[metric.result.column_name] = metric.result.drift_score

    ref_means = pd.DataFrame([ref_means.to_dict()])
    ref_stds = pd.DataFrame([ref_stds.to_dict()])

    # reference_histograms
    # current_histograms['date']=date.date()

    drift = pd.DataFrame([drift.to_dict()])
    curr_means = pd.DataFrame([curr_means.to_dict()])
    curr_stds = pd.DataFrame([curr_stds.to_dict()])

    drift['date'] = date.date()
    curr_means['date'] = date.date()
    curr_stds['date'] = date.date()

    # from IPython import embed as idbg; idbg(colors='Linux')
    # exit(0)

    engine = create_engine(monitoring_config.postgres_db.psycopg_conn_str)

    try:
        reference_cumul_hist_df.to_sql(
            name="reference_hist", con=engine, if_exists='fail'
        )
        ref_means.to_sql(name="ref_means", con=engine, if_exists='fail')
        ref_stds.to_sql(name="ref_stds", con=engine, if_exists='fail')
    except ValueError:
        pass

    curr_hists_data_df.to_sql(name="curr_hists", con=engine, if_exists='append')

    curr_means.to_sql(name='curr_means', con=engine, if_exists='append')
    curr_stds.to_sql(name='curr_stds', con=engine, if_exists='append')
    drift.to_sql(name='drift', con=engine, if_exists='append')
    rocs_df.to_sql(name='rocs', con=engine, if_exists='append')
    aucs_df.to_sql(name='aucs', con=engine, if_exists='append')
    if len(alarms_df) > 0:
        alarms_df.to_sql(name='alarms', con=engine, if_exists='append')

    with engine.connect() as connection:
        grafana_allow_sql_select = """
        GRANT USAGE ON SCHEMA public TO grafanareader;
        GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafanareader;
        """
        connection.execute(text(grafana_allow_sql_select))

    engine.dispose()


def runner():
    """
    starter function
    """
    monitoring_config = init_config()

    dag_params = Dict(json.loads(os.getenv("dag_params").replace("\'", "\"")))

    monitoring_config.fold_run_id = dag_params.fold_run_id

    start_date = datetime.strptime(dag_params.start_date, "%Y-%m-%d")

    # print(monitoring_config)
    # print(dag_params)
    # print(start_date)

    if dag_params.run_type == 'single':
        main(start_date, monitoring_config)
    elif dag_params.run_type == 'range':
        end_date = datetime.strptime(dag_params.end_date, "%Y-%m-%d")

        dates = []
        while start_date <= end_date:
            dates.append(start_date)
            start_date += timedelta(days=1)

        for date in tqdm(dates):
            print('########################################################')
            print(date)
            main(date, monitoring_config)


if __name__ == '__main__':
    runner()
