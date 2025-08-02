"""
data preprocessing task
"""

import io
import json
import os
import sys
import os.path as osp
import shutil
from pathlib import Path

#import albumentations as A
#import cv2

import fire
import h5py
import mlflow
import numpy as np
import pandas as pd
from PIL import Image, ImageStat
from tqdm import tqdm


sys.path.append('../../')
from utils.python_utils.data_pipeline_util import create_pipeline

# preprocess_transform = A.Compose(
#     [
#         A.Resize(
#             height=256,
#             width=256,
#             interpolation=cv2.INTER_CUBIC,  # pylint: disable=no-member
#             p=1.0,
#         ),
#         A.CenterCrop(height=224, width=224),
#     ]
# )

# def log_train_data(metadata_file, image_file):  # pylint: disable=too-many-locals
#     """
#     log training data to be used for comparison
#     in the monitoring component
#     """
#     print('####### in  log_train_data ##########')
#     hist_bins_size = int(os.getenv("monitoring_img_hist_bins_size", "50"))

#     bins = np.arange(0, 256, hist_bins_size)
#     bins[-1] = 255

#     bins_labels = [str(x / 2) for x in (bins[:-1] + bins[1:]).tolist()]

#     metadata = pd.read_csv(metadata_file, low_memory=False)
#     isic_id_lut = {idx: k for k, idx in enumerate(metadata['isic_id'])}

#     isic_ids=metadata['isic_id'].tolist()

#     data = []

#     hists = {}
#     for color in ['r', 'g', 'b']:
#         for bin_label in bins_labels:
#             hists[color][bin_label]['sum'] = 0.0
#             hists[color][bin_label]['count'] = 0
#     with h5py.File(image_file, 'r') as f_img:
#         for isic_id in tqdm(isic_ids):
#             # print(isic_id)
#             record = metadata.iloc[isic_id_lut[isic_id]].to_dict()

#             assert isic_id == record['isic_id']

#             img_np = f_img[isic_id][()].transpose(1, 2, 0)
#             img = Image.fromarray(img_np)

#             img_np = np.array(img)

#             band_count = img_np.size // 3

#             stats = ImageStat.Stat(img)

#             for j, color in enumerate(['r', 'g', 'b']):
#                 record[f"img_{color}_mean"] = stats.mean[j]
#                 record[f"img_{color}_std"] = stats.stddev[j]

#                 hist = np.histogram(img_np[..., j], bins)[0] / band_count
#                 for k, v in enumerate(hist.tolist()):
#                     # record[f"img_{color}_hist_{bins[k]}_{bins[k+1]}"]=hist[k].item()
#                     hists[color][bins_labels[k]]['sum'] += v
#                     hists[color][bins_labels[k]]['count'] += 1

#             data.append(record)

#     data_df = pd.DataFrame(data)
#     data_df.to_parquet('monitoring_reference_data.parquet', engine='pyarrow')

#     mlflow.log_artifact('monitoring_reference_data.parquet', 'monitoring_reference_data')

#     os.remove('monitoring_reference_data.parquet')

#     cumul_hist = []
#     for color in ['r', 'g', 'b']:
#         for bin_label in bins_labels:
#             cumul_hist.append(
#                 {
#                     "color": color,
#                     "bin_label": bin_label,
#                     "value": hists[color][bin_label]['sum'] / hists[color][bin_label]['count'],
#                 }
#             )

#     cumul_hist_df = pd.DataFrame(cumul_hist)
#     cumul_hist_df.to_parquet('monitoring_reference_cumul_hist.parquet', engine='pyarrow')

#     mlflow.log_artifact(
#         'monitoring_reference_cumul_hist.parquet', 'monitoring_reference_cumul_hist'
#     )

#     os.remove('monitoring_reference_cumul_hist.parquet')


def preprocess_data(
    input_data_path='../../project_data_prepare/split_dataset/',
    preprocessed_data_path='./preprocessed_data/',
    preprocess_transform_json='./preprocess_transform.json'
):
    """
    main preprocessing function
    """
    preprocess_transform = create_pipeline(preprocess_transform_json)
    if osp.exists(preprocessed_data_path):
        if osp.isdir(preprocessed_data_path):
            assert len(os.listdir(preprocessed_data_path)) == 0, 'output path is not empty'
        else:
            raise NotADirectoryError('invalid output path')
    else:
        os.makedirs(preprocessed_data_path)
    run_context = os.getenv("run_context").replace("\'", "\"")
    run_context = json.loads(run_context)



    mlflow.set_tracking_uri(uri=run_context["mlflow_server_uri"])
    mlflow.set_experiment(run_context["experiment_name"])
    with mlflow.start_run(run_id=run_context["run_id"]):

        mlflow.log_artifacts(
            str(Path(__file__).parent.resolve()), artifact_path="data_preprocessor"
        )

        shutil.copy(osp.join(input_data_path, 'train-metadata.csv'), preprocessed_data_path)
        # shutil.copy(osp.join(input_data_path,'test-metadata.csv'),
        #            preprocessed_data_path)
        hist_bins_size = int(os.getenv("monitoring_img_hist_bins_size", "50"))

        bins = np.arange(0, 256, hist_bins_size)
        bins[-1] = 255

        bins_labels = [str(x / 2) for x in (bins[:-1] + bins[1:]).tolist()]

        metadata = pd.read_csv(osp.join(input_data_path, 'train-metadata.csv'),
                               low_memory=False)

        isic_id_lut = {idx: k for k, idx in enumerate(metadata['isic_id'])}

        isic_ids=metadata['isic_id'].tolist()

        data = []

        hists = {}
        for color in ['r', 'g', 'b']:
            hists[color]={}
            for bin_label in bins_labels:
                hists[color][bin_label]={}
                hists[color][bin_label]['sum'] = 0.0
                hists[color][bin_label]['count'] = 0

        with h5py.File(osp.join(input_data_path, 'train-image.hdf5'), 'r') as f_in:
            with h5py.File(
                    osp.join(preprocessed_data_path, 'train-image.hdf5'), 'w'
                ) as f_out:
                for isic_id in tqdm(isic_ids):

                    record = metadata.iloc[isic_id_lut[isic_id]].to_dict()
                    assert isic_id == record['isic_id']

                    pil_image = Image.open(io.BytesIO(f_in[isic_id][()]))
                    stats = ImageStat.Stat(pil_image)
                    for j, color in enumerate(['r', 'g', 'b']):
                        record[f"img_{color}_mean"] = stats.mean[j]
                        record[f"img_{color}_std"] = stats.stddev[j]

                    image_np = np.array(pil_image)
                    band_count = image_np.size // 3

                    for j, color in enumerate(['r', 'g', 'b']):

                        hist = np.histogram(image_np[..., j], bins)[0] / band_count
                        for k, v in enumerate(hist.tolist()):
                            # record[f"img_{color}_hist_{bins[k]}_{bins[k+1]}"]=hist[k].item()
                            hists[color][bins_labels[k]]['sum'] += v
                            hists[color][bins_labels[k]]['count'] += 1

                    preprocessed_image_np = preprocess_transform(image=image_np)['image']
                    f_out[isic_id] = preprocessed_image_np[...].transpose(2, 0, 1)

                    data.append(record)

        data_df = pd.DataFrame(data)
        data_df.to_parquet('monitoring_reference_data.parquet', engine='pyarrow')

        mlflow.log_artifact('monitoring_reference_data.parquet', 'monitoring_reference_data')

        os.remove('monitoring_reference_data.parquet')

        cumul_hist = []
        for color in ['r', 'g', 'b']:
            for bin_label in bins_labels:
                cumul_hist.append(
                    {
                        "color": color,
                        "bin_label": bin_label,
                        "value": hists[color][bin_label]['sum'] / hists[color][bin_label]['count'],
                    }
                )

        cumul_hist_df = pd.DataFrame(cumul_hist)
        cumul_hist_df.to_parquet('monitoring_reference_cumul_hist.parquet', engine='pyarrow')

        mlflow.log_artifact(
            'monitoring_reference_cumul_hist.parquet', 'monitoring_reference_cumul_hist'
        )

        os.remove('monitoring_reference_cumul_hist.parquet')



    with open('/airflow/xcom/return.json', 'w', encoding="utf-8") as f:
        json.dump(run_context, f)
    # from IPython import embed; embed(colors='Linux')


if __name__ == '__main__':

    fire.Fire(preprocess_data)
