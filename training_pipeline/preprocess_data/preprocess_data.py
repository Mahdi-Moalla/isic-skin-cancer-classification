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

import albumentations as A
import cv2
import fire
import h5py
import mlflow
import numpy as np
from PIL import Image
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






def preprocess_data(
    input_data_path='../../project_data_prepare/split_dataset/',
    preprocess_transform_json='./preprocess_transform.json',
    preprocessed_data_path='./preprocessed_data/',
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

        for split in ['train']:  # ,'test']:
            with h5py.File(osp.join(input_data_path, f'{split}-image.hdf5'), 'r') as f_in:
                with h5py.File(
                    osp.join(preprocessed_data_path, f'{split}-image.hdf5'), 'w'
                ) as f_out:
                    for isic_id in tqdm(f_in.keys()):
                        pil_image = Image.open(io.BytesIO(f_in[isic_id][()]))
                        image_np = np.array(pil_image)
                        preprocessed_image_np = preprocess_transform(image=image_np)['image']
                        f_out[isic_id] = preprocessed_image_np[...].transpose(2, 0, 1)

    with open('/airflow/xcom/return.json', 'w', encoding="utf-8") as f:
        json.dump(run_context, f)
    # from IPython import embed; embed(colors='Linux')


if __name__ == '__main__':

    fire.Fire(preprocess_data)
