"""
Initial data prepare
"""
import os
import os.path as osp

import fire
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main(
    input_csv='./train-metadata.csv',
    input_hdf5='./train-image.hdf5',
    output_path='./split_dataset/',
    random_seed=1,
):
    """
    main script  function
    """
    if osp.exists(output_path):
        if osp.isdir(output_path):
            assert len(os.listdir(output_path)) == 0, 'output path is not empty'
        else:
            raise Exception('invalid output path') # pylint: disable=broad-exception-raised
    else:
        os.makedirs(output_path)

    np.random.seed(random_seed)

    csv_data = pd.read_csv(input_csv)
    # print(csv_data.head())

    train_data, test_data, _, _ = train_test_split(
        csv_data,
        csv_data['target'],
        test_size=1 / 8,
        random_state=random_seed,
        stratify=csv_data['target'],
    )

    train_data.to_csv(osp.join(output_path, 'train-metadata.csv'), index=False)
    test_data.to_csv(osp.join(output_path, 'test-metadata.csv'), index=False)

    with h5py.File(input_hdf5, 'r') as f_in:

        with h5py.File(osp.join(output_path, 'train-image.hdf5'), 'w') as f_out:
            # from IPython import embed; embed(colors='Linux')
            for isic_id in tqdm(train_data['isic_id'].tolist()):
                f_out[isic_id] = f_in[isic_id][...]

        with h5py.File(osp.join(output_path, 'test-image.hdf5'), 'w') as f_out:
            # from IPython import embed; embed(colors='Linux')
            for isic_id in tqdm(test_data['isic_id'].tolist()):
                f_out[isic_id] = f_in[isic_id][...]


if __name__ == '__main__':
    fire.Fire(main)
