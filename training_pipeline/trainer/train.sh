#!/usr/bin/env bash
cd "$(dirname "$0")"
pip install addict pytorch-lightning h5py
python train.py