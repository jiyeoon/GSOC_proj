import os
import argparse
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from gsoc_proj.get_conv_params import get_params

import warnings
warnings.filterwarnings(action='ignore')


def load_models():
    return joblib.load('./models.pkl')


def get_result(tflite_path):
    models = load_models()

    df = get_params(tflite_path)
    df2 = df[['kernel_size', 'filter_size', 'input_h', 'input_w', 'input_channel', 'stride']]
    df2['filter_size'] = np.log1p(df2['filter_size'])
    df2['input_h'] = np.log1p(df2['input_h'])
    df2['input_w'] = np.log1p(df2['input_w'])
    df2['input_channel'] = np.log1p(df2['input_channel'])

    predictions = np.column_stack([model.predict(df2) for model in models])
    y_pred = np.mean(predictions, axis=1)

    df['estimated latency time'] = y_pred
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFLite Latency Estimator')

    parser.add_argument('--path', required=True, help='tflite model path (required)')

    args = parser.parse_args()
    
    data = get_result(args.path)
    model_name = args.path.split('/')[-1].replace('tflite', '')
    data.to_csv('./{}_result.csv'.format(model_name))