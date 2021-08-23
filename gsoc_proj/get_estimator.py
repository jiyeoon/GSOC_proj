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


def get_estimator(from_manual, from_real_model):
    df = pd.read_csv(from_manual)
    df['filter_size'] = np.log1p(df['filter_size'])
    df['input_h'] = np.log1p(df['input_h'])
    df['input_w'] = np.log1p(df['input_w'])
    df['input_channel'] = np.log1p(df['input_channel'])

    test_df = pd.read_csv(from_real_model)
    test_df['filter_size'] = np.log1p(test_df['filter_size'])
    test_df['input_h'] = np.log1p(test_df['input_h'])
    test_df['input_w'] = np.log1p(test_df['input_w'])
    test_df['input_channel'] = np.log1p(test_df['input_channel'])

    train_test_df = pd.concat([df, test_df])
    train_test_target = train_test_df[['latency_time']]
    train_test_df = train_test_df[['kernel_size', 'filter_size', 'input_h', 'input_w','input_channel', 'stride']]

    X_train, X_test, y_train, y_test = train_test_split(
        train_test_df, train_test_target, test_size = 0.3, #random_state=2020
    )

    models = [DecisionTreeRegressor(), RandomForestRegressor(), XGBRegressor()]

    for model in models:
        model.fit(X_train, y_train)
        print("Model R2 Score : ", model.score(X_test, y_test))
    
    joblib.dump(models, './models.pkl')


def main(result_csv_path):
    if len(sys.argv) != 3:
        print("")
    get_estimator()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="get_estimator.py description")
    
    parser.add_argument('--result', required=True, help='result file path')
    parser.add_argument('--test', required=True, help='result file path for test')

    args = parser.parse_args()

    get_estimator(args.result, args.test)