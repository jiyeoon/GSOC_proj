import os
import sqlite3
import subprocess
from itertools import product 
import numpy as np
import pandas as pd

from run_bazel import run_bazel
from gsoc_proj.generate_model import generate_model


def test():
    kernel_list = [1, 3, 5]
    filter_list = [16, 32]
    input_img_list = [(112, 112, 3)]

    generate_model(kernel_list=kernel_list, filter_list=filter_list, input_img_list=input_img_list)
    
    conn = sqlite3.connect('gsoc_proj/gsoc.db')
    curr = conn.cursor()

    path = os.path.dirname(os.path.abspath('__file__'))
    path = os.path.join(path, 'gsoc_proj', 'MODELS')
    model_list = os.listdir(path)
    for model_name in model_list:
        print(model_name, " bazel run start!")
        latency = run_bazel(model_name)
        print(latency)
        sql = f"UPDATE gsoc SET latency_time={latency} WHERE tflite_name={model_name}"
        curr.execute(sql)
    
    conn.commit()
    conn.close()