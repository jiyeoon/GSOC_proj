"""
This code should be with tensorflow repository.
https://github.com/tensorflow/tensorflow

Fork above repo and execute `./configure`.
"""

import os
import time
import csv
import subprocess
from datetime import datetime
import numpy as np
import pandas as pd


# Run 1 convolution model on the mobile device
# model_name - it contains the informations about kernel, filter, input shape etc on its name
def run_bazel(model_name):
    proc = subprocess.Popen(
        ['sh', 'gsoc_proj/run.sh', './gsoc_proj/MODELS/{}'.format(model_name)],
        stdout = subprocess.PIPE
    )
    out, err = proc.communicate()
    res = out.decode('utf-8').split('\n')[-11:-1] # onlly gpu inference latency time
    try:
        latency_time = [float(i.replace('ms', '')) for i in res]
        return sum(latency_time) / len(latency_time)
    except ZeroDivisionError:
        print("Zero Division Error Occured!")
        return -1
    except:
        print(err)
        print(out.decode('utf-8').split('\n'))
        return -1


# get the database which is csv file form
# it saves on current directory
def get_csv():
    start_time = time.time()
    path = os.path.dirname(os.path.abspath('__file__'))
    target_path = os.path.join(path, 'gsoc_proj', 'MODELS')
    model_names = os.listdir(target_path)
    model_names.sort()
    print(target_path)

    # CSV Write
    today = datetime.today().strftime("%Y%m%d")
    f = open('./result_{}.csv'.format(today), 'w', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(['model_name', 'kernel_size', 'filter_size', 'input_h', 'input_w', 'input_channel', 'stride', 'padding', 'latency_time'])

    for model_name in model_names:
        if 'kernel' not in model_name:
            continue
        print("@@@@@", model_name, " start!!!")
        latency_time = run_bazel(model_name)
        if latency_time == -1:
            print(model_name, " Excluded!")
            continue
        
        # model name 전처리
        _, kernel_size, _, filter_size, _, _, input_shape, _, stride, _, padding  = model_name.replace('.tflite', '').split('_')

        input_channel, input_h, input_w = input_shape.split(',')
        
        wr.writerow([
            model_name,
            kernel_size,
            filter_size,
            input_h,
            input_w,
            input_channel,
            stride,
            padding,
            latency_time
        ])

        print("\n model : {}, latency : {} \n kernel_size : {}, filter_size : {}, input_shape : {} \n stride : {}, padding : {}".format(model_name, latency_time, kernel_size, filter_size, input_shape, stride, padding))

    end_time = time.time()
    print("Total Consumed Time : ", end_time - start_time)
    
    f.close()
    return


if __name__ == "__main__":
    get_csv()