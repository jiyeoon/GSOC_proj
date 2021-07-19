import os
import time
import subprocess
import sqlite3
import numpy as np
import pandas as pd

def run_bazel(model_name):
    proc = subprocess.Popen(
        ['sh', 'gsoc_proj/run.sh', './gsoc_proj/MODELS3/{}'.format(model_name)],
        stdout = subprocess.PIPE
    )
    out, err = proc.communicate()
    res = out.decode('utf-8').split('\n')[-11:-1] # onlly gpu inference latency time
    try:
        latency_time = [float(i.replace('ms', '')) for i in res]
    except:
        print(err)
        print(out.decode('utf-8').split('\n'))
        return -1
    return sum(latency_time) / len(latency_time)

def test(model_name):
    proc = subprocess.Popen(
        ['sh', 'gsoc_proj/run.sh', './gsoc_proj/MODELS/{}'.format(model_name)],
        stdout = subprocess.PIPE
    )
    out, err = proc.communicate()
    if err:
        return [], err
    else:
        res = out.decode('utf-8').split('\n')[-11:-1]
        latency_time = [float(i.replace('ms', '')) for i in res]
        return latency_time, err
    

def get_csv():
    start_time = time.time()
    path = os.path.dirname(os.path.abspath('__file__'))
    target_path = os.path.join(path, 'gsoc_proj', 'MODELS2')
    model_names = os.listdir(target_path)
    model_names.sort()
    print(target_path)

    tmp = {
        'model_name' : [],
        'kernel_size' : [],
        'filter_size' : [],
        'input_shape' : [],
        'latency_time' : [],
    }

    for model_name in model_names:
        if 'kernel' not in model_name:
            continue
        print("@@@@@", model_name, " start!!!")
        latency_time = run_bazel(model_name)
        if latency_time == -1:
            print(model_name, " 얘는 해당 안됨")
            continue
        
        # model name 전처리
        _, kernel_size, _, filter_size, _, _, input_shape = model_name.replace('.tflite', '').split('_')

        tmp['model_name'].append(model_name)
        tmp['latency_time'].append(latency_time)
        tmp['kernel_size'].append(kernel_size)
        tmp['filter_size'].append(filter_size)
        tmp['input_shape'].append('(' + input_shape + ')')

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n model : {}, latency : {} \n kernel_size : {}, filter_size : {}, input_shape : {}".format(model_name, latency_time, kernel_size, filter_size, input_shape))
    
    df = pd.DataFrame(tmp)
    df.to_csv('./result2.csv')

    end_time = time.time()
    print("End Time : ", end_time - start_time)

    return


def get_csv2():
    start_time = time.time()
    path = os.path.dirname(os.path.abspath('__file__'))
    target_path = os.path.join(path, 'gsoc_proj', 'MODELS3')
    model_names = os.listdir(target_path)
    model_names.sort()
    print(target_path)

    tmp = {
        'model_name' : [],
        'kernel_size' : [],
        'filter_size' : [],
        'input_shape' : [],
        'stride' : [],
        'padding' : [],
        'latency_time' : [],
    }

    for model_name in model_names:
        if 'kernel' not in model_name:
            continue
        print("@@@@@", model_name, " start!!!")
        latency_time = run_bazel(model_name)
        if latency_time == -1:
            print(model_name, " 얘는 해당 안됨")
            continue
        
        # model name 전처리
        _, kernel_size, _, filter_size, _, _, input_shape, _, stride, _, padding  = model_name.replace('.tflite', '').split('_')

        tmp['model_name'].append(model_name)
        tmp['latency_time'].append(latency_time)
        tmp['kernel_size'].append(kernel_size)
        tmp['filter_size'].append(filter_size)
        tmp['stride'].append(stride)
        tmp['padding'].append(padding)
        tmp['input_shape'].append('(' + input_shape + ')')

        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n model : {}, latency : {} \n kernel_size : {}, filter_size : {}, input_shape : {} \n stride : {}, padding : {}".format(model_name, latency_time, kernel_size, filter_size, input_shape, stride, padding))
    
    df = pd.DataFrame(tmp)
    df.to_csv('./result3.csv')

    end_time = time.time()
    print("Total Time : ", end_time - start_time)

    return


if __name__ == "__main__":
    get_csv2()
    #get_csv()
    #test('conv2d.tflite')
    """ 
    #sqlite3 connect
    conn = sqlite3.connect('gsoc.db')
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
    """