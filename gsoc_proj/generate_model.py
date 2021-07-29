import os
import subprocess
import sqlite3
from itertools import product
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

"""
small dataset => MODELS
kernel_list = [1, 3, 5]
#input_hw = [8, 16, 32, 64]
#input_channels = [16, 32, 48]
input_img_list = [(i, i, 3) for i in [8, 16, 32, 48]]
filter_list = [16, 32, 64]
"""
"""
kernel_list = [2*i+1 for i in range(3)]
filter_list = [2**i for i in range(4, 8)]
input_img_list = [(2*i, 2*i, 3) for i in range(32, 112)]
"""

kernel_list = [1, 3, 5]
filter_list = [16, 32, 64]
input_hw = [8, 16, 32, 64]
input_channels = [16, 32, 48]
input_img_list = []
for hw in input_hw:
    for ch in input_channels:
        input_img_list.append((ch, hw, hw))
stride = [1, 2, 3]
padding = ['valid', 'same']

def init():
    conn = sqlite3.connect('gsoc.db', isolation_level=None)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE gsoc(
            tflite_name varchar,
            kernel int,
            filter int,
            input_shape varchar,
            latency_time varchar
        )"""
    )
    conn.close()

def gen_model2():
    comb = list(product(kernel_list, filter_list, input_img_list, stride, padding))
    for com in comb:
        _kernel, _filter, _input_shape, _stride, _padding = com
        model = keras.Sequential(
            keras.layers.Conv2D(filters=_filter, kernel_size=_kernel, input_shape=_input_shape,
            padding=_padding, strides=_stride, activation='relu')
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        model_name = "kernel_{}_filter_{}_input_shape_{}_stride_{}_padding_{}.tflite".format(_kernel, _filter, _input_shape, _stride, _padding)
        model_name = model_name.replace(' ', '').replace('(', '').replace(')', '')
        with open("./MODELS3/" + model_name, 'wb') as f:
            f.write(tflite_model)



def generate_model(kernel_list=kernel_list, filter_list=filter_list, input_img_list=input_img_list):
    comb = list(product(kernel_list, filter_list, input_img_list))
    
    # conn = sqlite3.connect('gsoc.db', isolation_level=None)
    # curr = conn.cursor()

    for com in comb:
        _kernel, _filter, _input_shape = com
        model = keras.Sequential(
            keras.layers.Conv2D(filters=_filter, kernel_size=_kernel, input_shape=_input_shape,
            padding='same', activation='relu')
        )
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        model_name = "kernel_{}_filter_{}_input_shape_{}.tflite".format(_kernel, _filter, _input_shape)
        model_name = model_name.replace(' ', '').replace('(', '').replace(')', '')
        with open("./MODELS2/" + model_name, 'wb') as f:
            f.write(tflite_model)
        
        """
        sql = f'INSERT INTO gsoc VALUES(tflite_name={model_name}, {_kernel}, {_filter}, {_input_shape}, 0)'.replace('"', '')
        try:
            curr.execute('''CREATE TABLE gsoc(
                    tflite_name varchar(100),
                    kernel int,
                    filter int,
                    input_shape varchar(100),
                    latency_time varchar(100))
                ''')
        except:
            pass
        finally:
            try:
                curr.execute(sql)
            except:
                print(sql)
                break
        
        try:
            curr.execute(sql)
        except:
            print(sql)
        rows = curr.fetchall()
        if rows:
            pass
        else:
            curr.execute('''CREATE TABLE gsoc(
                    tflite_name varchar(100),
                    kernel int,
                    filter int,
                    input_shape varchar(100),
                    latency_time varchar(100))
                ''')
            try:
                curr.execute(sql)
            except:
                print(sql)
        """
    # conn.close()

if __name__ == "__main__":
    #generate_model()
    gen_model2()