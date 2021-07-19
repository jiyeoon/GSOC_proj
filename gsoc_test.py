import subprocess
import os

def run_bazel(model_path):
    proc = subprocess.Popen(
        ['sh', 'gsoc_proj/run.sh', './gsoc_proj/{}'.format(model_path)],
        #['sh', 'tmp.sh'],
        stdout = subprocess.PIPE
    )
    out, err = proc.communicate()
    tmp = out.decode('utf-8').split('\n')[-11:-1]
    lst = [float(t.replace('ms', '')) for t in tmp] # get only gpu inference time
    return sum(lst) / len(lst) # return the latency time result 

#print(run_bazel(''))

