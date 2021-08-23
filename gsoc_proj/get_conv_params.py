import re
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema_fb

def CamelCaseToSnakeCase(camel_case_input):
    """Converts an identifier in CamelCase to snake_case."""
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", camel_case_input)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def FlatbufferToDict(fb, preserve_as_numpy):
  """Converts a hierarchy of FB objects into a nested dict.
  We avoid transforming big parts of the flat buffer into python arrays. This
  speeds conversion from ten minutes to a few seconds on big graphs.
  Args:
    fb: a flat buffer structure. (i.e. ModelT)
    preserve_as_numpy: true if all downstream np.arrays should be preserved.
      false if all downstream np.array should become python arrays
  Returns:
    A dictionary representing the flatbuffer rather than a flatbuffer object.
  """
  if isinstance(fb, int) or isinstance(fb, float) or isinstance(fb, str):
    return fb
  elif hasattr(fb, "__dict__"):
    result = {}
    for attribute_name in dir(fb):
      attribute = fb.__getattribute__(attribute_name)
      if not callable(attribute) and attribute_name[0] != "_":
        snake_name = CamelCaseToSnakeCase(attribute_name)
        preserve = True if attribute_name == "buffers" else preserve_as_numpy
        result[snake_name] = FlatbufferToDict(attribute, preserve)
    return result
  elif isinstance(fb, np.ndarray):
    return fb if preserve_as_numpy else fb.tolist()
  elif hasattr(fb, "__len__"):
    return [FlatbufferToDict(entry, preserve_as_numpy) for entry in fb]
  else:
    return fb


def CreateDictFromFlatbuffer(buffer_data):
  model_obj = schema_fb.Model.GetRootAsModel(buffer_data, 0)
  model = schema_fb.ModelT.InitFromObj(model_obj)
  return FlatbufferToDict(model, preserve_as_numpy=False)

def TensorTypeToName(tensor_type):
    """Converts a numerical enum to a readable tensor type."""
    for name, value in schema_fb.TensorType.__dict__.items():
        if value == tensor_type:
            return name
    return None
  
def BuiltinCodeToName(code):
  """Converts a builtin op code enum to a readable name."""
  for name, value in schema_fb.BuiltinOperator.__dict__.items():
    if value == code:
      return name
  return None


def NameListToString(name_list):
  """Converts a list of integers to the equivalent ASCII string."""
  if isinstance(name_list, str):
    return name_list
  else:
    result = ""
    if name_list is not None:
      for val in name_list:
        result = result + chr(int(val))
    return result


class OpCodeMapper(object):
  """Maps an opcode index to an op name."""

  def __init__(self, data):
    self.code_to_name = {}
    for idx, d in enumerate(data["operator_codes"]):
      self.code_to_name[idx] = BuiltinCodeToName(d["builtin_code"])
      if self.code_to_name[idx] == "CUSTOM":
        self.code_to_name[idx] = NameListToString(d["custom_code"])

  def __call__(self, x):
    if x not in self.code_to_name:
      s = "<UNKNOWN>"
    else:
      s = self.code_to_name[x]
    return "%s (%d)" % (s, x)


class DataSizeMapper(object):
  """For buffers, report the number of bytes."""

  def __call__(self, x):
    if x is not None:
      return "%d bytes" % len(x)
    else:
      return "--"


class TensorMapper(object):
  """Maps a list of tensor indices to a tooltip hoverable indicator of more."""

  def __init__(self, subgraph_data):
    self.data = subgraph_data

  def __call__(self, x):
    res = {
        'tensor_idx' : [],
        'name' : [],
        'tensor_type' : [],
        'shape' : [],
        'shape_signature' : []
    }
    for tensor_idx in x:
        tensor = self.data['tensors'][tensor_idx]
        res['tensor_idx'].append(tensor_idx)
        res['name'].append(NameListToString(tensor['name']))
        res['tensor_type'].append(TensorTypeToName(tensor['type']))
        res['shape'].append(repr(tensor['shape']) if 'shape' in tensor else '[]')
        res['shape_signature'].append(repr(tensor['shape_signature']) if 'shape_signature' in tensor else '[]')
    
    return res

def GenerateGraph(subgraph_idx, g, opcode_mapper):
  """Produces the HTML required to have a d3 visualization of the dag."""

  def TensorName(idx):
    return "t%d" % idx

  def OpName(idx):
    return "o%d" % idx

  edges = []
  nodes = []
  first = {}
  second = {}
  pixel_mult = 200  # TODO(aselle): multiplier for initial placement
  width_mult = 170  # TODO(aselle): multiplier for initial placement
  for op_index, op in enumerate(g["operators"] or []):

    for tensor_input_position, tensor_index in enumerate(op["inputs"]):
      if tensor_index not in first:
        first[tensor_index] = ((op_index - 0.5 + 1) * pixel_mult,
                               (tensor_input_position + 1) * width_mult)
      edges.append({
          "source": TensorName(tensor_index),
          "target": OpName(op_index)
      })
    for tensor_output_position, tensor_index in enumerate(op["outputs"]):
      if tensor_index not in second:
        second[tensor_index] = ((op_index + 0.5 + 1) * pixel_mult,
                                (tensor_output_position + 1) * width_mult)
      edges.append({
          "target": TensorName(tensor_index),
          "source": OpName(op_index)
      })

    nodes.append({
        "id": OpName(op_index),
        "name": opcode_mapper(op["opcode_index"]),
        "group": 2,
        "x": pixel_mult,
        "y": (op_index + 1) * pixel_mult
    })
  for tensor_index, tensor in enumerate(g["tensors"]):
    initial_y = (
        first[tensor_index] if tensor_index in first else
        second[tensor_index] if tensor_index in second else (0, 0))

    nodes.append({
        "id": TensorName(tensor_index),
        "name": "%r (%d)" % (getattr(tensor, "shape", []), tensor_index),
        "group": 1,
        "x": initial_y[1],
        "y": initial_y[0]
    })
  graph_str = json.dumps({"nodes": nodes, "edges": edges})


def read_model(model_path):
    tflite_input = model_path
    with open(tflite_input, 'rb') as file_handle:
        file_data = bytearray(file_handle.read())
    data = CreateDictFromFlatbuffer(file_data)
    return data


def get_detail(items, keys_to_print):
    res = []
    for idx, tensor in enumerate(items):
        tmp = {}
        for h, mapper in keys_to_print:
            val = tensor[h] if h in tensor else None
            val = val if mapper is None else mapper(val)
            tmp[h] = val
        res.append(tmp)
    return res


def get_params(model_path):
    data = read_model(model_path)
    buffer_keys_to_display = [("data", DataSizeMapper())]
    operator_keys_to_display = [("builtin_code", BuiltinCodeToName),
                           ("custom_code", NameListToString),
                           ('version', None)]
    for d in data['operator_codes']:
        d['builtin_code'] = max(d['builtin_code'], d['deprecated_builtin_code'])

    res = {
        'name' : [],
        'kernel_size' : [],
        'filter_size' : [],
        'input_channel' : [],
        'input_h' : [],
        'input_w' : [],
        'stride' : [],
    }

    for subgraph_idx, g in enumerate(data['subgraphs']):
        tensor_mapper = TensorMapper(g)
        opcode_mapper = OpCodeMapper(data)
        op_keys_to_display = [('inputs', tensor_mapper),
                         ('outputs', tensor_mapper),
                         ('builtin_options', None),
                         ('opcode_index', opcode_mapper)]
        tensor_keys_to_display = [('name', NameListToString),
                             ('type', TensorTypeToName),
                             ('shape', None),
                             ('shape_signature', None),
                             ('buffer', None),
                             ('quantization', None)]
        tensors = get_detail(g['tensors'], tensor_keys_to_display)
        ops = get_detail(g['operators'], op_keys_to_display)

        for op in ops:
            if 'CONV_2D' in op['opcode_index']:
                inputs = op['inputs']
                shape = inputs['shape'][1][1:-1].replace(' ', '').split(',')
                input_shape = inputs['shape'][0][1:-1].replace(' ', '').split(',')
                options = op['builtin_options']
                
                res['name'].append(inputs['name'][1])
                res['kernel_size'].append(int(shape[1]))
                res['filter_size'].append(int(shape[0]))
                res['input_channel'].append(int(shape[3]))
                res['input_h'].append(int(input_shape[1]))
                res['input_w'].append(int(input_shape[2]))
                res['stride'].append(options['stride_h'])
        
    df = pd.DataFrame(res)
    return df


if __name__ == "__main__":
    model_path = sys.argv[-1]
    if model_path == 'get_params.py':
        print("[ERROR] Input model path!")
    else:
        data = get_params(model_path)
        model_name = model_path.split('/')[-1]
        model_name = model_name.replace('.tflite', '')
        data.to_csv('./{}_params.csv'.format(model_name), index=False)

