# tensorflow package -> path warning!
import re
import sys
import numpy as np
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


def main(model_path): # 모델 path
    tflite_input = model_path
    with open(tflite_input, 'rb') as file_handle:
        file_data = bytearray(file_handle.read())
    data = CreateDictFromFlatbuffer(file_data)
    print(len(data['subgraphs']))
    for subgraph in data['subgraphs']:
        for tensor in data['subgraphs'][0]['tensors']:
          print(tensor)
    print(data['subgraphs'][0].keys())

if __name__ == "__main__":
    model_path = sys.argv[-1]
    main(model_path)