#include <algorithm>
#include <chrono>
#include <iostream>
#include <string>

#include "absl/time/time.h"
#include "tensorflow/lite/delegates/gpu/cl/environment.h"
#include "tensorflow/lite/delegates/gpu/cl/inference_context.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/testing/tflite_model_reader.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

using namespace std;
using namespace std::chrono;
using namespace tflite;
using namespace tflite::gpu;
using namespace tflite::gpu::cl;

void FillInputTensor(Interpreter* interpreter) {
  float* p = interpreter->typed_input_tensor<float>(0);
  const auto n = NumElements(interpreter->tensor(interpreter->inputs()[0]));
  std::fill(p, p + n, 1.0f);
}

float TimeSinceStartMS(const high_resolution_clock::time_point& start) {
  return duration_cast<nanoseconds>(high_resolution_clock::now() - start)
             .count() *
         1e-6;
}

float GetInferenceLatencyMS(Interpreter* interpreter, int num_runs) {
  const auto start = high_resolution_clock::now();
  for (int i = 0; i < num_runs; ++i) interpreter->Invoke();
  return TimeSinceStartMS(start) / num_runs;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cout << "Expected model path as second argument.";
    return -1;
  }

  const int kNumRuns = 10;
  const int kNumExecutions = 150;

  // Load the TFLite model.
  auto model = FlatBufferModel::BuildFromFile(argv[1]);
  ops::builtin::BuiltinOpResolver op_resolver;
  InterpreterBuilder builder(*model, op_resolver);

  std::cout << "Execute TFLite inference with and without delegate.\n";
  // Initialize and run CPU inference.
  unique_ptr<Interpreter> cpu_inference;
  builder(&cpu_inference);
  cpu_inference->AllocateTensors();
  FillInputTensor(cpu_inference.get());
  cpu_inference->Invoke();
  const float* cpu_out = cpu_inference->typed_output_tensor<float>(0);

  // Initialize and run GPU inference.
  unique_ptr<Interpreter> gpu_inference;
  builder(&gpu_inference);
  TfLiteGpuDelegateOptionsV2 o = TfLiteGpuDelegateOptionsV2Default();
  o.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
  o.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
  o.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_MEMORY_USAGE;
  o.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
  auto* gpu_delegate = TfLiteGpuDelegateV2Create(&o);
  gpu_inference->ModifyGraphWithDelegate(gpu_delegate);
  FillInputTensor(gpu_inference.get());
  gpu_inference->Invoke();
  const float* gpu_out = gpu_inference->typed_output_tensor<float>(0);

  // Compare CPU vs GPU
  double mse = 0;
  auto out_n = NumElements(cpu_inference->tensor(cpu_inference->outputs()[0]));
  for (int i = 0; i < out_n; ++i) {
    double diff = gpu_out[i] - cpu_out[i];
    mse += diff * diff;
  }
  mse /= out_n;
  std::cout << "CPU vs GPU accuracy: " << mse << "." << std::endl;
  std::cout << "TFLite CPU->GPU->CPU inference latency:" << std::endl;
  for (int i = 0; i < kNumRuns; i++) {
    std::cout << GetInferenceLatencyMS(gpu_inference.get(), kNumExecutions)
              << " ms" << std::endl;
  }
  TfLiteGpuDelegateV2Delete(gpu_delegate);

  auto load_status = tflite::gpu::cl::LoadOpenCL();
  if (!load_status.ok()) {
    std::cout << load_status.message();
    return -1;
  }

  std::cout << "Execute GPU-only OpenCL inference (no TFLite interface).\n";
  GraphFloat32 graph_cl;
  BuildFromFlatBuffer(*model, op_resolver, &graph_cl);

  Environment env;
  CreateEnvironment(&env);

  InferenceContext::CreateInferenceInfo create_info;
  create_info.precision = env.IsSupported(CalculationsPrecision::F16)
                              ? CalculationsPrecision::F16
                              : CalculationsPrecision::F32;
  create_info.storage_type = GetFastestStorageType(env.device().GetInfo());
  create_info.hints.Add(ModelHints::kAllowSpecialKernels);
  InferenceContext context;
  context.InitFromGraphWithTransforms(create_info, &graph_cl, &env);

  std::cout << "OpenCL GPU-only inference latency:" << std::endl;
  auto* queue = env.profiling_queue();
  for (int i = 0; i < kNumRuns; ++i) {
    const auto start = high_resolution_clock::now();
    for (int k = 0; k < kNumExecutions; ++k) {
      context.AddToQueue(env.queue());
    }
    env.queue()->WaitForCompletion();
    std::cout << TimeSinceStartMS(start) / kNumExecutions << "ms" << std::endl;
  }
  return EXIT_SUCCESS;
}
