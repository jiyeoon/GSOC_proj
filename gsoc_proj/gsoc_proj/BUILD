cc_binary(
    name = "test",
    srcs = ["test.cc"],
    linkopts = [
        "-ldl",
        "-pie",
    ],
    deps = [
        "//tensorflow/lite:framework",
        "//tensorflow/lite/delegates/gpu:delegate",
        "//tensorflow/lite/delegates/gpu/cl:environment",
        "//tensorflow/lite/delegates/gpu/cl:inference_context",
        "//tensorflow/lite/delegates/gpu/common:model",
        "//tensorflow/lite/delegates/gpu/common:status",
        "//tensorflow/lite/delegates/gpu/common/testing:tflite_model_reader",
        "//tensorflow/lite/kernels:builtin_ops",
        "@com_google_absl//absl/time",
    ],
)
