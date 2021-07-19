set -e

if [ -z "$1" ]; then
    echo "No model provided."
    exit
fi

model_path=$1
model_name=${model_path##*/}  # finds last token after '/'
tmp_dir=/data/local/tmp/gsoc

bazel build --config=android_arm64  -c opt gsoc_proj:test
adb shell rm -rf $tmp_dir
adb shell mkdir $tmp_dir
adb push "${PWD}"/bazel-bin/gsoc_proj/test $tmp_dir
adb push $model_path $tmp_dir
adb shell chmod +x $tmp_dir/test
adb shell $tmp_dir/test $tmp_dir/$model_name
