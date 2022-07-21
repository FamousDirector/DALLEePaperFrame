#!/usr/bin/bash

# build docker images
if [[ "$(uname -a)" == *"x86"* ]]; then
    docker build -f ./x86/Dockerfile.api . -t art-generator-api:latest
    docker build -f ./x86/Dockerfile.triton . -t triton-inference-server:latest
    docker build -f ./x86/Dockerfile.export . -t pytorch-onnx-export:latest
elif [[ "$(uname -a)" == *"tegra"* ]]; then
  # see here if you want to build for arm64 on x86_64: https://github.com/multiarch/qemu-user-static
    docker build -f ./Jetson/Dockerfile.api . -t art-generator-api:latest
    docker build -f ./Jetson/Dockerfile.triton . -t triton-inference-server:latest
    docker build -f ./Jetson/Dockerfile.export . -t pytorch-onnx-export:latest
else
    echo "Unsupported system"
    exit 1
fi

# check if docker-compose is installed
if ! command -v docker-compose >/dev/null 2>&1; then
    echo "docker-compose is not installed - Please install!"
    exit 1
fi

# create onnx file
docker run -v ${PWD}:/workspace -it pytorch-onnx-export python3 extract_onnx_models.py
mv asr.onnx model_repo/asr_infer/1/asr.onnx
