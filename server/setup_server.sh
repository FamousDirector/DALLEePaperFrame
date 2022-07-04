# #!/usr/bin/bash

docker build . -t dalle-mini:latest

mkdir downloaded_models
cd downloaded_models
git clone https://huggingface.co/dalle-mini/dalle-mega
git clone https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384

