# #!/usr/bin/bash

export WANDB_API_KEY=$1

# build docker images
docker build . -t dalle-mini:latest

# download weights - some of them are in huggingface, others in WandB
pip install wandb
mkdir downloaded_models
cd downloaded_models
git clone https://huggingface.co/dalle-mini/vqgan_imagenet_f16_16384

python3 - << EOF
import wandb

wandb.login(key="$WANDB_API_KEY")
DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"
dir = "dalle-mini/"

# wandb artifact
if wandb.run is not None:
    artifact = wandb.run.use_artifact(DALLE_MODEL)
else:
    artifact = wandb.Api().artifact(DALLE_MODEL)
artifact.download(dir)

EOF
