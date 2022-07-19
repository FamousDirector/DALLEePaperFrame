FROM nvcr.io/nvidia/pytorch:22.06-py3

RUN apt-get update && apt-get install -y \
  git \
  python3 \
  python3-pip

RUN pip3 install -q \
    min_dalle==0.3.11 \
    fastapi==0.78.0 \
    uvicorn[standard] \
    python-multipart==0.0.5

WORKDIR /workspace

# predownload the models
ADD generation_model.py generation_model.py
RUN python3 generation_model.py
