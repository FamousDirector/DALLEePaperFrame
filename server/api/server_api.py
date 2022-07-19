import os
import time
import tritonclient.grpc as tritonclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse

app = FastAPI()
base_img_path = "../generated_images/"

triton_client = tritonclient.InferenceServerClient(url="triton-inference-server:8001")


# generate an image file from a given text
@app.get("/generate/{text}", response_class=FileResponse)
def generate_image(text: str, size: int = 256):
    text = ''.join(e for e in text if e.isalnum() or e == ' ')

    # generate an image from the text using the Triton model
    inputs = []
    outputs = []
    input_name = "text_prompt"
    output_name = "generated_image"
    inputs.append(tritonclient.InferInput(input_name, [1, 1], np_to_triton_dtype(np.object_)))
    outputs.append(tritonclient.InferRequestedOutput(output_name))

    inputs[0].set_data_from_numpy(np.asarray([[text]], dtype=np.object_))
    results = triton_client.infer(model_name="min_dalle",
                                  inputs=inputs,
                                  outputs=outputs)

    generated_image_array = results.as_numpy(output_name)

    image = Image.fromarray((generated_image_array.astype(np.uint8)))
    image = image.resize((size, size), Image.ANTIALIAS)

    image_path = os.path.join(base_img_path, f"{text.replace(' ', '_')}.png")

    os.makedirs(base_img_path, exist_ok=True)
    image.save(image_path)

    # remove any images that are older than 24 hours
    for file in os.listdir(base_img_path):
        file_path = os.path.join(base_img_path, file)
        if os.path.isfile(file_path):
            if os.stat(file_path).st_mtime < (time.time() - 1 * 60 * 60):
                os.remove(file_path)

    return FileResponse(image_path, media_type="image/png")


@app.get("/transcribe")
def transcribe_audio_file(file: UploadFile):
    if file.content_type != "audio/wav":
        return "Invalid file type"
    # TODO - transcribe audio file
    return {"transcribed_text": "TODO"}
