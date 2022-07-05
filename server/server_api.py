import os
import time
from fastapi import FastAPI
from fastapi.responses import FileResponse

from model import DALLEMini

app = FastAPI()
dalle_mini = DALLEMini()

base_img_path = "./generated_images/"


# generate an image file from a given text
@app.get("/generate/{text}", response_class=FileResponse)
def generate_image(text: str):
    text = ''.join(e for e in text if e.isalnum() or e == ' ')
    image = dalle_mini.generate_images(text, print_time=True)[0]

    image_path = os.path.join(base_img_path, f"{text.replace(' ','_')}.png")

    os.makedirs(base_img_path, exist_ok=True)
    image.save(image_path)

    # remove any images that are older than 24 hours
    for file in os.listdir(base_img_path):
        file_path = os.path.join(base_img_path, file)
        if os.path.isfile(file_path):
            if os.stat(file_path).st_mtime < (time.time() - 24 * 60 * 60):
                os.remove(file_path)

    return FileResponse(image_path, media_type="image/png")
