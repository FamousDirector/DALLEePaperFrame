from model import DALLEMini

if __name__ == "__main__":
    dalle_mini = DALLEMini()

    # Generate an image from a prompt
    prompt = "sunset over a lake in the mountains"
    image = dalle_mini.generate_images(prompt, print_time=True)[0]

    # save image
    image.save("generated_image.png")

