import io
import os.path
import random
import time
import threading
import requests
import inky
import argparse
from PIL import Image

from frame_composer import FrameComposer
from buttons import set_button_function, wait_forever_for_button_presses
from record_audio import record_audio

display = inky.auto()
width, height = display.resolution

fc = FrameComposer(width, height)

last_creation_time = 0
minimum_time_between_image_generations = 5

automated_image_generation = True
automated_image_generation_time = 60 * 60 * 1  # 1 hours

saved_image_folder = 'saved_images'
if not os.path.exists(saved_image_folder):
    os.makedirs(saved_image_folder)

with open("prompts.txt") as file:
    prompts = file.readlines()
    prompts = [p.rstrip() for p in prompts]

pre_prompts = ['',
               'a cartoon of',
               'a painting of',
               'a watercolor of',
               'a comic of',
               'a stencil of',
               'a picture of',
               'a sculpture of',
               'a drawing of',
               '']


def generate_sample_prompt():
    """
    Generates a random prompt from the list of prompts and a random pre-prompt.
    :return: a string containing the prompt
    """
    pp = random.choice(pre_prompts)
    p = random.choice(prompts)
    return pp + ' ' + p if pp != '' else p


def generate_new_image(text_prompt, generated_image_size=350):
    print('Generating new image...')

    # request the image from the server
    response = requests.get('http://' + args.server_address + ':' + args.server_port +
                            '/generate/' + text_prompt + '?size={}'.format(generated_image_size))
    generated_image = Image.open(io.BytesIO(response.content))

    print("Received image from server")
    return generated_image


def display_image_on_frame(image, text_prompt):
    print("Displaying image on frame")
    frame_image = fc.create_frame_image(image, text_prompt)
    display.set_image(frame_image)
    display.set_border(inky.BLACK)
    display.show()
    print("Displayed image on display")


def save_image_to_file(image, text_prompt):
    image.save(os.path.join(saved_image_folder, text_prompt.replace(' ', '_') + '.png'))

    # remove the oldest image if there are more than 100 images in the folder
    if len(os.listdir(saved_image_folder)) > 100:
        oldest_image_path = os.path.join(saved_image_folder, sorted(os.listdir(saved_image_folder))[0])
        os.remove(oldest_image_path)


def load_image_from_file(text_prompt):
    # find matching image in the saved images folder
    image_path = os.path.join(saved_image_folder, text_prompt.replace(' ', '_') + '.png')
    if os.path.isfile(image_path):
        return Image.open(image_path), text_prompt
    else:
        # return random image if no matching image is found
        random_file_name = random.choice(os.listdir(saved_image_folder))
        return Image.open(os.path.join(saved_image_folder, random_file_name)), \
               random_file_name.split('.')[0].replace('_', ' ')


def get_text_prompt_from_audio(audio_file_name):
    url = 'http://' + args.server_address + ':' + args.server_port + '/transcribe'
    files = {'file': (audio_file_name, open(audio_file_name, 'rb'), 'audio/x-wav')}
    response = requests.post(url, files=files)
    text_prompt = response.text.replace('"', '').strip()
    print("Received text prompt from server - {}".format(text_prompt))
    return text_prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server-address', help='Server address')
    parser.add_argument('-p', '--server-port', default='8000', help='Server port')

    args = parser.parse_args()

    GENERATOR_TEXT_PROMPT = generate_sample_prompt()


    def display_new_generated_image_w_same_prompt(_=None):
        global last_creation_time

        if time.time() - last_creation_time > minimum_time_between_image_generations:  # debounce the button press
            global GENERATOR_TEXT_PROMPT

            # generate and display a new image
            try:
                generated_image = generate_new_image(GENERATOR_TEXT_PROMPT)
                save_image_to_file(generated_image, GENERATOR_TEXT_PROMPT)
            except Exception as e:
                print("A problem occurred: ", e)
                generated_image, GENERATOR_TEXT_PROMPT = load_image_from_file(GENERATOR_TEXT_PROMPT)
            display_image_on_frame(generated_image, GENERATOR_TEXT_PROMPT)

            last_creation_time = time.time()


    def display_new_generated_image_w_new_prompt(_=None):
        global last_creation_time

        if time.time() - last_creation_time > minimum_time_between_image_generations:  # debounce the button press
            global GENERATOR_TEXT_PROMPT

            GENERATOR_TEXT_PROMPT = generate_sample_prompt()  # generate a new prompt

            # generate and display a new image
            try:
                generated_image = generate_new_image(GENERATOR_TEXT_PROMPT)
                save_image_to_file(generated_image, GENERATOR_TEXT_PROMPT)
            except Exception as e:
                print("A problem occurred: ", e)
                generated_image, GENERATOR_TEXT_PROMPT = load_image_from_file(GENERATOR_TEXT_PROMPT)
            display_image_on_frame(generated_image, GENERATOR_TEXT_PROMPT)

            last_creation_time = time.time()


    def display_new_generated_image_w_recorded_prompt(_=None):
        global last_creation_time

        if time.time() - last_creation_time > minimum_time_between_image_generations:
            global GENERATOR_TEXT_PROMPT

            # record the user's voice
            print('Recording audio...')
            audio_file_name = record_audio()
            print('Finished recording audio')

            # get the text prompt from the audio file
            GENERATOR_TEXT_PROMPT = get_text_prompt_from_audio(audio_file_name)

            # generate and display a new image
            try:
                generated_image = generate_new_image(GENERATOR_TEXT_PROMPT)
                save_image_to_file(generated_image, GENERATOR_TEXT_PROMPT)
            except Exception as e:
                print("A problem occurred: ", e)
                generated_image, GENERATOR_TEXT_PROMPT = load_image_from_file(GENERATOR_TEXT_PROMPT)
            display_image_on_frame(generated_image, GENERATOR_TEXT_PROMPT)

            last_creation_time = time.time()


    def toggle_auto_image_generation(_=None):
        global automated_image_generation
        automated_image_generation = not automated_image_generation

        if automated_image_generation:
            print("Automated image generation enabled")
        else:
            print("Automated image generation disabled")


    # Set up the buttons
    set_button_function('A', display_new_generated_image_w_same_prompt)
    set_button_function('B', display_new_generated_image_w_new_prompt)
    set_button_function('C', display_new_generated_image_w_recorded_prompt)
    set_button_function('D', toggle_auto_image_generation)


    # set display to auto create a new image every N hours
    def image_generation_timer():
        if automated_image_generation and time.time() - last_creation_time > minimum_time_between_image_generations:
            print('Automated image generation started')
            random.choice([display_new_generated_image_w_same_prompt, display_new_generated_image_w_new_prompt])()
        threading.Timer(automated_image_generation_time, image_generation_timer).start()


    image_generation_timer()

    # Wait forever for button presses (ie while true)
    print("Setup complete. Waiting for button presses...")
    wait_forever_for_button_presses()
