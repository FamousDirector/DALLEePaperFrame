# https://github.com/pimoroni/inky/tree/master/examples/7color
import io
import requests
import inky
import argparse
from PIL import Image

from frame_composer import FrameComposer
from buttons import set_button_function, wait_forever_for_button_presses

display = inky.auto()
width, height = display.resolution

fc = FrameComposer(width, height)

GENERATOR_TEXT_PROMPT = 'cartoon of earth'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server-address', help='Server address')
    parser.add_argument('-p', '--server-port', default='8000', help='Server port')

    args = parser.parse_args()


    def display_new_generated_image(_):
        print('Generating new image...')

        # request the image from the server
        response = requests.get('http://' +
                                args.server_address + ':' + args.server_port
                                + '/generate/' + GENERATOR_TEXT_PROMPT)
        generated_image = Image.open(io.BytesIO(response.content))
        print("Received image from server")

        display_image = fc.create_frame_image(generated_image, GENERATOR_TEXT_PROMPT)

        # finally, display the complete image on the display hardware
        display.set_image(display_image)
        display.set_border(inky.BLACK)
        display.show()
        print("Displayed image on display")


    # Set up the buttons
    set_button_function('A', display_new_generated_image)
    set_button_function('B', lambda _: print('B pressed'))
    set_button_function('C', lambda _: print('C pressed'))
    set_button_function('D', lambda _: print('D pressed'))

    # Wait forever for button presses (ie while true)
    print("Setup complete. Waiting for button presses...")
    wait_forever_for_button_presses()
