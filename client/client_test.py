# https://github.com/pimoroni/inky/tree/master/examples/7color
import io
import requests
import argparse
from PIL import Image

from frame_composer import FrameComposer

width, height = 600, 448
generated_image_size = 350

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--server-address', help='Server address')
    parser.add_argument('-p', '--server-port', default='8000', help='Server port')

    args = parser.parse_args()

    # request the image from the server
    query_text = 'astronaut in space eating a sandwich'
    response = requests.get('http://'+args.server_address+':'+args.server_port+
                            '/generate/'+query_text+'?size={}'.format(generated_image_size))
    generated_image = Image.open(io.BytesIO(response.content))

    # paste the image onto the display
    fc = FrameComposer(width, height)
    display_image = fc.create_frame_image(generated_image, query_text)
    display_image.show()
