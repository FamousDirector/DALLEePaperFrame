from PIL import Image, ImageDraw, ImageFont


class FrameComposer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.large_font_height = 35
        self.small_font_height = 15
        self.large_font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf", self.large_font_height)
        self.small_font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf", self.small_font_height)

        self.image_x = 25
        self.image_y = 25
        self.border_width = 5

    def create_frame_image(self, image, text_prompt):
        base_image = Image.new("RGB", (self.width, self.height), "white")

        draw = ImageDraw.Draw(base_image)

        # add text
        draw.rectangle((self.image_x - self.border_width,
                        self.image_y - self.border_width,
                        self.image_x + self.border_width + image.width,
                        self.image_y + self.border_width + image.height),
                       fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((self.image_x - self.border_width, self.image_y + self.border_width + image.height),
                  f"\"{text_prompt}\"", font=self.large_font, fill=(0, 255, 0))

        draw.text((self.image_x - self.border_width,
                   self.image_y + self.border_width + image.height + self.large_font_height + self.border_width),
                  "by: Dalle Mini, an AI", font=self.small_font, fill=(0, 255, 0))

        # add image to "frame"
        base_image.paste(image, (self.image_x, self.image_y))

        return base_image
