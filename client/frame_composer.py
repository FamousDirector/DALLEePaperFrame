from PIL import Image, ImageDraw, ImageFont


class FrameComposer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.large_font_height = 30
        self.small_font_height = 20
        self.large_font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
                                             self.large_font_height)
        self.small_font = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/arial.ttf",
                                             self.small_font_height)

        self.image_y = 25
        self.border_width = 5

        self.credit_text = "by: Dalle Mini, an AI"

    def create_frame_image(self, image, text_prompt):
        base_image = Image.new("RGB", (self.width, self.height), "white")

        draw = ImageDraw.Draw(base_image)

        # add text
        draw.rectangle(((self.width - image.width) // 2 - self.border_width,
                        self.image_y - self.border_width,
                        image.width + (self.width - image.width) // 2 + self.border_width,
                        self.image_y + self.border_width + image.height),
                       fill=(255, 255, 255), outline=(0, 0, 0))
        large_text_w, large_text_h = draw.textsize(text_prompt, font=self.large_font)
        draw.text(((self.width - large_text_w) // 2 - 10,
                   self.image_y + self.border_width + image.height),
                  f"\"{text_prompt}\"", font=self.large_font, fill=(0, 0, 0))

        small_text_w, small_text_h = draw.textsize(self.credit_text, font=self.small_font)
        draw.text(((self.width - small_text_w) // 2,
                   self.image_y + self.border_width + image.height + large_text_h),
                  self.credit_text, font=self.small_font, fill=(150, 0, 0))

        # add image to "frame"
        base_image.paste(image, ((self.width - image.width) // 2,
                                 self.image_y))

        return base_image
