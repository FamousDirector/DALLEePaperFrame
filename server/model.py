import jax
import random
import jax.numpy as jnp
import numpy as np
from PIL import Image
from functools import partial
from timeit import default_timer as timer
from flax.jax_utils import replicate
from flax.training.common_utils import shard_prng_key

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel


DALLE_MODEL_PATH = "downloaded_models/dalle-mini/"
VQGAN_MODEL_PATH = "downloaded_models/vqgan_imagenet_f16_16384/"


class DALLEMini:
    def __init__(self):
        print("Loading DALLE model...")
        self.dalle_model, self.dalle_params = DalleBart.from_pretrained(
            DALLE_MODEL_PATH, dtype=jnp.float16, _do_init=False
        )
        print("Loaded DALLE model.")

        print("Loading VQGAN model...")
        # Load VQGAN
        self.vqgan_model, self.vqgan_params = VQModel.from_pretrained(
            VQGAN_MODEL_PATH, _do_init=False
        )
        print("Loaded VQGAN model.")

        # Model parameters are replicated on each device for faster inference.

        self.dalle_params = replicate(self.dalle_params)
        self.vqgan_params = replicate(self.vqgan_params)

        print("Loading Text Processor model...")
        self.processor = DalleBartProcessor.from_pretrained(DALLE_MODEL_PATH)
        print("Loaded Text Processor model.")

        # model generation parameters
        self.gen_top_k = None
        self.gen_top_p = None
        self.temperature = None
        self.cond_scale = 10.0

        # model inference
        @partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
        def p_generate(
                tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
        ):
            return self.dalle_model.generate(
                **tokenized_prompt,
                prng_key=key,
                params=params,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                condition_scale=condition_scale,
            )

        self.p_generate = p_generate

        # decode image
        @partial(jax.pmap, axis_name="batch")
        def p_decode(indices, params):
            return self.vqgan_model.decode_code(indices, params=params)

        self.p_decode = p_decode

        # model warmup
        print("Warming up model...")
        self.generate_images("dummy text")
        print("Warmed up model.")

    def tokenize(self, text):
        tokenized_prompts = self.processor([text])
        return replicate(tokenized_prompts)

    def generate_images(self, text, number_of_predictions=1, print_time=False):
        """
        Generate images from a text prompt.
        :param text: a string of text
        :param number_of_predictions: a number of images to generate
        :param print_time: whether to print the time taken to generate the images
        :return: a list of images the length of number_of_predictions
        """
        timeit_start = timer()

        tokenized_prompt = self.tokenize(text)

        images = []

        seed = random.randint(0, 2 ** 32 - 1)
        key = jax.random.PRNGKey(seed)

        for i in range(max(number_of_predictions // jax.device_count(), 1)):

            # get a new key
            key, subkey = jax.random.split(key)

            # generate images
            encoded_images = self.p_generate(
                tokenized_prompt,
                shard_prng_key(subkey),
                self.dalle_params,
                self.gen_top_k,
                self.gen_top_p,
                self.temperature,
                self.cond_scale,
            )

            # remove BOS
            encoded_images = encoded_images.sequences[..., 1:]

            # decode images
            decoded_images = self.p_decode(encoded_images, self.vqgan_params)
            decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
            for decoded_img in decoded_images:
                img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
                images.append(img)

        timeit_end = timer()

        if print_time:
            t = (timeit_end-timeit_start)/number_of_predictions
            print(f"Avg. time per image generated: {t:.2f} secs")

        return images
