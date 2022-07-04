import jax
import jax.numpy as jnp

from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel


DALLE_MODEL_PATH = "downloaded_models/dalle-mega/"
VQGAN_MODEL_PATH = "downloaded_models/vqgan_imagenet_f16_16384"

class DALLEMini:
    def __init__(self):
        print("Loading DALLE model...")
        dalle_model, dalle_params = DalleBart.from_pretrained(
            DALLE_MODEL_PATH, dtype=jnp.float16, _do_init=False
        )
        print("Loaded DALLE model.")

        print("Loading VQGAN model...")
        # Load VQGAN
        vqgan_model, vqgan_params = VQModel.from_pretrained(
            VQGAN_MODEL_PATH, _do_init=False
        )
        print("Loaded VQGAN model.")