import os
import torch
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel

from .autoencoder_kl import AutoencoderKL

from .transformer_override import FluxTransformer2DModel

from .pipeline_flux_fill import FluxFillPipeline

dtype = torch.bfloat16

def load_flux_fill_nf4(flux_dir: str, flux_nf4_dir: str, four_bit=False):
    """ load flux fill nf4 """
    exclude_sub_model = ["transformer", "text_encoder_2"] if four_bit else ["transformer"]
    exclude_sub_model_dict = {model: None for model in exclude_sub_model}
    orig_pipeline = FluxFillPipeline.from_pretrained(flux_dir, torch_dtype=dtype, **exclude_sub_model_dict)
    if four_bit:
        print("Using four bit.")
        vae = AutoencoderKL.from_pretrained(os.path.join(flux_dir, "vae"), torch_dtype=dtype)

        transformer = FluxTransformer2DModel.from_pretrained(
            flux_nf4_dir, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            flux_nf4_dir, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        )
        pipeline = FluxFillPipeline.from_pipe(
            orig_pipeline,
            vae=vae,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_dir,
            subfolder="transformer",
            revision="refs/pr/4",
            torch_dtype=torch.bfloat16,
        )
        pipeline = FluxFillPipeline.from_pipe(orig_pipeline, transformer=transformer, torch_dtype=torch.bfloat16)
    return pipeline