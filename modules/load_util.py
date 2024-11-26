import torch
from diffusers import DiffusionPipeline
from transformers import T5EncoderModel

from .transformer_override import FluxTransformer2DModel

from .pipeline_flux_fill import FluxFillPipeline

dtype = torch.bfloat16

def load_flux_fill_nf4(flux_dir: str, flux_nf4_dir: str, four_bit=False):
    """ load flux fill nf4 """
    orig_pipeline = DiffusionPipeline.from_pretrained(flux_dir, transformer=None, torch_dtype=dtype)
    
    if four_bit:
        print("Using four bit.")
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_nf4_dir, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        text_encoder_2 = T5EncoderModel.from_pretrained(
            flux_nf4_dir, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        )
        pipeline = FluxFillPipeline.from_pipe(
            orig_pipeline, transformer=transformer, text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16
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