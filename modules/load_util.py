import os
import torch
from transformers import T5EncoderModel


from .transformer_override import FluxTransformer2DModel

from .pipeline_flux_fill import FluxFillPipeline

dtype = torch.bfloat16

def load_flux_fill_nf4(flux_dir: str, flux_nf4_dir: str, four_bit=False, step_call_back=None):
    """ load flux fill nf4 """
    exclude_sub_model = ["transformer", "text_encoder_2"] if four_bit else ["transformer"]
    exclude_sub_model_dict = {model: None for model in exclude_sub_model}
    orig_pipeline = FluxFillPipeline.from_pretrained(flux_dir, torch_dtype=dtype, **exclude_sub_model_dict)
    if step_call_back:
        step_call_back(0)
    if four_bit:
        print("Using four bit.")
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_nf4_dir, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        if step_call_back:
            step_call_back(1)
        
        text_encoder_2 = T5EncoderModel.from_pretrained(
            flux_nf4_dir, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        )
        if step_call_back:
            step_call_back(2)
        
        pipeline = FluxFillPipeline.from_pipe(
            orig_pipeline,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=torch.bfloat16
        )
        if step_call_back:
            step_call_back(3)
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_dir,
            subfolder="transformer",
            revision="refs/pr/4",
            torch_dtype=torch.bfloat16,
        )
        pipeline = FluxFillPipeline.from_pipe(orig_pipeline, transformer=transformer, torch_dtype=torch.bfloat16)
    return pipeline