import os
import logging
import torch
from transformers import T5EncoderModel

from .autoencoder_kl import AutoencoderKL
from .pipeline_flux_fill import FluxFillPipeline

try:
    from diffusers import FluxTransformer2DModel
except ImportError:
    from .transformer_override import FluxTransformer2DModel

try:
    from diffusers import GGUFQuantizationConfig
except ImportError:
    logging.warning("GGUFQuantizationConfig not found, please update diffusers >= 0.32.0")

dir_path = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR_PATH = os.path.join(os.path.dirname(dir_path), "config")

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
            flux_nf4_dir, subfolder="transformer", torch_dtype=dtype
        )
        if step_call_back:
            step_call_back(1)
        
        text_encoder_2 = T5EncoderModel.from_pretrained(
            flux_nf4_dir, subfolder="text_encoder_2", torch_dtype=dtype
        )
        if step_call_back:
            step_call_back(2)
        
        pipeline = FluxFillPipeline.from_pipe(
            orig_pipeline,
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=dtype
        )
        if step_call_back:
            step_call_back(3)
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_dir,
            subfolder="transformer",
            revision="refs/pr/4",
            torch_dtype=dtype,
        )
        pipeline = FluxFillPipeline.from_pipe(orig_pipeline, transformer=transformer, torch_dtype=dtype)
    return pipeline



def load_simple_flux_fill_nf4(flux_dir: str, flux_nf4_dir: str, four_bit=False, step_call_back=None, vae: AutoencoderKL = None):
    """ load flux fill nf4 """
    exclude_sub_model = ["transformer", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"] if four_bit else ["transformer"]
    exclude_sub_model_dict = {model: None for model in exclude_sub_model}
    
    kwargs = {}
    if vae is not None:
        exclude_sub_model_dict["vae"] = None
        kwargs = {
            "vae": vae.to(dtype=dtype)
        }
        
    if step_call_back:
        step_call_back(0)
    if four_bit:
        print("Using four bit.")
        orig_pipeline = FluxFillPipeline.from_pretrained(flux_dir, torch_dtype=dtype, **exclude_sub_model_dict)
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_nf4_dir, subfolder="transformer", torch_dtype=dtype
        )
        if step_call_back:
            step_call_back(1)
        
        pipeline = FluxFillPipeline.from_pipe(
            orig_pipeline,
            transformer=transformer,
            torch_dtype=dtype,
            **kwargs,
        )
        if step_call_back:
            step_call_back(2)
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            flux_dir,
            subfolder="transformer",
            revision="refs/pr/4",
        )
        pipeline = FluxFillPipeline.from_pipe(orig_pipeline, transformer=transformer, torch_dtype=dtype)
    return pipeline

def load_simple_flux_fill_gguf(flux_dir: str, flux_gguf_dir: str, step_call_back=None, vae: AutoencoderKL = None):
    """ load flux fill nf4 """
    exclude_sub_model = ["transformer", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"]
    exclude_sub_model_dict = {model: None for model in exclude_sub_model}
    
    kwargs = {}
    if vae is not None:
        exclude_sub_model_dict["vae"] = None
        kwargs = {
            "vae": vae.to(dtype=dtype)
        }
        
    if step_call_back:
        step_call_back(0)
    print("Using gguf.")
    orig_pipeline = FluxFillPipeline.from_pretrained(flux_dir, torch_dtype=dtype, **exclude_sub_model_dict)
    transformer_config_path = os.path.join(CONFIG_DIR_PATH, "fill", "transformer", 'config.json')
    
    transformer = FluxTransformer2DModel.from_single_file(
        flux_gguf_dir,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16,
        config=transformer_config_path,
        local_files_only=True
    )
    if step_call_back:
        step_call_back(1)
    
    pipeline = FluxFillPipeline.from_pipe(
        orig_pipeline,
        transformer=transformer,
        **kwargs,
    )
    if step_call_back:
        step_call_back(2)
    return pipeline

def load_vae(
    checkpoint_path: str,
):
    if not checkpoint_path.endswith("safetensors") and not checkpoint_path.endswith("sft"):
        raise ValueError(
            f"The checkpoint path must end with 'safetensors' or 'sft'. Got {checkpoint_path}."
        )
    vae_config_path = os.path.join(CONFIG_DIR_PATH, "vae", 'config.json')
    return AutoencoderKL.from_pretrained(
        checkpoint_path,
        local_config_path=vae_config_path,
        local_files_only=True,
    )