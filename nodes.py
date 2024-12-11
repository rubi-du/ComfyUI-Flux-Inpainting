
import os

import numpy as np
import torch
import logging

from .modules.image_util import pil2tensor, tensor2pil
from .modules.load_util import load_flux_fill_nf4, load_simple_flux_fill_nf4, load_vae
from folder_paths import models_dir, get_filename_list, get_full_path_or_raise

import comfy.model_management as mm
import comfy.utils

dir_path = os.path.dirname(os.path.abspath(__file__))
FLUX_FILL_DIR_PATH = os.path.join(dir_path, "config", "fill")

_pipeline = None

class FluxNF4Inpainting:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 60, "step": 1}),
                "cached": ("BOOLEAN", {"default": False}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 30.0, "step": 0.1}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpainting"
    CATEGORY = "Inpainting"
        
    def inpainting(self,
                    prompt,
                    image,
                    mask,
                    num_inference_steps,
                    cached,
                    guidance_scale
                    ):
        mm.unload_all_models()
        
        global _pipeline
        
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        mask = tensor2pil(mask[0])
        
        if image.dim() == 2:
            image = torch.unsqueeze(image, 0)
        image = tensor2pil(image[0])
        
        width = (image.width // 16) * 16
        height = (image.height // 16) * 16
        
        image.resize((width, height))
        mask.resize((width, height))
        
        try:
            pipeline = _pipeline
            logging.info("Loading Flux NF4 Inpainting")
            if not cached or pipeline is None:
                pbar1 = comfy.utils.ProgressBar(4)
                def callback_on_step_end1(i):
                    pbar1.update(1)
                    # hack to prevent crash
                    return {}
                flux_dir = os.path.join(models_dir, "FLUX.1-Fill-dev")
                if not os.path.isdir(flux_dir):
                    flux_dir = "black-forest-labs/FLUX.1-Fill-dev"
                
                flux_nf4_dir = os.path.join(models_dir, "FLUX.1-Fill-dev-nf4")
                if not os.path.isdir(flux_nf4_dir):
                    flux_nf4_dir = "sayakpaul/FLUX.1-Fill-dev-nf4"
                _pipeline = load_flux_fill_nf4(
                    flux_dir=flux_dir,
                    flux_nf4_dir=flux_nf4_dir,
                    four_bit=True,
                    step_call_back=callback_on_step_end1
                )
                logging.info("Flux NF4 Inpainting loaded")
                _pipeline.enable_model_cpu_offload()
                logging.info("Flux NF4 Inpainting enabled model cpu offload")
                pipeline = _pipeline
            
            logging.info("Running Flux NF4 Inpainting")
            pbar2 = comfy.utils.ProgressBar(num_inference_steps)
            def callback_on_step_end2(self, i, t, callback_kwargs):
                pbar2.update(1)
                # hack to prevent crash
                return {}
            res = pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                callback_on_step_end=callback_on_step_end2,
            )
            logging.info("Flux NF4 Inpainting finished")
            
            if not cached:
                del pipeline
                del _pipeline
                _pipeline = None
                
            mm.soft_empty_cache()
                
            processed_images = []
            for image in res.images:
                image_tensor = pil2tensor(image)
                processed_images.append(image_tensor)
            
            res_images = torch.cat(processed_images, dim=0)
        except torch.cuda.OutOfMemoryError as e:
            mm.free_memory(mm.get_total_memory(mm.get_torch_device()), mm.get_torch_device())
            mm.soft_empty_cache()
            raise e
        return (res_images,)

class FluxSimpleInpainting:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 60, "step": 1}),
                "cached": ("BOOLEAN", {"default": False}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 30.0, "step": 0.1}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpainting"
    CATEGORY = "Inpainting"
    def inpainting(self,
                   conditioning,
                   image,
                   mask,
                   num_inference_steps,
                   cached,
                   guidance_scale,
                   ):
        mm.unload_all_models()
        
        global _pipeline
        
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        mask = tensor2pil(mask[0])
        
        if image.dim() == 2:
            image = torch.unsqueeze(image, 0)
        image = tensor2pil(image[0])
        
        width = (image.width // 16) * 16
        height = (image.height // 16) * 16
        
        image.resize((width, height))
        mask.resize((width, height))
        
        cond_from: torch.Tensor = conditioning[0][0]
        pooled_output_from: torch.Tensor = conditioning[0][1].get("pooled_output", None)
        cond_from = cond_from.to(dtype=torch.bfloat16)
        pooled_output_from = pooled_output_from.to(dtype=torch.bfloat16)
        try:
            pipeline = _pipeline
            logging.info("Loading Flux NF4 Inpainting")
            if not cached or pipeline is None:
                pbar1 = comfy.utils.ProgressBar(2)
                def callback_on_step_end1(i):
                    pbar1.update(1)
                    # hack to prevent crash
                    return {}
                flux_dir = os.path.join(models_dir, "FLUX.1-Fill-dev")
                if not os.path.isdir(flux_dir):
                    flux_dir = "black-forest-labs/FLUX.1-Fill-dev"
                flux_nf4_dir = os.path.join(models_dir, "FLUX.1-Fill-dev-nf4")
                if not os.path.isdir(flux_nf4_dir):
                    flux_nf4_dir = "sayakpaul/FLUX.1-Fill-dev-nf4"
                _pipeline = load_simple_flux_fill_nf4(
                    flux_dir=flux_dir,
                    flux_nf4_dir=flux_nf4_dir,
                    four_bit=True,
                    step_call_back=callback_on_step_end1,
                )
                logging.info("Flux NF4 Inpainting loaded")
                _pipeline.enable_model_cpu_offload()
                logging.info("Flux NF4 Inpainting enabled model cpu offload")
                pipeline = _pipeline
            
            logging.info("Running Flux NF4 Inpainting")
            pbar2 = comfy.utils.ProgressBar(num_inference_steps)
            def callback_on_step_end2(self, i, t, callback_kwargs):
                pbar2.update(1)
                # hack to prevent crash
                return {}
            res = pipeline(
                prompt_embeds=cond_from,
                pooled_prompt_embeds=pooled_output_from,
                image=image,
                mask_image=mask,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                callback_on_step_end=callback_on_step_end2,
            )
            logging.info("Flux NF4 Inpainting finished")
            
            if not cached:
                del pipeline
                del _pipeline
                _pipeline = None
                
            mm.soft_empty_cache()
            
            processed_images = []
            for image in res.images:
                image_tensor = pil2tensor(image)
                processed_images.append(image_tensor)
            
            res_images = torch.cat(processed_images, dim=0)
        except torch.cuda.OutOfMemoryError as e:
            mm.free_memory(mm.get_total_memory(mm.get_torch_device()), mm.get_torch_device())
            mm.soft_empty_cache()
            raise e
        return (res_images,)
    
class FluxVaeLoader:
    def __init__(self):
        pass
    
    @staticmethod
    def vae_list():
        return get_filename_list("vae")
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (s.vae_list(), )}}
    RETURN_TYPES = ("AUTOENCODER",)
    RETURN_NAMES = ("vae",)
    CATEGORY = "loaders"
    FUNCTION = "load_vae"
    
    def load_vae(self, vae_name):
        vae_path = get_full_path_or_raise("vae", vae_name)
        vae = load_vae(vae_path)
        return (vae, )
    
class FluxTransformerInpainting:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "num_inference_steps": ("INT", {"default": 50, "min": 10, "max": 60, "step": 1}),
                "cached": ("BOOLEAN", {"default": False}),
                "guidance_scale": ("FLOAT", {"default": 30.0, "min": 0.1, "max": 30.0, "step": 0.1}),
                "vae": ("AUTOENCODER",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpainting"
    CATEGORY = "Inpainting"
    def inpainting(self,
                    conditioning,
                    image,
                    mask,
                    num_inference_steps,
                    cached,
                    guidance_scale,
                    vae
                    ):
        mm.unload_all_models()
        
        global _pipeline
        
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        mask = tensor2pil(mask[0])
        
        if image.dim() == 2:
            image = torch.unsqueeze(image, 0)
        image = tensor2pil(image[0])
        
        width = (image.width // 16) * 16
        height = (image.height // 16) * 16
        
        image.resize((width, height))
        mask.resize((width, height))
        
        cond_from: torch.Tensor = conditioning[0][0]
        pooled_output_from: torch.Tensor = conditioning[0][1].get("pooled_output", None)
        cond_from = cond_from.to(dtype=torch.bfloat16)
        pooled_output_from = pooled_output_from.to(dtype=torch.bfloat16)
        try:
            pipeline = _pipeline
            logging.info("Loading Flux NF4 Inpainting")
            if not cached or pipeline is None:
                pbar1 = comfy.utils.ProgressBar(2)
                def callback_on_step_end1(i):
                    pbar1.update(1)
                    # hack to prevent crash
                    return {}
                flux_nf4_dir = os.path.join(models_dir, "FLUX.1-Fill-dev-nf4")
                if not os.path.isdir(flux_nf4_dir):
                    flux_nf4_dir = "sayakpaul/FLUX.1-Fill-dev-nf4"
                _pipeline = load_simple_flux_fill_nf4(
                    flux_dir=FLUX_FILL_DIR_PATH,
                    flux_nf4_dir=flux_nf4_dir,
                    four_bit=True,
                    step_call_back=callback_on_step_end1,
                    vae=vae,
                )
                logging.info("Flux NF4 Inpainting loaded")
                _pipeline.enable_model_cpu_offload()
                logging.info("Flux NF4 Inpainting enabled model cpu offload")
                pipeline = _pipeline
            
            logging.info("Running Flux NF4 Inpainting")
            pbar2 = comfy.utils.ProgressBar(num_inference_steps)
            def callback_on_step_end2(self, i, t, callback_kwargs):
                pbar2.update(1)
                # hack to prevent crash
                return {}
            res = pipeline(
                prompt_embeds=cond_from,
                pooled_prompt_embeds=pooled_output_from,
                image=image,
                mask_image=mask,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                callback_on_step_end=callback_on_step_end2,
            )
            logging.info("Flux NF4 Inpainting finished")
            
            if not cached:
                del pipeline
                del _pipeline
                _pipeline = None
                
            mm.soft_empty_cache()
            
            processed_images = []
            for image in res.images:
                image_tensor = pil2tensor(image)
                processed_images.append(image_tensor)
            
            res_images = torch.cat(processed_images, dim=0)
        except torch.cuda.OutOfMemoryError as e:
            mm.free_memory(mm.get_total_memory(mm.get_torch_device()), mm.get_torch_device())
            mm.soft_empty_cache()
            raise e
        return (res_images,)
    
NODE_CLASS_MAPPINGS = {
    "Flux Inpainting": FluxNF4Inpainting, ## NOTE: should be renamed to FluxInpainting
    "FluxInpainting": FluxNF4Inpainting,
    "FluxTransformerInpainting": FluxTransformerInpainting,
    "FluxSimpleInpainting": FluxSimpleInpainting,
    "FluxVAELoader": FluxVaeLoader
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxInpainting": "Flux Inpainting",
    "FluxSimpleInpainting": "Flux Simple Inpainting",
    "FluxTransformerInpainting": "Flux Transformer Inpainting",
    "FluxVAELoader": "Load Flux VAE"
}
