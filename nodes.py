
import os

import numpy as np
import torch
import logging

from .modules.image_util import pil2tensor, tensor2pil
from .modules.load_util import load_flux_fill_nf4
from folder_paths import models_dir

import comfy.model_management as mm
import comfy.utils

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
                def callback_on_step_end1(self, i, t, callback_kwargs):
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


NODE_CLASS_MAPPINGS = {
    "Flux Inpainting": FluxNF4Inpainting
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux Inpainting": "Flux Inpainting"
}
