
import os

import numpy as np
import torch

from .modules.image_util import pil2tensor, tensor2pil
from .modules.load_util import load_flux_fill_nf4
from folder_paths import models_dir

def clear_memory():
    import gc
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

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
                    ):
        
        global _pipeline
        
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        mask = tensor2pil(mask[0])
        
        if image.dim() == 2:
            image = torch.unsqueeze(image, 0)
        image = tensor2pil(image[0])
        
        pipeline = _pipeline
        if not cached or pipeline is None:
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
            )
            _pipeline.enable_model_cpu_offload()
            pipeline = _pipeline
            
        res = pipeline(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
        )
        
        if not cached:
            del pipeline
            del _pipeline
            _pipeline = None
            
            clear_memory()
            
        processed_images = []
        for image in res.images:
            image = pil2tensor(image)
        
        res_images = torch.cat(processed_images, dim=0)
        return (res_images,)


NODE_CLASS_MAPPINGS = {
    "Flux Inpainting": FluxNF4Inpainting
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux Inpainting": "Flux Inpainting"
}

            
        
        
        