from .wanhumo_conditioning import WanHuMo_Conditioning
from .wanhumo_sampler import WanHuMo_Sampler


NODE_CLASS_MAPPINGS = {
    "WanHuMo_Conditioning": WanHuMo_Conditioning,
    "WanHuMo_Sampler": WanHuMo_Sampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanHuMo_Conditioning": "WanHuMo Conditioning (w/ Scales)",
    "WanHuMo_Sampler": "WanHuMo Sampler (Multi-CFG)",
}


__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
