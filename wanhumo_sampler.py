# custom_nodes/wanhumo_custom_sampler/wanhumo_sampler.py
import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview


class LatentWrapper(dict):
    """A wrapper for dict that has an is_nested attribute and compatible methods."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_nested = False

    @property
    def shape(self):
        # Return the shape of the samples tensor, if it exists
        samples = self.get("samples", None)
        if hasattr(samples, "shape"):
            return samples.shape
        return None

    def copy(self):
        # Return another wrapper over a copy of the dictionary
        return LatentWrapper(dict(self))

    def to(self, device):
        # Move all tensor values ​​to the specified device and return the new wrapper
        new = {}
        for k, v in self.items():
            try:
                if isinstance(v, torch.Tensor):
                    new[k] = v.to(device)
                elif isinstance(v, list):
                    # If the list is tensors (for example reference_latents), move the elements
                    new_list = []
                    for item in v:
                        if isinstance(item, torch.Tensor):
                            new_list.append(item.to(device))
                        else:
                            new_list.append(item)
                    new[k] = new_list
                else:
                    new[k] = v
            except Exception:
                new[k] = v
        return LatentWrapper(new)

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")


class SimpleLatent:
    def __init__(self, latent_dict):
        self.latent_dict = latent_dict
        self.is_nested = False 

    def __getitem__(self, key):
        return self.latent_dict[key]

    def get(self, key, default=None):
        return self.latent_dict.get(key, default)
    
    @property
    def shape(self):
        # It is necessary that the fragments of this cry out in the middle.
        return self.latent_dict["samples"].shape

    def copy(self):
        return SimpleLatent(self.latent_dict.copy())
    
    def to(self, device):
        """Moving latent on device."""
        new_dict = {}
        for key, value in self.latent_dict.items():
            if isinstance(value, torch.Tensor):
                new_dict[key] = value.to(device)
            else:
                new_dict[key] = value
        return SimpleLatent(new_dict)
    
    def cpu(self):
        """Moving to CPU."""
        return self.to("cpu")
    
    def cuda(self):
        """Moving to GPU."""
        return self.to("cuda")


# --- MODEL WRAPPER CLASS (WanHuMo_Model_Wrapper) ---
class WanHuMo_Model_Wrapper:
    def __init__(self, original_model, c_tia, c_ti, c_neg, c_neg_null, scale_a, scale_t, step_change):
        
        self.inner_model = original_model 
        self.model = original_model.model
        self.load_device = getattr(original_model, 'load_device', None)
        self.offload_device = getattr(original_model, 'offload_device', None)
        self.model_type = getattr(original_model, 'model_type', getattr(original_model.model, 'model_type', None))
        self.c_tia = c_tia
        self.c_ti = c_ti
        self.c_neg = c_neg
        self.c_neg_null = c_neg_null
        self.scale_a = scale_a
        self.scale_t = scale_t
        self.step_change = step_change
        self._safe_attributes = {'inner_model', 'model', 'load_device', 'offload_device', 'model_type', 
                                 'c_tia', 'c_ti', 'c_neg', 'c_neg_null', 'scale_a', 'scale_t', 'step_change'}

    def __getattr__(self, name):
        if name in self._safe_attributes or name.startswith('_'):
            return object.__getattribute__(self, name)
        return getattr(self.inner_model, name)
        
    def apply_model(self, x, timestep, c_cond, c_uncond):
        with torch.no_grad():
            
            def apply_with_cond(cond_list, x, timestep):
                tokens = cond_list[0][0]
                extra_params = cond_list[0][1]
                return self.inner_model.apply_model(
                    x, 
                    timestep, 
                    c_cond=tokens, 
                    c_uncond=self.c_neg[0][0], 
                    cond_concat=extra_params.get("pooled_output", None), 
                    uncond_concat=self.c_neg[0][1].get("pooled_output", None)
                )

            pos_tia_out = apply_with_cond(self.c_tia, x, timestep)
            pos_ti_out = apply_with_cond(self.c_ti, x, timestep)
            current_t_val = timestep[0].item() 
            current_neg_cond = self.c_neg if current_t_val > self.step_change else self.c_neg_null
            neg_out = apply_with_cond(current_neg_cond, x, timestep)

            if current_t_val > self.step_change:
                noise_pred = self.scale_a * (pos_tia_out - pos_ti_out) + self.scale_t * (pos_ti_out - neg_out) + neg_out
            else:
                noise_pred = self.scale_a * (pos_tia_out - pos_ti_out) + (self.scale_t - 2.0) * (pos_ti_out - neg_out) + neg_out
            
            return noise_pred


# WanHuMo Sampler node
class WanHuMo_Sampler:
    # Added: friendly node name and description (shown in UI/documentation)
    NODE_NAME = "WanHuMo Sampler"
    NODE_DESCRIPTION = "WanHuMo specialized sampler: combines audio/text conditioning and manages CFG switching via step_change and scale_a/scale_t."
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,), 
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "step_change": ("INT", {"default": 980, "min": 0, "max": 1000,
                                       "tooltip": "Time-step index to switch WanHuMo CFG behavior (default 980). Determines the moment when the combination formula for scale_a/scale_t changes."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/WanHuMo"

    def sample(self, model, seed, steps, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, step_change=980):
        # Create a SimpleLatent object for convenient access if needed
        if isinstance(latent_image, dict) and "samples" in latent_image:
            latent_obj = SimpleLatent(latent_image)
        else:
            latent_obj = latent_image

        # 1. Prepare base data
        latent = latent_obj["samples"]
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        batch_inds = latent_obj.get("batch_index")
        noise = comfy.sample.prepare_noise(latent, seed, batch_inds)
        
        # 2. Prepare 4 types of conditioning
        c_tia = positive
        c_ti_data = positive[0][1].copy()
        if "audio_embed" in c_ti_data:
            c_ti_data["audio_embed"] = torch.zeros_like(c_ti_data["audio_embed"])
        c_ti = [[positive[0][0], c_ti_data]]
        c_neg = negative
        c_neg_null_data = negative[0][1].copy()
        if "reference_latents" in c_neg_null_data:
            ref_latents = c_neg_null_data["reference_latents"]
            if isinstance(ref_latents, list) and len(ref_latents) > 0:
                c_neg_null_data["reference_latents"] = [torch.zeros_like(ref_latents[0])]
        c_neg_null = [[negative[0][0], c_neg_null_data]]
        
        # Choose the source for scale_a / scale_t:
        # if wanhumo_conditioning (in c_tia) includes these params — use them,
        # otherwise fall back to defaults.
        DEFAULT_SCALE_A = 5.5
        DEFAULT_SCALE_T = 5.0
        if isinstance(c_tia, list) and len(c_tia) > 0 and isinstance(c_tia[0][1], dict):
            cond_params = c_tia[0][1]
            effective_scale_a = cond_params.get("scale_a", DEFAULT_SCALE_A)
            effective_scale_t = cond_params.get("scale_t", DEFAULT_SCALE_T)
        else:
            effective_scale_a = DEFAULT_SCALE_A
            effective_scale_t = DEFAULT_SCALE_T
        
        # 3. Create model wrapper
        # Try to reuse a previously created wrapper attached to the original model object.
        # Это позволяет избежать создания нового обёрточного объекта и повторной загрузки модели.
        cached_attr = "_wanhumo_wrapper"
        existing = getattr(model, cached_attr, None)
        if existing is None or not isinstance(existing, WanHuMo_Model_Wrapper):
            wrapped_model = WanHuMo_Model_Wrapper(
                model,
                c_tia, c_ti, c_neg, c_neg_null, effective_scale_a, effective_scale_t, step_change
            )
            setattr(model, cached_attr, wrapped_model)
        else:
            # Переиспользуем обёртку, обновляем поля кондишнинга/параметры (чтобы не создавать новый объект)
            wrapped_model = existing
            wrapped_model.c_tia = c_tia
            wrapped_model.c_ti = c_ti
            wrapped_model.c_neg = c_neg
            wrapped_model.c_neg_null = c_neg_null
            wrapped_model.scale_a = effective_scale_a
            wrapped_model.scale_t = effective_scale_t
            wrapped_model.step_change = step_change
            # убедимся, что inner_model не потерян (на случай реструктуризации)
            wrapped_model.inner_model = model
        
        # 4. Handle noise_mask and prepare callback
        noise_mask = None
        if isinstance(latent_image, dict):
            noise_mask = latent_image.get("noise_mask", None)
        callback = latent_preview.prepare_callback(wrapped_model, steps)
        
        # 5. Call main sampling function
        # (pass the tensor latent itself, not the wrapper)
        # latent_dict = LatentWrapper({"samples": latent})
        # if noise_mask is not None:
        #     latent_dict["noise_mask"] = noise_mask
        
        samples = comfy.sample.sample(
            wrapped_model, 
            noise, 
            steps, 
            1.0, # CFG = 1.0
            sampler_name, 
            scheduler, 
            c_tia, 
            c_neg, 
            latent,       # Transfer of the latent tensor itself
            denoise=denoise, 
            disable_noise=False, 
            force_full_denoise=False, 
            noise_mask=noise_mask, 
            callback=callback, 
            seed=seed,
        )

        # 6. Build output
        out = {}
        if isinstance(latent_image, dict):
            out.update(latent_image)
            
        out["samples"] = samples
        return (out, )
