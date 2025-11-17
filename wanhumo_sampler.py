# custom_nodes/wanhumo_custom_sampler/wanhumo_sampler.py
import torch
import os
WANHUMO_DEBUG = False

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
        def _move_obj(o):
            if isinstance(o, torch.Tensor):
                return o.to(device)
            if isinstance(o, list):
                return [_move_obj(x) for x in o]
            if isinstance(o, tuple):
                return tuple(_move_obj(x) for x in o)
            if isinstance(o, dict):
                return {kk: _move_obj(vv) for kk, vv in o.items()}
            return o

        return LatentWrapper({k: _move_obj(v) for k, v in self.items()})

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")


class SimpleLatent(LatentWrapper):
    """
    Лёгкая обёртка над LatentWrapper для совместимости с существующим кодом.
    Теперь SimpleLatent наследует правильное поведение (to/cpu/cuda/copy/shape).
    """
    def __init__(self, latent_dict):
        super().__init__(latent_dict)
        self.is_nested = False 


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
        # 
        try:
            self.scale_a = float(scale_a)
        except Exception:
            self.scale_a = scale_a
        try:
            self.scale_t = float(scale_t)
        except Exception:
            self.scale_t = scale_t
        self.step_change = step_change

        env_debug = False
        try:
            env_debug = bool(int(os.environ.get("WANHUMO_DEBUG", "0")))
        except Exception:
            env_debug = False
        self._debug = bool(WANHUMO_DEBUG) or env_debug
        self._debug_count = 0
        self._debug_max = 4  
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
                extra_params = cond_list[0][1] if len(cond_list[0]) > 1 and isinstance(cond_list[0][1], dict) else {}
                uncond_params = self.c_neg[0][1] if len(self.c_neg[0]) > 1 and isinstance(self.c_neg[0][1], dict) else {}
                return self.inner_model.apply_model(
                    x,
                    timestep,
                    c_cond=tokens,
                    c_uncond=self.c_neg[0][0],
                    cond_concat=extra_params.get("pooled_output", None),
                    uncond_concat=uncond_params.get("pooled_output", None),
                )

            pos_tia_out = apply_with_cond(self.c_tia, x, timestep)
            pos_ti_out = apply_with_cond(self.c_ti, x, timestep)

            # Robust handling of timestep which can be: tensor scalar, float, list/tuple of tensor/float
            t0 = timestep
            try:
                # unpack single-element list/tuple
                if isinstance(t0, (list, tuple)) and len(t0) == 1:
                    t0 = t0[0]
            except Exception:
                pass
            if isinstance(t0, torch.Tensor):
                current_t_raw = float(t0.item())
            else:
                current_t_raw = float(t0)
            num_ts = getattr(self.inner_model, "num_timesteps", None) or getattr(self.inner_model, "num_train_timesteps", None) or 1000
            current_t_val = int(current_t_raw * num_ts) if current_t_raw <= 1.0 else int(current_t_raw)

            current_neg_cond = self.c_neg if current_t_val > self.step_change else self.c_neg_null
            neg_out = apply_with_cond(current_neg_cond, x, timestep)

            # Debug: show basic diagnostics so user can confirm scale_* имеет эффект.
            if getattr(self.inner_model, "_wanhumo_debug", False) or self._debug:
                if self._debug_count < self._debug_max:
                    
                    try:
                        d_tia_ti = float((pos_tia_out - pos_ti_out).pow(2).mean().sqrt().item())
                        d_ti_neg = float((pos_ti_out - neg_out).pow(2).mean().sqrt().item())
                    except Exception:
                        d_tia_ti = None
                        d_ti_neg = None
                    print(f"[WanHuMo DEBUG] raw_t={current_t_raw:.6f} val={current_t_val} step_change={self.step_change} "
                          f"scale_a={self.scale_a} scale_t={self.scale_t} d(tia-ti)={d_tia_ti} d(ti-neg)={d_ti_neg}")
                    self._debug_count += 1

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

        
        try:
            effective_scale_a = float(effective_scale_a)
        except Exception:
            pass
        try:
            effective_scale_t = float(effective_scale_t)
        except Exception:
            pass
       
        try:
            effective_scale_a = max(0.0, min(8.0, effective_scale_a))
            effective_scale_t = max(0.0, min(8.0, effective_scale_t))
        except Exception:
            pass
                
        cached_attr = "_wanhumo_wrapper"
        existing = getattr(model, cached_attr, None)
        if existing is None or not isinstance(existing, WanHuMo_Model_Wrapper):
            wrapped_model = WanHuMo_Model_Wrapper(
                model,
                c_tia, c_ti, c_neg, c_neg_null, effective_scale_a, effective_scale_t, step_change
            )
            setattr(model, cached_attr, wrapped_model)
        else:
            wrapped_model = existing
            wrapped_model.c_tia = c_tia
            wrapped_model.c_ti = c_ti
            wrapped_model.c_neg = c_neg
            wrapped_model.c_neg_null = c_neg_null
            wrapped_model.scale_a = effective_scale_a
            wrapped_model.scale_t = effective_scale_t
            wrapped_model.step_change = step_change
            wrapped_model.inner_model = model
        
        # --- NEW: ensure model is placed on same device as latent for sampling ---
        def _module_device(module):
            # try to get device from parameters or buffers
            try:
                for p in module.parameters():
                    return p.device
            except Exception:
                pass
            try:
                for b in module.buffers():
                    return b.device
            except Exception:
                pass
            return None

        # determine module to inspect/move (prefer wrapper.model, fall back to inner_model)
        model_module = getattr(wrapped_model, "model", None) or getattr(wrapped_model, "inner_model", None)
        orig_dev = _module_device(model_module)  # may be None
        target_dev = latent.device if isinstance(latent, torch.Tensor) else None

        moved_model = False
        if target_dev is not None and orig_dev is not None and orig_dev != target_dev:
            try:
                # try to move the original model object (prefer inner_model.to)
                if hasattr(wrapped_model, "inner_model") and hasattr(wrapped_model.inner_model, "to"):
                    wrapped_model.inner_model.to(target_dev)
                elif hasattr(model_module, "to"):
                    model_module.to(target_dev)
                moved_model = True
                # free CUDA cache to reduce fragmentation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                moved_model = False

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

        # --- NEW: return model to original device if we moved it ---
        if moved_model:
            try:
                # prefer inner_model if present
                target_module = getattr(wrapped_model, "inner_model", None) or model_module
                # If original device unknown or CPU-like -> offload explicitly to CPU
                orig_is_cpu = False
                try:
                    orig_is_cpu = orig_dev is None or str(orig_dev).lower().startswith("cpu")
                except Exception:
                    orig_is_cpu = False

                if target_module is not None:
                    try:
                        if orig_is_cpu:
                            # try .cpu() first for safer offload
                            if hasattr(target_module, "cpu"):
                                target_module.cpu()
                            elif hasattr(target_module, "to"):
                                target_module.to("cpu")
                        else:
                            if hasattr(target_module, "to"):
                                target_module.to(orig_dev)
                    except Exception:
                        # best-effort, ignore on failure
                        pass
                # always try to free CUDA memory after moves
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # 6. Build output
        out = {}
        if isinstance(latent_image, dict):
            out.update(latent_image)
            
        out["samples"] = samples
        return (out, )
