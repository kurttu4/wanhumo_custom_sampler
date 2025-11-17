# Файл: custom_nodes/wanhumo_custom_sampler/wanhumo_sampler.py
import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview


class LatentWrapper(dict):
    """Обертка для dict, которая имеет атрибут is_nested и совместимые методы."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_nested = False

    @property
    def shape(self):
        # Возвращаем shape тензора samples, если он есть
        samples = self.get("samples", None)
        if hasattr(samples, "shape"):
            return samples.shape
        return None

    def copy(self):
        # Возвращаем ещё одну обёртку поверх копии словаря
        return LatentWrapper(dict(self))

    def to(self, device):
        # Перемещаем все tensor-значения на указанный device и возвращаем новую обёртку
        new = {}
        for k, v in self.items():
            try:
                if isinstance(v, torch.Tensor):
                    new[k] = v.to(device)
                elif isinstance(v, list):
                    # Если список тензоров (например reference_latents), переносим элементы
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
        # Потрібно, оскільки це викликається безпосередньо.
        return self.latent_dict["samples"].shape

    def copy(self):
        # Також потрібен.
        return SimpleLatent(self.latent_dict.copy())
    
    def to(self, device):
        """Переміщення latent на пристрій."""
        new_dict = {}
        for key, value in self.latent_dict.items():
            if isinstance(value, torch.Tensor):
                new_dict[key] = value.to(device)
            else:
                new_dict[key] = value
        return SimpleLatent(new_dict)
    
    def cpu(self):
        """Переміщення на CPU."""
        return self.to("cpu")
    
    def cuda(self):
        """Переміщення на GPU."""
        return self.to("cuda")


# --- КЛАС-ОБГОРТКА МОДЕЛІ (WanHuMo_Model_Wrapper) ---
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


# Клас WanHuMo_Sampler
class WanHuMo_Sampler:
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
                "step_change": ("INT", {"default": 980, "min": 0, "max": 1000, "tooltip": "Timestep для перемикання CFG WanHuMo."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/WanHuMo"

    def sample(self, model, seed, steps, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, step_change=980):
        
        # Створення SimpleLatent-ОБ'ЄКТА для зручного доступу
        if isinstance(latent_image, dict) and "samples" in latent_image:
            latent_obj = SimpleLatent(latent_image)
        else:
            latent_obj = latent_image

        # 1. Підготовка базових даних
        latent = latent_obj["samples"]
        latent = comfy.sample.fix_empty_latent_channels(model, latent)
        batch_inds = latent_obj.get("batch_index")
        noise = comfy.sample.prepare_noise(latent, seed, batch_inds)
        
        # 2. Підготовка 4-х типів кондиціонування
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
        
        # Выбираем единый источник для scale_a / scale_t:
        # если wanhumo_conditioning (в c_tia) содержит эти параметры — используем их,
        # иначе используем жёстко заданные дефолты.
        DEFAULT_SCALE_A = 5.5
        DEFAULT_SCALE_T = 5.0
        if isinstance(c_tia, list) and len(c_tia) > 0 and isinstance(c_tia[0][1], dict):
            cond_params = c_tia[0][1]
            effective_scale_a = cond_params.get("scale_a", DEFAULT_SCALE_A)
            effective_scale_t = cond_params.get("scale_t", DEFAULT_SCALE_T)
        else:
            effective_scale_a = DEFAULT_SCALE_A
            effective_scale_t = DEFAULT_SCALE_T
        
        # 3. СТВОРЕННЯ ОБГОРТКИ МОДЕЛІ
        wrapped_model = WanHuMo_Model_Wrapper(
            model, 
            c_tia, c_ti, c_neg, c_neg_null, effective_scale_a, effective_scale_t, step_change
        )
        
        # 4. Обробка noise_mask та callback
        noise_mask = None
        if isinstance(latent_image, dict):
            noise_mask = latent_image.get("noise_mask", None)
        callback = latent_preview.prepare_callback(wrapped_model, steps)
        
        # 5. ВИКЛИК ГОЛОВНОЇ ФУНКЦІЇ СЕМПЛІНГУ
        # (не передаём обёртку — передаём собственно тензор latent)
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
            latent,       # Передача самóго тензора latent
            denoise=denoise, 
            disable_noise=False, 
            force_full_denoise=False, 
            noise_mask=noise_mask, 
            callback=callback, 
            seed=seed,
        )

        # 6. Формування виходу
        out = {}
        if isinstance(latent_image, dict):
            out.update(latent_image)
            
        out["samples"] = samples
        return (out, )