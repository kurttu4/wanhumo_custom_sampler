# Файл: custom_nodes/wanhumo_nodes/wanhumo_conditioning.py
import torch
import comfy.model_management
import comfy.utils
import nodes
import node_helpers

AUDIO_ENCODER_OUTPUT_TYPE = "AUDIO_ENCODER_OUTPUT" 

class WanHuMo_Conditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "length": ("INT", {"default": 97, "min": 1, "max": 2048, "step": 4, "tooltip": "Кількість кадрів відео."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "scale_a": ("FLOAT", {"default": 5.5, "min": 0.1, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "Audio Guidance Scale (scale_a)"}),
                "scale_t": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "Text Guidance Scale (scale_t)"}),
            },
            "optional": {
                "audio_encoder_output": (AUDIO_ENCODER_OUTPUT_TYPE, {"optional": True}),
                "ref_image": ("IMAGE", {"optional": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/wanhumo"

    def execute(self, positive, negative, vae, width, height, length, batch_size, scale_a, scale_t, ref_image=None, audio_encoder_output=None):
        
        latent_t = ((length - 1) // 4) + 1
        latent = torch.zeros([batch_size, 16, latent_t, height // 8, width // 8], device=comfy.model_management.intermediate_device())

        if ref_image is not None:
            ref_image = comfy.utils.common_upscale(ref_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            ref_latent = vae.encode(ref_image[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [torch.zeros_like(ref_latent)]}, append=True)
        else:
            zero_latent = torch.zeros([batch_size, 16, 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [zero_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [zero_latent]}, append=True)

        if audio_encoder_output is not None:
            # Тут ваш складний код для обробки аудіо
            audio_emb = torch.stack(audio_encoder_output["encoded_audio_all_layers"], dim=2)
            audio_len = audio_encoder_output["audio_samples"] // 640
            audio_emb = audio_emb[:, :audio_len * 2]

            feat0 = linear_interpolation(audio_emb[:, :, 0: 8].mean(dim=2), 50, 25)
            feat1 = linear_interpolation(audio_emb[:, :, 8: 16].mean(dim=2), 50, 25)
            feat2 = linear_interpolation(audio_emb[:, :, 16: 24].mean(dim=2), 50, 25)
            feat3 = linear_interpolation(audio_emb[:, :, 24: 32].mean(dim=2), 50, 25)
            feat4 = linear_interpolation(audio_emb[:, :, 32], 50, 25)
            audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]  # [T, 5, 1280]
            audio_emb, _ = get_audio_emb_window(audio_emb, length, frame0_idx=0)

            audio_emb = audio_emb.unsqueeze(0)
            audio_emb_neg = torch.zeros_like(audio_emb)
            
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": audio_emb})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": audio_emb_neg})
        else:
            zero_audio = torch.zeros([batch_size, latent_t + 1, 8, 5, 1280], device=comfy.model_management.intermediate_device())
            positive = node_helpers.conditioning_set_values(positive, {"audio_embed": zero_audio})
            negative = node_helpers.conditioning_set_values(negative, {"audio_embed": zero_audio})
            
        # ДОДАВАННЯ scale_a та scale_t
        positive = node_helpers.conditioning_set_values(positive, {"scale_a": [scale_a], "scale_t": [scale_t]}, append=True)
        negative = node_helpers.conditioning_set_values(negative, {"scale_a": [scale_a], "scale_t": [scale_t]}, append=True)
        
        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)

# ДОПОМІЖНІ ФУНКЦІЇ (як у вашому коді)
def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = torch.nn.functional.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)

def get_audio_emb_window(audio_emb, frame_num, frame0_idx, audio_shift=2):
    zero_audio_embed = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    zero_audio_embed_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    iter_ = 1 + (frame_num - 1) // 4
    audio_emb_wind = []
    for lt_i in range(iter_):
        if lt_i == 0:
            st = frame0_idx + lt_i - 2
            ed = frame0_idx + lt_i + 3
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
            wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)
        else:
            st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
            ed = frame0_idx + 1 + 4 * lt_i + audio_shift
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
        audio_emb_wind.append(wind_feat)
    audio_emb_wind = torch.stack(audio_emb_wind, dim=0)

    return audio_emb_wind, ed - audio_shift