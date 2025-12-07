from typing import Any, Dict, List, Optional, Tuple, Union
import os

from insightface.app import FaceAnalysis
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    DDIMScheduler, 
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LCMScheduler
)
from diffusers.models import ImageProjection
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from .utils import get_faceid_embeds
from .dcg import decoupled_cfg_predict


def get_style_embeds(
    pipe,
    style_image,
    do_cfg=False,
    decoupled=False,
    dcg_type=1,
    batch_size=1,
    dtype=torch.float16,
):
    """Get CLIP image embeddings for style transfer via IP-Adapter"""
    from PIL import Image
    import numpy as np

    if isinstance(style_image, str):
        style_image = Image.open(style_image).convert("RGB")

    # Prepare image for CLIP
    clip_embeds = pipe.prepare_ip_adapter_image_embeds(
        [[style_image]],
        None,
        torch.device(pipe.device),
        batch_size,
        do_cfg,
        do_decoupled_cfg=decoupled,
        dcg_type=dcg_type
    )[0]

    return clip_embeds.to(dtype=dtype)


class DecoupledGuidancePipelineXL(StableDiffusionXLPipeline):
    """pipeline to be used together with IpAdapter, adds separate image and textual guidance"""
    def prepare_ip_adapter_image_embeds(
        self,
        ip_adapter_image,
        ip_adapter_image_embeds,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        do_decoupled_cfg,
        dcg_type=1,
    ):
        image_embeds = []
        if do_classifier_free_guidance:
            negative_image_embeds = []

        if ip_adapter_image_embeds is None:
            # Encode images from scratch
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != len(self.unet.encoder_hid_proj.image_projection_layers):
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(self.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
                )

            for single_ip_adapter_image, image_proj_layer in zip(
                ip_adapter_image, self.unet.encoder_hid_proj.image_projection_layers
            ):
                output_hidden_state = not isinstance(image_proj_layer, ImageProjection)
                single_image_embeds, single_negative_image_embeds = self.encode_image(
                    single_ip_adapter_image, device, 1, output_hidden_state
                )

                image_embeds.append(single_image_embeds[None, :])
                if do_classifier_free_guidance:
                    negative_image_embeds.append(single_negative_image_embeds[None, :])
        else:
            # Use pre-computed embeddings
            for single_image_embeds in ip_adapter_image_embeds:
                # Check if embeddings are already stacked for DCG
                # (batch size 3 for DCG type 1-3, batch size 4 for DCG type 4)
                is_dcg_batched = (
                    (dcg_type == 4 and single_image_embeds.shape[0] == 4) or
                    (dcg_type in [1, 2, 3] and single_image_embeds.shape[0] == 3)
                )
                if is_dcg_batched:
                    # Already properly stacked - use as-is
                    image_embeds.append(single_image_embeds)
                    if do_classifier_free_guidance:
                        # First batch is the negative/uncond
                        negative_image_embeds.append(single_image_embeds[0:1])
                else:
                    # Legacy behavior: split if chunked
                    if do_classifier_free_guidance and single_image_embeds.shape[0] >= 2:
                        single_negative_image_embeds, single_image_embeds = single_image_embeds.chunk(2)
                        negative_image_embeds.append(single_negative_image_embeds)
                    image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for i, single_image_embeds in enumerate(image_embeds):
            # Check if already properly batched for DCG
            is_dcg_batched = (
                (dcg_type == 4 and single_image_embeds.shape[0] == 4) or
                (dcg_type in [1, 2, 3] and single_image_embeds.shape[0] == 3)
            )
            if is_dcg_batched:
                # Already stacked for DCG - use directly
                single_image_embeds = single_image_embeds.to(device=device)
                ip_adapter_image_embeds.append(single_image_embeds)
                continue

            # Original batching logic for other cases
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            if do_classifier_free_guidance:
                single_negative_image_embeds = torch.cat([negative_image_embeds[i]] * num_images_per_prompt, dim=0)

                if not do_decoupled_cfg:
                    single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds], dim=0)
                elif dcg_type == 4:
                    # Dual adapter mode: 4 batches (fallback if not pre-stacked)
                    # [uncond, face_only, style_only, face+text]
                    single_image_embeds = torch.cat([
                        single_negative_image_embeds,  # uncond
                        single_image_embeds,           # face_only
                        single_image_embeds,           # style_only (same as face for now)
                        single_image_embeds,           # face+text
                    ], dim=0)
                else:
                    # DCG Type 1, 2, 3: 3 batches
                    embed = single_negative_image_embeds if dcg_type in [1, 2] else single_image_embeds
                    single_image_embeds = torch.cat([
                        single_negative_image_embeds,
                        embed,
                        single_image_embeds
                    ], dim=0)

            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds
    
    def preprocess_prompt_embeds(
        self,
        dcg_type,
        negative_prompt_embeds,
        prompt_embeds,
        negative_pooled_prompt_embeds,
        add_text_embeds,
        negative_add_time_ids,
        add_time_ids
    ):
        if dcg_type == 1:
            prompt_embeds = torch.cat([
                negative_prompt_embeds,
                prompt_embeds,
                prompt_embeds],
            dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids, add_time_ids], dim=0)
        elif dcg_type == 2:
            prompt_embeds = torch.cat([
                negative_prompt_embeds,
                prompt_embeds,
                negative_prompt_embeds],
            dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds, negative_pooled_prompt_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids, negative_add_time_ids], dim=0)
        elif dcg_type == 3:
            prompt_embeds = torch.cat([
                negative_prompt_embeds,
                negative_prompt_embeds,
                prompt_embeds,
            ], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, negative_add_time_ids, add_time_ids], dim=0)
        elif dcg_type == 4:
            # Dual adapter mode: 4 batches
            # [uncond, face_only (no text), style_only (no text), face+text]
            prompt_embeds = torch.cat([
                negative_prompt_embeds,   # uncond
                negative_prompt_embeds,   # face_only (no text prompt)
                negative_prompt_embeds,   # style_only (no text prompt)
                prompt_embeds,            # face+text
            ], dim=0)
            add_text_embeds = torch.cat([
                negative_pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                add_text_embeds
            ], dim=0)
            add_time_ids = torch.cat([
                negative_add_time_ids,
                negative_add_time_ids,
                negative_add_time_ids,
                add_time_ids
            ], dim=0)

        return prompt_embeds, add_text_embeds, add_time_ids

    @torch.no_grad()
    def execute(
        self,
        face_image,
        prompt,
        generator,
        pipe_kwargs,
        after_hook_fn=None,
        style_image=None,
        style_strength=0.3,
        dual_adapter_mode=False,
        negative_prompt=None,
        progress_callback=None,
    ):
        from PIL import Image

        dcg_type = pipe_kwargs["dcg_kwargs"]["dcg_type"]

        # Dual Adapter Mode v3: True Dual IP-Adapter Architecture
        #
        # Uses TWO separate IP-Adapters:
        #   1. IP-Adapter FaceID Plus v2: For face identity (InsightFace + CLIP)
        #   2. IP-Adapter Plus: For style transfer (general CLIP)
        #
        # Each adapter processes its own input independently:
        #   - FaceID adapter: face_image → InsightFace embedding → face identity
        #   - Style adapter: style_image → CLIP embedding → visual style
        #
        # This provides TRUE separation of face identity and style.
        # No blending, no cross-contamination.

        if dual_adapter_mode and style_image is not None and getattr(self, 'dual_adapter_enabled', False):
            from PIL import Image
            import cv2
            import numpy as np
            from insightface.utils import face_align

            # 1. Extract Face ID embedding directly (avoid prepare_ip_adapter_image_embeds)
            # This is needed because dual adapters expect 2 images, but we only have face image here
            if isinstance(face_image, str):
                face_pil = Image.open(face_image)
            else:
                face_pil = face_image
            image = cv2.cvtColor(np.asarray(face_pil), cv2.COLOR_BGR2RGB)
            faces = self.app.get(image)

            if len(faces) == 0:
                raise ValueError("No face detected in the input image")

            # Get InsightFace embedding
            face_embedding = torch.from_numpy(faces[0].normed_embedding)
            ref_images_embeds = face_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(0)

            # Build id_embeds with DCG batching
            dtype = self.unet.dtype if hasattr(self.unet, 'dtype') else torch.float16
            neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)

            if dcg_type == 4:
                id_embeds = torch.cat([
                    neg_ref_images_embeds,  # uncond
                    ref_images_embeds,      # face_only
                    ref_images_embeds,      # style_only (placeholder)
                    ref_images_embeds,      # combined
                ], dim=0).to(dtype=dtype, device=self.device)
            else:
                id_embeds = torch.cat([
                    neg_ref_images_embeds,
                    ref_images_embeds,
                    ref_images_embeds,
                ], dim=0).to(dtype=dtype, device=self.device)

            # Set Face CLIP embedding for FaceID adapter (first adapter)
            face_crop = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)
            face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            face_clip_embeds, neg_face_clip_embeds = self.encode_image(
                face_crop_pil, torch.device(self.device), 1, output_hidden_states=True
            )
            # ImageProjection expects 4D: [batch, num_images, seq_len, hidden_dim]
            face_clip_embeds = face_clip_embeds.unsqueeze(1)  # [1, 1, 257, 1280]
            neg_face_clip_embeds = neg_face_clip_embeds.unsqueeze(1)  # [1, 1, 257, 1280]
            if dcg_type == 4:
                face_clip_batched = torch.cat([
                    neg_face_clip_embeds, face_clip_embeds, face_clip_embeds, face_clip_embeds
                ], dim=0)  # [4, 1, 257, 1280]
            else:
                face_clip_batched = torch.cat([
                    neg_face_clip_embeds, face_clip_embeds, face_clip_embeds
                ], dim=0)  # [3, 1, 257, 1280]
            self.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = face_clip_batched.to(dtype=dtype)
            self.unet.encoder_hid_proj.image_projection_layers[0].shortcut = True

            # 2. Load style image for the Style adapter
            if isinstance(style_image, str):
                style_pil = Image.open(style_image).convert("RGB")
            else:
                style_pil = style_image

            # 3. Set adapter scales: [FaceID, Style]
            # style_strength directly controls the Style adapter's influence
            face_scale = self._config.get("ip_adapter_scale", 0.5)
            style_scale = style_strength  # Direct control
            self.set_ip_adapter_scale([face_scale, style_scale])

            print(f">>> [DEBUG] Dual IP-Adapter Mode (True Separation)")
            print(f">>> [DEBUG] FaceID scale: {face_scale}, Style scale: {style_scale}")
            print(f">>> [DEBUG] id_embeds shape: {id_embeds.shape}")

            # Create step callback for progress reporting
            total_steps = pipe_kwargs.get("num_inference_steps", 4)
            def step_callback(step_idx, t, latents):
                if progress_callback:
                    progress_callback(step_idx + 1, total_steps)

            # 4. Encode style image for the Style adapter
            # The Style adapter (IP-Adapter Plus) expects CLIP embeddings
            style_embeds = self._encode_style_for_adapter(style_pil, dcg_type)
            print(f">>> [DEBUG] style_embeds shape: {style_embeds.shape}")

            # 5. Generate with both adapters
            # ip_adapter_image_embeds: [FaceID embeds, Style embeds]
            # Each adapter processes its own embeddings independently
            res = self(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image_embeds=[id_embeds, style_embeds],
                num_images_per_prompt=1,
                generator=generator,
                # NO init_latents - pure txt2img to prevent structure bleeding
                callback=step_callback,
                callback_steps=1,
                **pipe_kwargs
            ).images[0]

        elif dual_adapter_mode and style_image is not None:
            # Fallback: dual_adapter_mode requested but pipeline doesn't have dual adapters
            # Use original CLIP blending approach
            print(">>> [WARNING] Dual adapter mode requested but pipeline loaded without dual adapters")
            print(">>> [WARNING] Falling back to CLIP blending mode")

            id_embeds, face_clip_embeds = get_faceid_embeds(
                self, self.app, face_image, "plus-v2",
                do_cfg=True, decoupled=True, dcg_type=dcg_type
            )
            if id_embeds is None:
                raise ValueError("No face detected in the input image")

            style_clip_embeds = self._get_style_clip_embeds(style_image, dcg_type=dcg_type)
            blended_clip = (1 - style_strength) * face_clip_embeds + style_strength * style_clip_embeds
            dtype = self.unet.dtype if hasattr(self.unet, 'dtype') else torch.float16
            self.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = blended_clip.to(dtype=dtype)

            total_steps = pipe_kwargs.get("num_inference_steps", 4)
            def step_callback(step_idx, t, latents):
                if progress_callback:
                    progress_callback(step_idx + 1, total_steps)

            res = self(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image_embeds=[id_embeds],
                num_images_per_prompt=1,
                generator=generator,
                callback=step_callback,
                callback_steps=1,
                **pipe_kwargs
            ).images[0]

        else:
            # Original mode: single adapter or img2img style transfer
            id_embeds, _ = get_faceid_embeds(
                self,
                self.app,
                face_image,
                "plus-v2",
                do_cfg=True,
                decoupled=True,
                dcg_type=dcg_type
            )

            print(f">>> [DEBUG] face_image: {face_image}")
            print(f">>> [DEBUG] id_embeds is None: {id_embeds is None}")
            if id_embeds is not None:
                print(f">>> [DEBUG] id_embeds shape: {id_embeds.shape}")

            # Prepare style latents for img2img approach
            init_latents = None
            if style_image is not None:
                height = pipe_kwargs.get("height", 1024)
                width = pipe_kwargs.get("width", 1024)
                # Use VAE's dtype for consistency (float32 on MPS, float16 on CUDA)
                vae_dtype = self.vae.dtype if hasattr(self.vae, 'dtype') else torch.float16
                init_latents = self._prepare_style_latents(
                    style_image, height, width, generator, vae_dtype
                )
                print(f">>> [DEBUG] style_latents shape: {init_latents.shape}, strength: {style_strength}")

            # Invert style_strength to denoising_strength:
            # High style_strength = preserve more style = lower denoising
            # Low style_strength = change more = higher denoising
            denoising = 1.0 - style_strength if init_latents is not None else 1.0

            # Create step callback for progress reporting
            total_steps = pipe_kwargs.get("num_inference_steps", 4)
            def step_callback(step_idx, t, latents):
                if progress_callback:
                    progress_callback(step_idx + 1, total_steps)

            res = self(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image_embeds=[id_embeds],
                num_images_per_prompt=1,
                generator=generator,
                init_latents=init_latents,
                denoising_strength=denoising,
                callback=step_callback,
                callback_steps=1,
                **pipe_kwargs
            ).images[0]

        if after_hook_fn is not None:
            after_hook_fn(self, res)

        return res

    def _encode_style_for_adapter(self, style_image, dcg_type=3):
        """Encode style image for the Style IP-Adapter (second adapter).

        This method encodes the style image using the CLIP encoder and formats
        the embeddings for use with the standard IP-Adapter Plus.

        Args:
            style_image: PIL Image for style reference
            dcg_type: DCG type for batch configuration

        Returns:
            Style embeddings formatted for the IP-Adapter
        """
        from PIL import Image

        # Resize to CLIP input size
        clip_image_size = self.image_encoder.config.image_size
        style_resized = style_image.resize((clip_image_size, clip_image_size), Image.LANCZOS)

        # Encode through CLIP
        image_embeds, negative_image_embeds = self.encode_image(
            style_resized,
            torch.device(self.device),
            1,  # num_images_per_prompt
            output_hidden_states=True  # IP-Adapter Plus uses hidden states
        )
        # encode_image returns [1, 257, 1280] (3D)
        # IP-Adapter Plus expects 4D: [batch, num_images, seq_len, embed_dim]
        # Add num_images dimension
        image_embeds = image_embeds.unsqueeze(1)  # [1, 1, 257, 1280]
        negative_image_embeds = negative_image_embeds.unsqueeze(1)  # [1, 1, 257, 1280]

        # Format for DCG batching (match FaceID embedding batch structure)
        # For DCG type 3: [uncond, cond, cond] -> [3, 1, 257, 1280]
        # For DCG type 4: [uncond, cond, cond, cond] -> [4, 1, 257, 1280]
        if dcg_type == 4:
            style_embeds = torch.cat([
                negative_image_embeds,  # uncond
                image_embeds,           # face_only batch (not used but needed for shape)
                image_embeds,           # style_only batch
                image_embeds,           # combined batch
            ], dim=0)  # [4, 1, 257, 1280]
        else:
            # DCG type 3
            style_embeds = torch.cat([
                negative_image_embeds,  # uncond
                image_embeds,           # intermediate
                image_embeds,           # final
            ], dim=0)  # [3, 1, 257, 1280]

        dtype = self.unet.dtype if hasattr(self.unet, 'dtype') else torch.float16
        return style_embeds.to(dtype=dtype)

    def _get_style_clip_embeds(self, style_image, dcg_type=3, dtype=None):
        """Extract CLIP embeddings from style image for IP-Adapter.

        Returns embeddings in the same format as face CLIP embeddings
        so they can be blended together.
        """
        from PIL import Image

        if isinstance(style_image, str):
            style_image = Image.open(style_image).convert("RGB")

        # Resize to 224x224 for CLIP (same as face alignment size)
        style_image_resized = style_image.resize((224, 224), Image.LANCZOS)

        # Get CLIP embeddings using the pipeline's prepare method
        # Must use same dcg_type as face CLIP to ensure matching shapes
        clip_embeds = self.prepare_ip_adapter_image_embeds(
            [[style_image_resized]],
            None,
            torch.device(self.device),
            1,  # batch_size
            True,  # do_cfg
            do_decoupled_cfg=True,
            dcg_type=dcg_type
        )[0]

        # Use UNet's dtype if not specified (float32 on MPS, float16 on CUDA)
        if dtype is None:
            dtype = self.unet.dtype if hasattr(self.unet, 'dtype') else torch.float16
        return clip_embeds.to(dtype=dtype)

    def _get_style_clip_embeds_simple(self, style_image, dcg_type, dtype=None):
        """Extract CLIP embeddings from style image matching the face CLIP format"""
        from PIL import Image

        if isinstance(style_image, str):
            style_image = Image.open(style_image).convert("RGB")

        # Resize to 224x224 for CLIP
        style_image_resized = style_image.resize((224, 224), Image.LANCZOS)

        # Get CLIP embeddings using the same method as face CLIP
        clip_embeds = self.prepare_ip_adapter_image_embeds(
            [[style_image_resized]],
            None,
            torch.device(self.device),
            1,  # batch_size
            True,  # do_cfg
            do_decoupled_cfg=True,
            dcg_type=dcg_type
        )[0]

        # Use UNet's dtype if not specified (float32 on MPS, float16 on CUDA)
        if dtype is None:
            dtype = self.unet.dtype if hasattr(self.unet, 'dtype') else torch.float16
        return clip_embeds.to(dtype=dtype)

    def _get_raw_style_clip_embeds(self, style_image, dtype=None):
        """Extract raw CLIP embeddings from style image for batch-wise stacking.

        Returns a single embedding (no batch expansion) that matches the shape
        of face_id_embeds so they can be concatenated in the batch dimension.
        """
        from PIL import Image

        if isinstance(style_image, str):
            style_image = Image.open(style_image).convert("RGB")

        # Resize to CLIP input size
        clip_image_size = self.image_encoder.config.image_size
        style_image_resized = style_image.resize((clip_image_size, clip_image_size), Image.LANCZOS)

        # Use the pipeline's encode_image method directly
        # This returns (image_embeds, negative_image_embeds)
        image_embeds, _ = self.encode_image(
            style_image_resized,
            torch.device(self.device),
            1,  # num_images_per_prompt
            output_hidden_states=False  # ImageProjection expects final hidden state
        )

        # image_embeds shape: [1, num_tokens, embed_dim]
        # Use UNet's dtype if not specified
        if dtype is None:
            dtype = self.unet.dtype if hasattr(self.unet, 'dtype') else torch.float16

        return image_embeds.to(dtype=dtype)

    def _prepare_style_latents(self, style_image, height, width, generator, dtype):
        """Encode style image to latents for img2img style transfer"""
        from PIL import Image
        import torchvision.transforms.functional as TF

        if isinstance(style_image, str):
            style_image = Image.open(style_image).convert("RGB")

        # Resize to target dimensions
        style_image = style_image.resize((width, height), Image.LANCZOS)

        # Convert to tensor: [0, 255] -> [-1, 1]
        style_tensor = TF.to_tensor(style_image).unsqueeze(0)  # [1, 3, H, W]
        # Always use float32 for VAE encoding to prevent overflow/NaN
        style_tensor = (style_tensor * 2.0 - 1.0).to(self.device, dtype=torch.float32)

        # Encode with VAE (upcast VAE to float32 if needed to prevent overflow)
        with torch.no_grad():
            # Check if VAE needs upcasting (common with SDXL VAE in float16)
            needs_upcast = self.vae.dtype == torch.float16
            if needs_upcast:
                self.vae.to(dtype=torch.float32)

            encoded = self.vae.encode(style_tensor)
            if hasattr(encoded, 'latent_dist'):
                init_latents = encoded.latent_dist.sample(generator)
            else:
                init_latents = encoded.latents
            init_latents = init_latents * self.vae.config.scaling_factor

            # Restore VAE dtype and convert latents to target dtype
            if needs_upcast:
                self.vae.to(dtype=torch.float16)
            init_latents = init_latents.to(dtype=dtype)

        return init_latents

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        denoising_end: Optional[float] = None,

        guidance_scale_a: float = 5.0, 
        guidance_scale_b: float = 5.0,
        dcg_kwargs: Dict = {"dcg_type": 3},

        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        callback_on_step_end = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        init_latents: Optional[torch.Tensor] = None,
        denoising_strength: float = 1.0,
        **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        # Pop dual adapter mode params (handled in execute method)
        kwargs.pop("dual_adapter_mode", None)
        kwargs.pop("style_strength", None)

        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = 999. # arbitrary number to always enable do_cfg inside pipeline
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._denoising_end = denoising_end
        self._interrupt = False
        dcg_type=dcg_kwargs["dcg_type"]

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 4.1 Handle img2img: adjust timesteps based on denoising_strength
        if init_latents is not None and denoising_strength < 1.0:
            # Calculate how many steps to skip (more strength = less skip = more change)
            init_timestep = min(int(num_inference_steps * denoising_strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = timesteps[t_start * self.scheduler.order:]
            num_inference_steps = num_inference_steps - t_start
            print(f">>> [DEBUG] img2img: skipping {t_start} steps, running {num_inference_steps} steps")

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        if init_latents is not None:
            # img2img mode: use style image latents with added noise
            init_latents = init_latents.to(device=device, dtype=prompt_embeds.dtype)

            # Add noise to init_latents at the starting timestep
            noise = torch.randn(
                init_latents.shape,
                generator=generator,
                device=device,
                dtype=init_latents.dtype
            )

            # Get the starting timestep
            if len(timesteps) > 0:
                latent_timestep = timesteps[:1]
                latents = self.scheduler.add_noise(init_latents, noise, latent_timestep)
            else:
                latents = init_latents
            print(f">>> [DEBUG] img2img latents prepared, shape: {latents.shape}")
        else:
            # txt2img mode: generate random latents
            latents = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds, add_text_embeds, add_time_ids = self.preprocess_prompt_embeds(
                dcg_type,
                negative_prompt_embeds,
                prompt_embeds,
                negative_pooled_prompt_embeds,
                add_text_embeds,
                negative_add_time_ids,
                add_time_ids
            )

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # Encode face image for ip_adapter
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
                do_decoupled_cfg=True,
                dcg_type=dcg_type
            )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 8.1 Apply denoising_end
        if (
            self.denoising_end is not None
            and isinstance(self.denoising_end, float)
            and self.denoising_end > 0
            and self.denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (self.denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # 9. Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        self._num_timesteps = len(timesteps)

        if hasattr(self, "track_norms") and self.track_norms:
            self.norms = [[], [], [], [], []]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                # NOTE: expanding to 3 or 4 to enable decoupled guidance
                if self.do_classifier_free_guidance:
                    num_batches = 4 if dcg_type == 4 else 3
                    latent_model_input = torch.cat([latents] * num_batches)
                else:
                    latent_model_input = latents

                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
                    added_cond_kwargs["image_embeds"] = image_embeds
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:                 
                    dcg_out = decoupled_cfg_predict(
                        noise_pred,
                        guidance_scale_a,
                        guidance_scale_b,
                        (i, t),
                        **dcg_kwargs
                    )
                    noise_pred = dcg_out["pred"]
                    if hasattr(self, "track_norms") and self.track_norms:
                        for idx, n in enumerate(self.norms):
                            n.append(dcg_out["norms"][idx])

                # if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
                    add_text_embeds = callback_outputs.pop("add_text_embeds", add_text_embeds)
                    negative_pooled_prompt_embeds = callback_outputs.pop(
                        "negative_pooled_prompt_embeds", negative_pooled_prompt_embeds
                    )
                    add_time_ids = callback_outputs.pop("add_time_ids", add_time_ids)
                    negative_add_time_ids = callback_outputs.pop("negative_add_time_ids", negative_add_time_ids)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                # if XLA_AVAILABLE:
                #     xm.mark_step()

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
            elif latents.dtype != self.vae.dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    self.vae = self.vae.to(latents.dtype)

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
            has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
            if has_latents_mean and has_latents_std:
                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents_std = (
                    torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
                )
                latents = latents * latents_std / self.vae.config.scaling_factor + latents_mean
            else:
                latents = latents / self.vae.config.scaling_factor

            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else:
            image = latents

        if not output_type == "latent":
            # apply watermark if available
            if self.watermark is not None:
                image = self.watermark.apply_watermark(image)

            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


def prepare_dcg_pipeline(config, device, *args, **kwargs):
    model_name = config["model_name"]
    pipe_kwargs = config["pipe_kwargs"]
    n_steps = pipe_kwargs["num_inference_steps"]
    adaptation_method = config["method"]

    method2pipeline = {
        "faceid": DecoupledGuidancePipelineXL, # currently support only faceid-plusv2
    }

    if adaptation_method not in method2pipeline:
        raise ValueError(f"Face adaptation method {adaptation_method} not recongized, supported values: {str(list(method2pipeline.keys()))}")

    if model_name == "base":
        pipe = method2pipeline[adaptation_method].from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            variant="fp16",
            torch_dtype=torch.float16,
            cache_dir="models_cache/sdxl",
            *args,
            **kwargs
        ).to(device)
    if model_name == "hyper":
        if n_steps not in [1, 2, 4, 8]:
            raise ValueError(f"num_inference_steps={n_steps} not supported for hyper checkpoint")
        
        if not os.path.exists(f"models_cache/hyper-{n_steps}-pipe-fused"):
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                variant="fp16", 
                torch_dtype=torch.float16, 
                cache_dir="models_cache/sdxl"
            ).to(device)
            pipe.load_lora_weights(hf_hub_download(
                "ByteDance/Hyper-SD", 
                f"Hyper-SDXL-{n_steps}step{'s' if n_steps > 1 else ''}-lora.safetensors", 
                cache_dir=f"models_cache/sdxl-hyper-{n_steps}")
            )
            pipe.fuse_lora()
            pipe.unload_lora_weights()

            # save pipeline for later use
            pipe.save_pretrained(f"models_cache/hyper-{n_steps}-pipe-fused")

        pipe = method2pipeline[adaptation_method].from_pretrained(
            f"models_cache/hyper-{n_steps}-pipe-fused",
            torch_dtype=torch.float16,
            cache_dir="models_cache/sdxl",
            *args,
            **kwargs,
        ).to(device)

        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    elif model_name == "lcm":
        if not os.path.exists("models_cache/lcm-lora-fused"):
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0", 
                variant="fp16", 
                torch_dtype=torch.float16, 
                cache_dir="models_cache/sdxl"
            ).to(device)

            # let's merge lora weights and save this model
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", cache_dir="models_cache/lcm-lora")
            pipe.fuse_lora()
            pipe.unload_lora_weights()

            # save pipeline for later use
            pipe.save_pretrained("models_cache/lcm-lora-fused")
        pipe = method2pipeline[adaptation_method].from_pretrained(
            "models_cache/lcm-lora-fused",  
            torch_dtype=torch.float16,
            *args,
            **kwargs
        ).to(device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    elif model_name == "lightning":
        if n_steps not in [2, 4, 8]:
            raise ValueError(f"n_steps={n_steps} not supported for lightning checkpoint")

        # Cache full pipeline with loaded UNet for faster subsequent loads
        lightning_cache_path = f"models_cache/lightning-{n_steps}-pipe"

        if not os.path.exists(lightning_cache_path):
            unet = UNet2DConditionModel.from_config(
                "stabilityai/stable-diffusion-xl-base-1.0",
                subfolder="unet",
                cache_dir="models_cache/sdxl").to(device, torch.float16)

            unet.load_state_dict(load_file(
                hf_hub_download(
                    "ByteDance/SDXL-Lightning",
                    f"sdxl_lightning_{n_steps}step_unet.safetensors",
                    cache_dir=f"models_cache/sdxl-lightning-{n_steps}"),
                    device="cpu"
                )
            )
            # Create base pipeline with lightning UNet
            base_pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                unet=unet,
                torch_dtype=torch.float16,
                cache_dir="models_cache/sdxl",
                variant="fp16",
            ).to(device)
            base_pipe.scheduler = EulerDiscreteScheduler.from_config(
                base_pipe.scheduler.config,
                timestep_spacing="trailing"
            )
            # Save for later use
            base_pipe.save_pretrained(lightning_cache_path)

        # Load cached pipeline
        pipe = method2pipeline[adaptation_method].from_pretrained(
            lightning_cache_path,
            torch_dtype=torch.float16,
            *args,
            **kwargs
        ).to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config,
            timestep_spacing="trailing"
        )
    elif model_name == "turbo":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/sdxl-turbo", subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            subfolder="unet",
            cache_dir="models_cache/sdxl-turbo"
        ).to(device)
        pipe = method2pipeline[adaptation_method].from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            unet=unet,
            cache_dir="models_cache/sdxl",
            *args,
            **kwargs
        ).to(device)
        pipe.scheduler = scheduler
    elif model_name == "realvis":
        # RealVisXL V4.0 - Photorealistic model with excellent skin texture
        if n_steps not in [1, 2, 4, 8]:
            raise ValueError(f"num_inference_steps={n_steps} not supported for realvis checkpoint")

        # Load RealVisXL as base, then apply Hyper-SD LoRA for speed
        if not os.path.exists(f"models_cache/realvis-hyper-{n_steps}-fused"):
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "SG161222/RealVisXL_V4.0",
                torch_dtype=torch.float16,
                cache_dir="models_cache/realvis"
            ).to(device)
            # Apply Hyper-SD LoRA for few-step generation
            pipe.load_lora_weights(hf_hub_download(
                "ByteDance/Hyper-SD",
                f"Hyper-SDXL-{n_steps}step{'s' if n_steps > 1 else ''}-lora.safetensors",
                cache_dir=f"models_cache/sdxl-hyper-{n_steps}")
            )
            pipe.fuse_lora()
            pipe.unload_lora_weights()
            pipe.save_pretrained(f"models_cache/realvis-hyper-{n_steps}-fused")

        pipe = method2pipeline[adaptation_method].from_pretrained(
            f"models_cache/realvis-hyper-{n_steps}-fused",
            torch_dtype=torch.float16,
            cache_dir="models_cache/realvis",
            *args,
            **kwargs,
        ).to(device)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    return pipe


def get_faceid_pipeline(config, device, dual_adapter=True):
    """Load FaceID pipeline with optional dual IP-Adapter support.

    Args:
        config: Pipeline configuration
        device: Torch device
        dual_adapter: If True, load both FaceID and Style IP-Adapters
    """
    os.makedirs("models_cache", exist_ok=True)

    pipe = prepare_dcg_pipeline(config, device)
    ip_adapter_scale = config.get("ip_adapter_scale", 0.5)

    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        torch_dtype=torch.float16,
        cache_dir="models_cache/faceid_img_encoder",
    ).to(device)

    if ("t2i" not in config) or (not config["t2i"]):
        pipe.register_modules(image_encoder=image_encoder)

        clip_image_size = pipe.image_encoder.config.image_size
        feature_extractor = CLIPImageProcessor(size=clip_image_size, crop_size=clip_image_size)
        pipe.register_modules(feature_extractor=feature_extractor)

        if dual_adapter:
            # Load BOTH IP-Adapters: FaceID (for identity) + Standard (for style)
            # This allows completely separate control of face and style
            print(">>> Loading Dual IP-Adapters (FaceID + Style)...")
            pipe.load_ip_adapter(
                ["h94/IP-Adapter-FaceID", "h94/IP-Adapter"],
                subfolder=[None, "sdxl_models"],
                weight_name=[
                    "ip-adapter-faceid-plusv2_sdxl.bin",
                    "ip-adapter-plus_sdxl_vit-h.safetensors"
                ],
                image_encoder_folder=None,
                cache_dir="models_cache/ipadapter"
            )
            # Set scales: [FaceID scale, Style scale]
            # Style scale will be dynamically adjusted based on style_strength
            pipe.set_ip_adapter_scale([ip_adapter_scale, 0.0])
            pipe.dual_adapter_enabled = True
            print(">>> Dual IP-Adapters loaded successfully")
        else:
            # Single adapter mode (FaceID only)
            pipe.load_ip_adapter(
                "h94/IP-Adapter-FaceID",
                subfolder=None,
                weight_name="ip-adapter-faceid-plusv2_sdxl.bin",
                image_encoder_folder=None,
                cache_dir="models_cache/ipadapter"
            )
            pipe.set_ip_adapter_scale(ip_adapter_scale)
            pipe.dual_adapter_enabled = False

        pipe.track_norms = False

    # Set up FaceAnalysis providers based on device type
    if torch.cuda.is_available() and 'cuda' in str(device):
        device_id = device.index if hasattr(device, 'index') and device.index is not None else 0
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [{"device_id": device_id}, {}]
    else:
        # MPS or CPU - use CPU provider for ONNX (insightface doesn't support MPS directly)
        device_id = 0
        providers = ['CPUExecutionProvider']
        provider_options = [{}]

    print("DEVICE: ", device, " | FaceAnalysis providers: ", providers)
    app = FaceAnalysis(name="buffalo_l", providers=providers, provider_options=provider_options)
    app.prepare(ctx_id=0, det_size=(640, 640))
    pipe.app = app

    return pipe


name2pipe = {
    "faceid": get_faceid_pipeline,
}