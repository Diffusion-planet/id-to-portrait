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
    LCMScheduler,
    ControlNetModel,
)
from diffusers.models import ImageProjection
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import retrieve_timesteps
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from .utils import get_faceid_embeds
from .dcg import decoupled_cfg_predict
from .face_parsing import FaceParser, extract_hair_region


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
    """pipeline to be used together with IpAdapter, adds separate image and textual guidance.

    v5 Update: Added ControlNet support for structure preservation.
    When style transfer is enabled, depth is extracted from face image
    and used as ControlNet conditioning to ensure face structure is preserved.
    """

    # ControlNet components (lazy loaded)
    _controlnet = None
    _depth_estimator = None

    # v6: Face parser module (lazy loaded, modular design)
    _face_parser_module = None

    @property
    def controlnet(self):
        """Lazy load ControlNet model."""
        return self._controlnet

    @controlnet.setter
    def controlnet(self, value):
        self._controlnet = value

    @property
    def depth_estimator(self):
        """Lazy load depth estimator."""
        if self._depth_estimator is None:
            try:
                from controlnet_aux import MidasDetector
                print(">>> Loading MiDaS depth estimator...")
                self._depth_estimator = MidasDetector.from_pretrained(
                    "lllyasviel/Annotators"
                )
                print(">>> MiDaS depth estimator loaded")
            except ImportError:
                print(">>> Warning: controlnet_aux not installed, depth estimation disabled")
                return None
        return self._depth_estimator

    @property
    def face_parser(self) -> FaceParser:
        """Lazy load FaceParser module for face+hair segmentation (v6)."""
        if self._face_parser_module is None:
            device = self.device if hasattr(self, 'device') else "cpu"
            self._face_parser_module = FaceParser(device=device)
        return self._face_parser_module

    def get_depth_map(self, image, target_size=(1024, 1024)):
        """Extract depth map from image using MiDaS.

        Args:
            image: PIL Image or path to image
            target_size: Output size (width, height)

        Returns:
            Depth map as PIL Image, or None if depth estimation is disabled
        """
        from PIL import Image

        if self.depth_estimator is None:
            return None

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Resize to target size
        image_resized = image.resize(target_size, Image.LANCZOS)

        # Get depth map
        depth_map = self.depth_estimator(image_resized)

        return depth_map

    def apply_mask_to_depth(self, depth_image, face_mask, mask_edge_blur=40, depth_blur_radius=80):
        """Apply face mask to depth map to remove face shape constraint.

        Blurs the face region in the depth map so ControlNet doesn't enforce
        the original face's shape while preserving background/body structure.

        Args:
            depth_image: PIL Image (depth map)
            face_mask: PIL Image (face mask, white=face region)
            mask_edge_blur: Gaussian blur radius for mask edge softening
            depth_blur_radius: Gaussian blur radius for depth map blurring

        Returns:
            Modified depth map as PIL Image
        """
        from PIL import Image, ImageFilter
        import numpy as np

        # Ensure same size
        if face_mask.size != depth_image.size:
            face_mask = face_mask.resize(depth_image.size, Image.LANCZOS)

        # Convert to numpy
        depth_array = np.array(depth_image).astype(np.float32)
        mask_array = np.array(face_mask.convert('L')).astype(np.float32) / 255.0

        # Expand mask slightly and blur edges for smooth transition
        # v7: Use frontend-controlled mask_edge_blur parameter
        mask_pil = Image.fromarray((mask_array * 255).astype(np.uint8))
        if mask_edge_blur > 0:
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=mask_edge_blur))
        mask_array = np.array(mask_pil).astype(np.float32) / 255.0

        # Create heavily blurred version of depth map
        # v7: Use frontend-controlled depth_blur_radius parameter
        blurred_depth = depth_image.filter(ImageFilter.GaussianBlur(radius=depth_blur_radius))
        blurred_array = np.array(blurred_depth).astype(np.float32)

        # Blend: original depth outside face, blurred depth inside face
        if len(depth_array.shape) == 3:
            mask_array = mask_array[:, :, np.newaxis]

        result_array = depth_array * (1 - mask_array) + blurred_array * mask_array
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)

        return Image.fromarray(result_array)

    def prepare_controlnet_image(self, image, width, height, batch_size, num_images_per_prompt, device, dtype, dcg_type=3):
        """Prepare depth image for ControlNet conditioning.

        Args:
            image: PIL Image (depth map)
            width, height: Target dimensions
            batch_size: Batch size
            num_images_per_prompt: Images per prompt
            device: Torch device
            dtype: Torch dtype
            dcg_type: DCG type for batch expansion

        Returns:
            Processed image tensor for ControlNet
        """
        from PIL import Image
        import torchvision.transforms.functional as TF

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Resize to target dimensions
        image = image.resize((width, height), Image.LANCZOS)

        # Convert to tensor: [0, 255] -> [0, 1]
        image_tensor = TF.to_tensor(image).unsqueeze(0)  # [1, 3, H, W]
        image_tensor = image_tensor.to(device=device, dtype=dtype)

        # Expand for DCG batching (same as latents)
        num_batches = 4 if dcg_type == 4 else 3
        image_tensor = torch.cat([image_tensor] * num_batches, dim=0)

        return image_tensor

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
        denoising_strength=0.6,
        dual_adapter_mode=False,
        negative_prompt=None,
        progress_callback=None,
        ip_adapter_scale=0.8,
        # v6: Face masking parameters
        mask_style_face=True,
        face_mask_method='gaussian_blur',
        include_hair_in_mask=True,
        face_mask_blur_radius=50,
        # v7: Advanced Face masking parameters (frontend control)
        mask_expand_pixels=10,
        mask_edge_blur=10,
        controlnet_scale=0.4,
        depth_blur_radius=80,
        style_strength_cap=0.10,
        denoising_min=0.90,
        # v7.2: Hair coverage ratio
        bbox_expand_ratio=1.5,
        # v7.3: Hair preservation from face reference
        hair_strength=0.5,
        # v7.4: Aspect ratio adjustment toggle
        adjust_mask_aspect_ratio=False,
        # v7.6: Face size matching - constrain generated face to style face size
        match_style_face_size=True,
        # v7: Output directory for saving masked style image
        output_dir=None,
    ):
        from PIL import Image

        dcg_type = pipe_kwargs["dcg_kwargs"]["dcg_type"]

        # Style Transfer Mode v4: CLIP Blending
        #
        # Previous v3 approach (Dual IP-Adapter) had a critical issue:
        # - Style adapter's CLIP embedding contains full semantic info
        # - When style image has no person, "no person" signal overrides FaceID
        # - Result: Generated image loses the person entirely
        #
        # v4 Solution: CLIP Embedding Blending
        # - Blend face_clip and style_clip: (1-α)*face + α*style
        # - face_clip always contains "person exists" information
        # - Blending preserves person while transferring style attributes
        #
        # See GitHub Issue #1 for detailed analysis.

        if style_image is not None:
            # ============================================================
            # CLIP Blending Mode (v4)
            # ============================================================
            # Problem: Dual IP-Adapter approach causes identity loss when
            # style image has no person (CLIP encodes "no person" semantic)
            #
            # Solution: Blend face_clip and style_clip embeddings
            # - face_clip contains "person_presence" information
            # - Blending preserves person information while adding style
            #
            # Formula: blended = (1 - α) * face_clip + α * style_clip
            # ============================================================
            from PIL import Image
            import cv2
            import numpy as np
            from insightface.utils import face_align

            print(f">>> [img2img + FaceID Mode] style_strength={style_strength}, denoising={denoising_strength}")

            # 1. Extract Face ID embedding directly (bypass prepare_ip_adapter_image_embeds)
            if isinstance(face_image, str):
                face_pil = Image.open(face_image)
            else:
                face_pil = face_image
            image = cv2.cvtColor(np.asarray(face_pil), cv2.COLOR_BGR2RGB)

            # v7: Detect input face early for aspect ratio and size calculation
            input_faces = self.app.get(image)
            input_face_aspect_ratio = None
            input_face_area = None  # v7.6: For face size matching
            if len(input_faces) > 0:
                input_bbox = input_faces[0].bbox  # [x1, y1, x2, y2]
                input_face_width = input_bbox[2] - input_bbox[0]
                input_face_height = input_bbox[3] - input_bbox[1]
                if input_face_width > 0:
                    input_face_aspect_ratio = input_face_height / input_face_width
                    input_face_area = input_face_width * input_face_height  # v7.6
                    print(f">>> [v7] Input face aspect ratio: {input_face_aspect_ratio:.2f} (w={input_face_width:.0f}, h={input_face_height:.0f})")
                    print(f">>> [v7.6] Input face area: {input_face_area:.0f}px")

            # Load style image
            if isinstance(style_image, str):
                style_pil = Image.open(style_image).convert("RGB")
            else:
                style_pil = style_image

            height = pipe_kwargs.get("height", 1024)
            width = pipe_kwargs.get("width", 1024)
            dtype = self.unet.dtype if hasattr(self.unet, 'dtype') else torch.float16

            # ============================================================
            # v6: Face Masking in Style Image
            # ============================================================
            # Problem: If style image has a face, its structure conflicts
            # with FaceID trying to generate a different face.
            #
            # Solution: Detect face in style image and mask it before processing
            # - Mask face region in style image (blur/fill)
            # - Mask face region in depth map (neutral depth)
            # - This allows FaceID to control face generation entirely
            # ============================================================
            style_face_mask = None
            style_pil_for_depth = style_pil
            style_pil_for_latents = style_pil

            if mask_style_face:
                # Check if style image has a face
                try:
                    style_cv = cv2.cvtColor(np.asarray(style_pil), cv2.COLOR_RGB2BGR)
                    style_faces = self.app.get(style_cv)
                    style_has_face = len(style_faces) > 0
                except Exception as e:
                    print(f">>> [v6] Face detection in style image failed: {e}")
                    style_has_face = False

                if style_has_face:
                    print(f">>> [v6] Style image has face detected, extracting mask...")
                    # v7.1: Get face bbox from InsightFace to filter SegFormer noise
                    style_face_bbox = None
                    effective_bbox_expand_ratio = bbox_expand_ratio  # v7.6: May be adjusted for size matching
                    if len(style_faces) > 0:
                        bbox = style_faces[0].bbox  # [x1, y1, x2, y2]
                        style_face_bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                        print(f">>> [v7.1] Style face bbox: {style_face_bbox}")

                        # v7.6: Face size matching - adjust bbox_expand_ratio based on face size comparison
                        if match_style_face_size and input_face_area is not None:
                            style_face_width = bbox[2] - bbox[0]
                            style_face_height = bbox[3] - bbox[1]
                            style_face_area = style_face_width * style_face_height

                            # Calculate size ratio (normalized to image dimensions)
                            input_img_area = face_pil.size[0] * face_pil.size[1]
                            style_img_area = style_pil.size[0] * style_pil.size[1]

                            # Normalize face areas by their respective image sizes
                            input_face_ratio = input_face_area / input_img_area
                            style_face_ratio = style_face_area / style_img_area

                            size_ratio = style_face_ratio / input_face_ratio if input_face_ratio > 0 else 1.0

                            print(f">>> [v7.6] Style face area: {style_face_area:.0f}px ({style_face_ratio*100:.1f}% of image)")
                            print(f">>> [v7.6] Input face ratio: {input_face_ratio*100:.1f}%, Style face ratio: {style_face_ratio*100:.1f}%")
                            print(f">>> [v7.6] Size ratio (style/input): {size_ratio:.2f}")

                            # If input face is larger (ratio < 1), reduce expansion to keep generated face smaller
                            if size_ratio < 1.0:
                                # Reduce expansion proportionally: smaller style face = less expansion
                                # Clamp to minimum of 1.0 (no expansion)
                                effective_bbox_expand_ratio = max(1.0, bbox_expand_ratio * size_ratio)
                                print(f">>> [v7.6] Adjusted bbox_expand_ratio: {bbox_expand_ratio:.2f} -> {effective_bbox_expand_ratio:.2f}")
                            else:
                                print(f">>> [v7.6] Style face is larger/equal, keeping bbox_expand_ratio={bbox_expand_ratio:.2f}")

                    # v7: Use frontend-controlled mask_expand_pixels and mask_edge_blur
                    style_face_mask = self.face_parser.get_face_hair_mask(
                        style_pil,
                        target_size=(width, height),
                        include_hair=include_hair_in_mask,
                        expand_pixels=mask_expand_pixels,
                        blur_radius=mask_edge_blur,
                        face_bbox=style_face_bbox,  # v7.1: Pass bbox to filter noise
                        bbox_expand_ratio=effective_bbox_expand_ratio  # v7.6: Size-adjusted
                    )

                    # v7.4: Only adjust mask aspect ratio if toggle is enabled
                    if adjust_mask_aspect_ratio and style_face_mask is not None and input_face_aspect_ratio is not None:
                        style_face_mask = self.face_parser.adjust_mask_for_aspect_ratio(
                            style_face_mask,
                            target_aspect_ratio=input_face_aspect_ratio
                        )
                        print(f">>> [Pipeline] Mask adjusted for aspect ratio")
                    elif input_face_aspect_ratio is not None:
                        print(f">>> [Pipeline] Aspect ratio adjustment disabled (toggle off)")

                    if style_face_mask is not None:
                        print(f">>> [v6] Applying face mask to style image (method={face_mask_method})")
                        style_pil_masked = self.face_parser.apply_mask(
                            style_pil,
                            style_face_mask,
                            method=face_mask_method,
                            blur_radius=face_mask_blur_radius
                        )
                        # Use masked image for depth and latents
                        style_pil_for_depth = style_pil_masked
                        style_pil_for_latents = style_pil_masked

                        # v7: Save masked style image for preview
                        if output_dir is not None:
                            import uuid
                            masked_style_id = str(uuid.uuid4())
                            masked_style_filename = f"masked_style_{masked_style_id}.png"
                            masked_style_path = f"{output_dir}/{masked_style_filename}"
                            style_pil_masked.save(masked_style_path)
                            self._masked_style_path = masked_style_path
                            print(f">>> [v7] Saved masked style image: {masked_style_path}")
                        else:
                            self._masked_style_path = None
                    else:
                        print(f">>> [v6] Face parsing failed, using original style image")
                        self._masked_style_path = None
                else:
                    print(f">>> [v6] No face detected in style image, skipping mask")
                    self._masked_style_path = None
            else:
                print(f">>> [v6] Face masking disabled (mask_style_face=False)")
                self._masked_style_path = None

            # v5: Extract depth from STYLE image for ControlNet (preserves style's structure)
            # v6: Mask face region in depth map when face masking is active
            # This preserves background/body structure but removes face shape constraint
            depth_image = None
            # v7: Use frontend-controlled controlnet_scale parameter (no hardcoded default)

            if self.controlnet is not None:
                print(f">>> [ControlNet] Extracting depth from STYLE image...")
                depth_image = self.get_depth_map(style_pil, target_size=(width, height))

                if depth_image is not None:
                    print(f">>> [ControlNet] Depth map extracted: {depth_image.size}")

                    # v7: Apply face mask to depth with frontend-controlled parameters
                    if style_face_mask is not None:
                        print(f">>> [ControlNet] Applying face mask to depth map...")
                        depth_image = self.apply_mask_to_depth(
                            depth_image,
                            style_face_mask,
                            mask_edge_blur=mask_edge_blur,
                            depth_blur_radius=depth_blur_radius
                        )
                        print(f">>> [v7] ControlNet face masking active - scale={controlnet_scale}, depth_blur={depth_blur_radius}")
                else:
                    print(f">>> [ControlNet] Depth extraction failed, proceeding without ControlNet")

            # v7: Use frontend-controlled denoising parameters (no auto-override)
            init_latents = None
            if style_face_mask is not None:
                # Use frontend-controlled denoising_min when face masking is active
                effective_denoising = denoising_min
                print(f">>> [v7] Face masking active - using denoising_min={effective_denoising}")
            else:
                effective_denoising = denoising_strength

            if effective_denoising < 1.0:
                print(f">>> [img2img] Encoding style image to latents...")
                init_latents = self._prepare_style_latents(style_pil_for_latents, height, width, generator, dtype)
                print(f">>> [img2img] Style latents ready: {init_latents.shape}")

            # Reuse already detected input faces (v7: detected earlier for aspect ratio)
            faces = input_faces

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
                    neg_ref_images_embeds,
                    ref_images_embeds,
                    ref_images_embeds,
                    ref_images_embeds,
                ], dim=0).to(dtype=dtype, device=self.device)
            else:
                id_embeds = torch.cat([
                    neg_ref_images_embeds,
                    ref_images_embeds,
                    ref_images_embeds,
                ], dim=0).to(dtype=dtype, device=self.device)

            # 2. Extract Face CLIP embedding
            face_crop = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)
            face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            face_clip_embeds, neg_face_clip_embeds = self.encode_image(
                face_crop_pil, torch.device(self.device), 1, output_hidden_states=True
            )

            # 2.5 v7.3: Extract Hair CLIP embedding from face reference
            hair_clip_embeds = None
            neg_hair_clip_embeds = None
            effective_hair_strength = hair_strength

            if hair_strength > 0:
                print(f">>> [v7.3] Extracting hair region from face reference (strength={hair_strength})")
                hair_region_image = extract_hair_region(face_pil, self.face_parser)

                if hair_region_image is not None:
                    # Resize hair region to CLIP input size
                    hair_resized = hair_region_image.resize((224, 224), Image.LANCZOS)
                    hair_clip_embeds, neg_hair_clip_embeds = self.encode_image(
                        hair_resized, torch.device(self.device), 1, output_hidden_states=True
                    )
                    print(f">>> [v7.3] Hair CLIP embedding extracted: {hair_clip_embeds.shape}")
                else:
                    print(f">>> [v7.3] Hair extraction failed, using face_clip only")
                    effective_hair_strength = 0

            # 3. CLIP Embedding: v7 - Use frontend-controlled style_strength_cap (no auto-override)
            if style_face_mask is not None:
                # Use frontend-controlled style_strength_cap when face masking is active
                effective_style_strength = style_strength_cap
                print(f">>> [v7] Face masking active - using style_strength_cap={effective_style_strength}")
            else:
                effective_style_strength = style_strength

            style_for_clip = style_pil_for_latents if style_face_mask is not None else style_pil
            style_resized = style_for_clip.resize((224, 224), Image.LANCZOS)
            style_clip_embeds, neg_style_clip_embeds = self.encode_image(
                style_resized, torch.device(self.device), 1, output_hidden_states=True
            )

            # v7.3: Three-way CLIP blending: Face + Hair + Style
            # Step 1: Blend face and hair (within face reference)
            if hair_clip_embeds is not None and effective_hair_strength > 0:
                face_hair_blend_pos = (1 - effective_hair_strength) * face_clip_embeds + effective_hair_strength * hair_clip_embeds
                face_hair_blend_neg = (1 - effective_hair_strength) * neg_face_clip_embeds + effective_hair_strength * neg_hair_clip_embeds
                print(f">>> [v7.3] Face+Hair blend: Face={1-effective_hair_strength:.1%}, Hair={effective_hair_strength:.1%}")
            else:
                face_hair_blend_pos = face_clip_embeds
                face_hair_blend_neg = neg_face_clip_embeds

            # Step 2: Blend (Face+Hair) with Style
            blended_pos = (1 - effective_style_strength) * face_hair_blend_pos + effective_style_strength * style_clip_embeds
            blended_neg = (1 - effective_style_strength) * face_hair_blend_neg + effective_style_strength * neg_style_clip_embeds

            if style_face_mask is not None:
                print(f">>> [CLIP] Using MASKED style image with reduced strength")
            print(f">>> [DEBUG] face_clip_embeds shape: {face_clip_embeds.shape}")
            print(f">>> [DEBUG] style_clip_embeds shape: {style_clip_embeds.shape}")
            if hair_clip_embeds is not None:
                print(f">>> [DEBUG] hair_clip_embeds shape: {hair_clip_embeds.shape}")
            print(f">>> [DEBUG] Final blend - FaceRef: {1-effective_style_strength:.1%}, Style: {effective_style_strength:.1%}")

            print(f">>> [DEBUG] blended_pos shape: {blended_pos.shape}")

            # 5. Build batched CLIP embedding for DCG
            # ImageProjection expects 4D: [batch, num_images, seq_len, hidden_dim]
            # encode_image returns 3D: [1, seq_len, hidden_dim], so add num_images dim
            blended_pos_4d = blended_pos.unsqueeze(1)  # [1, 1, 257, 1280]
            blended_neg_4d = blended_neg.unsqueeze(1)  # [1, 1, 257, 1280]

            if dcg_type == 4:
                blended_clip_batched = torch.cat([
                    blended_neg_4d, blended_pos_4d, blended_pos_4d, blended_pos_4d
                ], dim=0)  # [4, 1, 257, 1280]
            else:
                blended_clip_batched = torch.cat([
                    blended_neg_4d, blended_pos_4d, blended_pos_4d
                ], dim=0)  # [3, 1, 257, 1280]

            print(f">>> [DEBUG] blended_clip_batched shape: {blended_clip_batched.shape}")

            # 6. Set blended CLIP embedding to FaceID adapter
            self.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = blended_clip_batched.to(dtype=dtype)
            self.unet.encoder_hid_proj.image_projection_layers[0].shortcut = True

            total_steps = pipe_kwargs.get("num_inference_steps", 4)
            def step_callback(step_idx, t, latents):
                if progress_callback:
                    progress_callback(step_idx + 1, total_steps)

            # 7. Use only FaceID adapter (Style adapter disabled if dual adapter loaded)
            if getattr(self, 'dual_adapter_enabled', False):
                self.set_ip_adapter_scale([ip_adapter_scale, 0.0])
                print(f">>> [DEBUG] FaceID scale={ip_adapter_scale}, Style adapter disabled (scale=0)")

                # Create dummy style embedding with correct shape for Style adapter
                # Style adapter (IP-Adapter Plus) expects [batch, 1, 257, 1280]
                batch_size = 4 if dcg_type == 4 else 3
                dummy_style_embeds = torch.zeros(
                    batch_size, 1, 257, 1280,
                    dtype=dtype, device=self.device
                )
                ip_adapter_embeds = [id_embeds, dummy_style_embeds]
            else:
                ip_adapter_embeds = [id_embeds]

            # 8. Call pipeline with img2img + ControlNet + FaceID
            call_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "ip_adapter_image_embeds": ip_adapter_embeds,
                "num_images_per_prompt": 1,
                "generator": generator,
                "callback": step_callback,
                "callback_steps": 1,
                **pipe_kwargs
            }

            # Add img2img params if init_latents available
            if init_latents is not None:
                call_kwargs["init_latents"] = init_latents
                call_kwargs["denoising_strength"] = effective_denoising
                print(f">>> [img2img] Using style latents with denoising={effective_denoising}")

            # Add ControlNet params if depth image is available
            if depth_image is not None and self.controlnet is not None:
                call_kwargs["control_image"] = depth_image
                call_kwargs["controlnet_conditioning_scale"] = controlnet_scale
                print(f">>> [ControlNet] Using depth conditioning (scale={controlnet_scale})")

            res = self(**call_kwargs).images[0]

        else:
            # Original mode: single adapter (no style image)
            # When dual adapters are loaded, we need to provide embeddings for both
            from PIL import Image
            import cv2
            import numpy as np
            from insightface.utils import face_align

            print(f">>> [FaceID Only Mode] No style image provided")

            # Extract face embedding directly (same as CLIP blending mode)
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
                    neg_ref_images_embeds,
                    ref_images_embeds,
                    ref_images_embeds,
                    ref_images_embeds,
                ], dim=0).to(dtype=dtype, device=self.device)
            else:
                id_embeds = torch.cat([
                    neg_ref_images_embeds,
                    ref_images_embeds,
                    ref_images_embeds,
                ], dim=0).to(dtype=dtype, device=self.device)

            # Extract Face CLIP embedding for FaceID adapter
            face_crop = face_align.norm_crop(image, landmark=faces[0].kps, image_size=224)
            face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            face_clip_embeds, neg_face_clip_embeds = self.encode_image(
                face_crop_pil, torch.device(self.device), 1, output_hidden_states=True
            )

            # Set face CLIP embedding to FaceID adapter
            face_clip_4d = face_clip_embeds.unsqueeze(1)
            neg_face_clip_4d = neg_face_clip_embeds.unsqueeze(1)

            if dcg_type == 4:
                face_clip_batched = torch.cat([
                    neg_face_clip_4d, face_clip_4d, face_clip_4d, face_clip_4d
                ], dim=0)
            else:
                face_clip_batched = torch.cat([
                    neg_face_clip_4d, face_clip_4d, face_clip_4d
                ], dim=0)

            self.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = face_clip_batched.to(dtype=dtype)
            self.unet.encoder_hid_proj.image_projection_layers[0].shortcut = True

            # Handle dual adapter case: provide dummy style embedding
            if getattr(self, 'dual_adapter_enabled', False):
                self.set_ip_adapter_scale([ip_adapter_scale, 0.0])
                print(f">>> [DEBUG] FaceID scale={ip_adapter_scale}, Style adapter disabled (scale=0)")

                # Create dummy style embedding for Style adapter
                batch_size = 4 if dcg_type == 4 else 3
                dummy_style_embeds = torch.zeros(
                    batch_size, 1, 257, 1280,
                    dtype=dtype, device=self.device
                )
                ip_adapter_embeds = [id_embeds, dummy_style_embeds]
            else:
                ip_adapter_embeds = [id_embeds]

            # Create step callback for progress reporting
            total_steps = pipe_kwargs.get("num_inference_steps", 4)
            def step_callback(step_idx, t, latents):
                if progress_callback:
                    progress_callback(step_idx + 1, total_steps)

            res = self(
                prompt=prompt,
                negative_prompt=negative_prompt,
                ip_adapter_image_embeds=ip_adapter_embeds,
                num_images_per_prompt=1,
                generator=generator,
                callback=step_callback,
                callback_steps=1,
                **pipe_kwargs
            ).images[0]

        if after_hook_fn is not None:
            after_hook_fn(self, res)

        # v7: Return dict with both image and masked style path
        return {
            "image": res,
            "masked_style_path": getattr(self, '_masked_style_path', None)
        }

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
        # ControlNet parameters (v5)
        control_image = None,
        controlnet_conditioning_scale: float = 0.5,
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

        # 7.5 Prepare ControlNet image (v5)
        controlnet_image = None
        if control_image is not None and self.controlnet is not None:
            controlnet_image = self.prepare_controlnet_image(
                control_image,
                width,
                height,
                batch_size,
                num_images_per_prompt,
                device,
                prompt_embeds.dtype,
                dcg_type
            )
            print(f">>> [ControlNet] Prepared control image: {controlnet_image.shape}")

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

                # ControlNet conditioning (v5)
                down_block_res_samples = None
                mid_block_res_sample = None
                if controlnet_image is not None and self.controlnet is not None:
                    down_block_res_samples, mid_block_res_sample = self.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=controlnet_image,
                        conditioning_scale=controlnet_conditioning_scale,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )

                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
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


def get_faceid_pipeline(config, device, dual_adapter=True, load_controlnet=True):
    """Load FaceID pipeline with optional dual IP-Adapter and ControlNet support.

    Args:
        config: Pipeline configuration
        device: Torch device
        dual_adapter: If True, load both FaceID and Style IP-Adapters
        load_controlnet: If True, load ControlNet for depth conditioning (v5)
    """
    os.makedirs("models_cache", exist_ok=True)

    pipe = prepare_dcg_pipeline(config, device)
    ip_adapter_scale = config.get("ip_adapter_scale", 0.5)

    # Load ControlNet for depth conditioning (v5)
    if load_controlnet:
        try:
            print(">>> Loading ControlNet depth model...")
            # Use float16 on CUDA and MPS (Mac), float32 only on CPU
            use_fp16 = torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
            dtype = torch.float16 if use_fp16 else torch.float32
            controlnet = ControlNetModel.from_pretrained(
                "diffusers/controlnet-depth-sdxl-1.0",
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                cache_dir="models_cache/controlnet"
            ).to(device)
            pipe.controlnet = controlnet
            print(">>> ControlNet depth model loaded successfully")
        except Exception as e:
            print(f">>> Warning: Failed to load ControlNet: {e}")
            print(">>> Proceeding without ControlNet support")
            pipe.controlnet = None
    else:
        pipe.controlnet = None

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