"""
FastFace v7.2 Face Parsing Module

This module provides face+hair segmentation functionality using BiSeNet.
v7.2: Switched from SegFormer to BiSeNet for more accurate face contour parsing.

Usage:
    from src.face_parsing import FaceParser

    parser = FaceParser(device="mps")
    mask = parser.get_face_hair_mask(image, face_bbox=bbox)
    masked_image = parser.apply_mask(image, mask, method='gaussian_blur')

Classes:
    FaceParser: Main class for face parsing and masking operations
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional, Tuple, Union, Literal
import torchvision.transforms as transforms
from torchvision.models import resnet18


# BiSeNet face parsing label mapping (CelebAMask-HQ dataset)
# 19 classes from face-parsing.PyTorch
BISENET_LABELS = {
    0: 'background',
    1: 'skin',
    2: 'left_brow',
    3: 'right_brow',
    4: 'left_eye',
    5: 'right_eye',
    6: 'eyeglasses',
    7: 'left_ear',
    8: 'right_ear',
    9: 'earring',
    10: 'nose',
    11: 'mouth',
    12: 'upper_lip',
    13: 'lower_lip',
    14: 'neck',
    15: 'neckline',
    16: 'cloth',
    17: 'hair',
    18: 'hat'
}

# Labels to include in face+hair mask for BiSeNet
BISENET_FACE_LABELS = [1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13]  # Face features (no hair)
BISENET_HAIR_LABELS = [17]  # Hair only
BISENET_FACE_HAIR_LABELS = BISENET_FACE_LABELS + BISENET_HAIR_LABELS


# BiSeNet Model Components
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet = resnet18(weights=None)
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self._get_resnet_features(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up

    def _get_resnet_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        feat8 = self.resnet.layer1(x)
        feat8 = self.resnet.layer2(feat8)
        feat16 = self.resnet.layer3(feat8)
        feat32 = self.resnet.layer4(feat16)
        return feat8, feat16, feat32


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)

        return feat_out


class FaceParser:
    """
    Face parsing and masking module using BiSeNet.

    v7.2: Switched from SegFormer to BiSeNet for more accurate face contour parsing.
    - BiSeNet provides accurate face contour following
    - Falls back to elliptical mask from InsightFace bbox if BiSeNet fails
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        cache_dir: str = "models_cache/face_parser"
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.cache_dir = cache_dir
        self.model_path = os.path.join(cache_dir, "79999_iter.pth")

        self._model = None
        self._transform = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> bool:
        """Load BiSeNet model."""
        if self._is_loaded:
            return True

        try:
            os.makedirs(self.cache_dir, exist_ok=True)

            # Check if model exists, if not download
            if not os.path.exists(self.model_path):
                print(f">>> [FaceParser] BiSeNet model not found at {self.model_path}")
                print(f">>> [FaceParser] Downloading BiSeNet model...")
                self._download_model()

            print(f">>> [FaceParser] Loading BiSeNet from {self.model_path}...")

            self._model = BiSeNet(n_classes=19)
            state_dict = torch.load(self.model_path, map_location='cpu', weights_only=True)
            # Use strict=False to ignore missing fc layer (not used in inference)
            self._model.load_state_dict(state_dict, strict=False)
            self._model = self._model.to(self.device)
            self._model.eval()

            # Preprocessing transform
            self._transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            self._is_loaded = True
            print(f">>> [FaceParser] BiSeNet loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f">>> [FaceParser] Failed to load BiSeNet: {e}")
            self._is_loaded = False
            return False

    def _download_model(self):
        """Download BiSeNet pretrained weights."""
        import urllib.request

        url = "https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth"
        print(f">>> [FaceParser] Downloading from {url}")

        try:
            urllib.request.urlretrieve(url, self.model_path)
            print(f">>> [FaceParser] Downloaded to {self.model_path}")
        except Exception as e:
            # Try alternative URL
            alt_url = "https://drive.google.com/uc?export=download&id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
            print(f">>> [FaceParser] Primary download failed, trying alternative...")
            try:
                urllib.request.urlretrieve(alt_url, self.model_path)
            except Exception as e2:
                raise RuntimeError(f"Failed to download BiSeNet model: {e2}")

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._transform = None
        self._is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print(">>> [FaceParser] Model unloaded")

    def get_segmentation(
        self,
        image: Union[str, Image.Image],
        target_size: Optional[Tuple[int, int]] = None
    ) -> Optional[np.ndarray]:
        """Get full segmentation map from image using BiSeNet."""
        if not self.load():
            return None

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if target_size is None:
            target_size = image.size

        orig_w, orig_h = image.size

        # Pad image to square to preserve aspect ratio before BiSeNet
        # BiSeNet's transform resizes to 512x512, which distorts non-square images
        max_dim = max(orig_w, orig_h)
        padded_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        # Center the original image
        pad_left = (max_dim - orig_w) // 2
        pad_top = (max_dim - orig_h) // 2
        padded_img.paste(image, (pad_left, pad_top))

        # Process padded square image through BiSeNet
        img_tensor = self._transform(padded_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self._model(img_tensor)
            pred = output.argmax(dim=1).squeeze().cpu().numpy()

        # pred is 512x512 (from padded square input)
        # First resize to padded square size, then crop out padding
        pred_pil = Image.fromarray(pred.astype(np.uint8))
        pred_square = pred_pil.resize((max_dim, max_dim), Image.NEAREST)

        # Crop out the padding to get back original aspect ratio
        pred_cropped = pred_square.crop((pad_left, pad_top, pad_left + orig_w, pad_top + orig_h))

        # Resize to target size
        pred_resized = pred_cropped.resize(target_size, Image.NEAREST)

        return np.array(pred_resized)

    def get_face_hair_mask(
        self,
        image: Union[str, Image.Image],
        target_size: Optional[Tuple[int, int]] = None,
        include_hair: bool = True,
        expand_pixels: int = 10,
        blur_radius: int = 10,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        bbox_expand_ratio: float = 1.5
    ) -> Optional[Image.Image]:
        """
        Extract face+hair mask from image using BiSeNet.

        v7.2: Primary method uses BiSeNet for accurate face contours.
        Falls back to elliptical mask from InsightFace bbox if BiSeNet fails.

        Args:
            image: PIL Image or path to image
            target_size: Output size (width, height). If None, uses image size.
            include_hair: Whether to include hair in mask
            expand_pixels: Pixels to expand mask boundary (frontend controlled)
            blur_radius: Gaussian blur radius for soft edges
            face_bbox: Face bounding box from InsightFace (x1, y1, x2, y2)
            bbox_expand_ratio: Ratio to expand bbox for hair (frontend controlled, 1.0-3.0)

        Returns:
            Grayscale mask (white=face region, black=background), or None
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if target_size is None:
            target_size = image.size

        mask = None

        # Try BiSeNet first for accurate face contours
        if self.load():
            try:
                # Get segmentation
                seg_map = self.get_segmentation(image, target_size)

                if seg_map is not None:
                    # Create mask from face labels
                    labels_to_include = BISENET_FACE_LABELS.copy()
                    if include_hair:
                        labels_to_include.extend(BISENET_HAIR_LABELS)

                    mask = np.zeros(seg_map.shape, dtype=np.uint8)
                    for label in labels_to_include:
                        mask[seg_map == label] = 255

                    coverage = np.sum(mask > 0) / mask.size * 100
                    print(f">>> [FaceParser] BiSeNet mask coverage: {coverage:.1f}%")

                    # Check if mask is reasonable (at least 5% coverage for a face)
                    if coverage < 5:
                        print(f">>> [FaceParser] BiSeNet coverage too low ({coverage:.1f}%), falling back to ellipse")
                        mask = None
                    else:
                        # Apply morphological cleanup and expansion
                        try:
                            from scipy import ndimage
                            # Fill small holes
                            mask = ndimage.binary_fill_holes(mask > 127).astype(np.uint8) * 255
                            # Remove small islands
                            labeled, num_features = ndimage.label(mask > 127)
                            if num_features > 1:
                                sizes = ndimage.sum(mask > 127, labeled, range(1, num_features + 1))
                                largest = np.argmax(sizes) + 1
                                mask = (labeled == largest).astype(np.uint8) * 255

                            # v7.2: Apply dilation based on bbox_expand_ratio
                            # Higher ratio = more expansion for hair coverage
                            if bbox_expand_ratio > 1.0:
                                # Calculate dilation iterations based on image size and ratio
                                base_iterations = int(min(target_size) * 0.02)  # ~2% of image size
                                dilation_iterations = int(base_iterations * (bbox_expand_ratio - 1.0))
                                if dilation_iterations > 0:
                                    mask = ndimage.binary_dilation(
                                        mask > 127,
                                        iterations=dilation_iterations
                                    ).astype(np.uint8) * 255
                                    new_coverage = np.sum(mask > 0) / mask.size * 100
                                    print(f">>> [FaceParser] After dilation (ratio={bbox_expand_ratio}): {new_coverage:.1f}%")
                        except ImportError:
                            pass

            except Exception as e:
                print(f">>> [FaceParser] BiSeNet inference failed: {e}, falling back to ellipse")
                mask = None

        # Fallback: Create elliptical mask from InsightFace bbox
        if mask is None and face_bbox is not None:
            print(f">>> [FaceParser] Using elliptical mask fallback (bbox_expand_ratio={bbox_expand_ratio})")
            mask = self._create_ellipse_mask(image, target_size, face_bbox, include_hair, bbox_expand_ratio)
        elif mask is None:
            print(">>> [FaceParser] ERROR: No face_bbox provided and BiSeNet failed")
            return None

        # Convert to PIL
        mask_pil = Image.fromarray(mask, mode='L')

        # Expand mask
        if expand_pixels > 0:
            mask_pil = mask_pil.filter(
                ImageFilter.MaxFilter(expand_pixels * 2 + 1)
            )

        # Apply gaussian blur for soft edges
        if blur_radius > 0:
            mask_pil = mask_pil.filter(
                ImageFilter.GaussianBlur(radius=blur_radius)
            )

        return mask_pil

    def _create_ellipse_mask(
        self,
        image: Image.Image,
        target_size: Tuple[int, int],
        face_bbox: Tuple[int, int, int, int],
        include_hair: bool = True,
        bbox_expand_ratio: float = 1.5
    ) -> np.ndarray:
        """Create elliptical mask from face bounding box."""
        x1, y1, x2, y2 = face_bbox

        # Scale bbox coordinates to target size
        orig_w, orig_h = image.size
        scale_x = target_size[0] / orig_w
        scale_y = target_size[1] / orig_h

        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)

        # Calculate face dimensions
        face_w = x2_scaled - x1_scaled
        face_h = y2_scaled - y1_scaled
        center_x = (x1_scaled + x2_scaled) // 2
        center_y = (y1_scaled + y2_scaled) // 2

        # v7.2: Use bbox_expand_ratio from frontend for hair coverage
        # bbox_expand_ratio=1.0 means tight to face, 2.0 means double size for hair
        hair_expand = bbox_expand_ratio if include_hair else 1.0

        # Calculate ellipse dimensions based on expand ratio
        expand_w = int(face_w * (hair_expand - 1) * 0.5)  # Horizontal expansion
        expand_h_down = int(face_h * 0.3)  # 30% down for chin/neck
        expand_h_up = int(face_h * (hair_expand - 0.5))  # Up expansion for hair

        # Calculate ellipse parameters
        ellipse_w = face_w + expand_w * 2
        ellipse_h = face_h + expand_h_up + expand_h_down
        ellipse_center_y = center_y - (expand_h_up - expand_h_down) // 2

        # Create elliptical mask using numpy meshgrid
        mask = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

        y_coords, x_coords = np.ogrid[:target_size[1], :target_size[0]]
        a = ellipse_w / 2
        b = ellipse_h / 2

        ellipse_mask = ((x_coords - center_x) / max(a, 1)) ** 2 + \
                      ((y_coords - ellipse_center_y) / max(b, 1)) ** 2 <= 1
        mask[ellipse_mask] = 255

        coverage = np.sum(mask > 0) / mask.size * 100
        print(f">>> [FaceParser] Ellipse mask: center=({center_x}, {ellipse_center_y}), "
              f"size=({ellipse_w}x{ellipse_h}), coverage={coverage:.1f}%, expand_ratio={bbox_expand_ratio}")

        return mask

    def apply_mask(
        self,
        image: Union[str, Image.Image],
        mask: Image.Image,
        method: Literal['gaussian_blur', 'fill', 'noise'] = 'gaussian_blur',
        blur_radius: int = 50,
        fill_color: Tuple[int, int, int] = (128, 128, 128)
    ) -> Image.Image:
        """Apply mask to neutralize face region in image."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)

        img_array = np.array(image, dtype=np.float32)
        mask_array = np.array(mask, dtype=np.float32) / 255.0
        mask_array = mask_array[:, :, np.newaxis]

        if method == 'gaussian_blur':
            blurred = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            blurred_array = np.array(blurred, dtype=np.float32)
            result_array = img_array * (1 - mask_array) + blurred_array * mask_array

        elif method == 'fill':
            fill_array = np.array(fill_color, dtype=np.float32)
            result_array = img_array * (1 - mask_array) + fill_array * mask_array

        elif method == 'noise':
            noise = np.random.normal(128, 30, img_array.shape).astype(np.float32)
            noise = np.clip(noise, 0, 255)
            result_array = img_array * (1 - mask_array) + noise * mask_array

        else:
            raise ValueError(f"Unknown method: {method}")

        result_array = np.clip(result_array, 0, 255).astype(np.uint8)
        print(f">>> [FaceParser] Mask applied with method='{method}'")

        return Image.fromarray(result_array)

    def apply_mask_to_depth(
        self,
        depth_image: Image.Image,
        face_mask: Image.Image,
        fill_value: float = 0.5
    ) -> Image.Image:
        """Apply face mask to depth map."""
        if face_mask.size != depth_image.size:
            face_mask = face_mask.resize(depth_image.size, Image.LANCZOS)

        depth_array = np.array(depth_image.convert('L'), dtype=np.float32) / 255.0
        mask_array = np.array(face_mask, dtype=np.float32) / 255.0

        result = depth_array * (1 - mask_array) + fill_value * mask_array
        result = (result * 255).astype(np.uint8)

        print(f">>> [FaceParser] Depth mask applied (fill_value={fill_value})")
        return Image.fromarray(result).convert('RGB')

    def get_mask_bbox(self, mask: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        """Get bounding box of non-zero region in mask."""
        mask_array = np.array(mask)
        non_zero = np.where(mask_array > 127)

        if len(non_zero[0]) == 0:
            return None

        y_min, y_max = non_zero[0].min(), non_zero[0].max()
        x_min, x_max = non_zero[1].min(), non_zero[1].max()

        return (x_min, y_min, x_max, y_max)

    def get_face_aspect_ratio(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate aspect ratio (height/width) from bounding box."""
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min

        if width == 0:
            return 1.0

        return height / width

    def adjust_mask_for_aspect_ratio(
        self,
        mask: Image.Image,
        target_aspect_ratio: float,
        max_scale: float = 1.5
    ) -> Image.Image:
        """Adjust mask vertically/horizontally to match target aspect ratio."""
        bbox = self.get_mask_bbox(mask)
        if bbox is None:
            print(">>> [FaceParser] No mask region found, skipping aspect ratio adjustment")
            return mask

        current_ratio = self.get_face_aspect_ratio(bbox)
        x_min, y_min, x_max, y_max = bbox

        ratio_diff = target_aspect_ratio / current_ratio
        ratio_diff = min(max(ratio_diff, 1/max_scale), max_scale)

        if abs(ratio_diff - 1.0) < 0.1:
            print(f">>> [FaceParser] Aspect ratio similar ({current_ratio:.2f} vs {target_aspect_ratio:.2f}), no adjustment")
            return mask

        print(f">>> [FaceParser] Adjusting mask aspect ratio: {current_ratio:.2f} -> {target_aspect_ratio:.2f} (scale: {ratio_diff:.2f})")

        mask_array = np.array(mask).astype(np.float32)
        height, width = mask_array.shape

        mask_height = y_max - y_min
        mask_width = x_max - x_min
        mask_center_y = (y_min + y_max) // 2
        mask_center_x = (x_min + x_max) // 2

        if ratio_diff > 1.0:
            result = np.zeros_like(mask_array)
            for y in range(height):
                for x in range(width):
                    if mask_array[y, x] > 0:
                        rel_y = y - mask_center_y
                        new_rel_y = int(rel_y * ratio_diff)
                        new_y = mask_center_y + new_rel_y
                        if 0 <= new_y < height:
                            result[new_y, x] = max(result[new_y, x], mask_array[y, x])
        else:
            scale = 1.0 / ratio_diff
            result = np.zeros_like(mask_array)
            for y in range(height):
                for x in range(width):
                    if mask_array[y, x] > 0:
                        rel_x = x - mask_center_x
                        new_rel_x = int(rel_x * scale)
                        new_x = mask_center_x + new_rel_x
                        if 0 <= new_x < width:
                            result[y, new_x] = max(result[y, new_x], mask_array[y, x])

        result_pil = Image.fromarray(result.astype(np.uint8), mode='L')
        result_pil = result_pil.filter(ImageFilter.GaussianBlur(radius=3))

        return result_pil


def create_face_parser(device: str = "cpu") -> FaceParser:
    """Create and load a FaceParser instance."""
    parser = FaceParser(device=device)
    parser.load()
    return parser


def extract_hair_region(
    image: Image.Image,
    parser: FaceParser,
    background_color: Tuple[int, int, int] = (128, 128, 128)
) -> Optional[Image.Image]:
    """
    Extract hair-only region from image using BiSeNet segmentation.

    This function extracts only the hair region from a face image,
    which can then be encoded by CLIP to capture hairstyle information.

    Args:
        image: PIL Image (face reference image)
        parser: FaceParser instance
        background_color: Color to fill non-hair regions (neutral gray)

    Returns:
        PIL Image with only hair visible (rest is neutral gray), or None if failed
    """
    if not parser.load():
        print(">>> [extract_hair_region] Failed to load FaceParser")
        return None

    try:
        # Get full segmentation map
        seg_map = parser.get_segmentation(image, target_size=image.size)

        if seg_map is None:
            print(">>> [extract_hair_region] Segmentation failed")
            return None

        # Create hair-only mask (label 17 = hair)
        hair_mask = np.zeros(seg_map.shape, dtype=np.uint8)
        hair_mask[seg_map == 17] = 255

        # Check if hair was detected
        hair_coverage = np.sum(hair_mask > 0) / hair_mask.size * 100
        print(f">>> [extract_hair_region] Hair coverage: {hair_coverage:.1f}%")

        if hair_coverage < 1:
            print(">>> [extract_hair_region] No significant hair detected")
            return None

        # Apply morphological operations to clean up the mask
        try:
            from scipy import ndimage
            # Fill small holes
            hair_mask = ndimage.binary_fill_holes(hair_mask > 127).astype(np.uint8) * 255
            # Small dilation for smoother edges
            hair_mask = ndimage.binary_dilation(hair_mask > 127, iterations=2).astype(np.uint8) * 255
        except ImportError:
            pass

        # Create output image: hair visible, rest is neutral gray
        img_array = np.array(image, dtype=np.float32)
        mask_array = hair_mask.astype(np.float32) / 255.0
        mask_array = mask_array[:, :, np.newaxis]

        # Background fill
        bg_array = np.array(background_color, dtype=np.float32)
        result_array = img_array * mask_array + bg_array * (1 - mask_array)
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)

        print(f">>> [extract_hair_region] Hair region extracted successfully")
        return Image.fromarray(result_array)

    except Exception as e:
        print(f">>> [extract_hair_region] Error: {e}")
        return None


def extract_face_only_region(
    image: Image.Image,
    parser: FaceParser,
    background_color: Tuple[int, int, int] = (128, 128, 128)
) -> Optional[Image.Image]:
    """
    Extract face-only region (without hair) from image using BiSeNet.

    Args:
        image: PIL Image (face reference image)
        parser: FaceParser instance
        background_color: Color to fill non-face regions

    Returns:
        PIL Image with only face visible (no hair), or None if failed
    """
    if not parser.load():
        return None

    try:
        seg_map = parser.get_segmentation(image, target_size=image.size)
        if seg_map is None:
            return None

        # Create face-only mask (exclude hair label 17)
        face_mask = np.zeros(seg_map.shape, dtype=np.uint8)
        for label in BISENET_FACE_LABELS:
            face_mask[seg_map == label] = 255

        face_coverage = np.sum(face_mask > 0) / face_mask.size * 100
        if face_coverage < 3:
            return None

        try:
            from scipy import ndimage
            face_mask = ndimage.binary_fill_holes(face_mask > 127).astype(np.uint8) * 255
        except ImportError:
            pass

        img_array = np.array(image, dtype=np.float32)
        mask_array = face_mask.astype(np.float32) / 255.0
        mask_array = mask_array[:, :, np.newaxis]

        bg_array = np.array(background_color, dtype=np.float32)
        result_array = img_array * mask_array + bg_array * (1 - mask_array)
        result_array = np.clip(result_array, 0, 255).astype(np.uint8)

        return Image.fromarray(result_array)

    except Exception as e:
        print(f">>> [extract_face_only_region] Error: {e}")
        return None
