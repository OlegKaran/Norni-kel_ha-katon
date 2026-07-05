from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from segment_anything import sam_model_registry


MODEL_PATH = Path(__file__).resolve().parent / "models" / "best_model_b_v2.pth"
WINDOW_SIZE = 1024
STRIDE = 512
THRESHOLD = 0.5
DILATION_RADIUS = 5
MIN_AREA = 50


@dataclass(frozen=True)
class SAMPrediction:
    mask: np.ndarray
    overlay: Image.Image
    positive_percentage: float
    positive_pixels: int
    total_pixels: int


class SAMWrapper(nn.Module):
    """Prompt-free SAM wrapper из обучающего ноутбука."""

    def __init__(self, sam_model: nn.Module):
        super().__init__()
        self.sam = sam_model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        input_images = torch.stack([self.sam.preprocess(image) for image in images])
        image_embeddings = self.sam.image_encoder(input_images)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(points=None, boxes=None, masks=None)
        if sparse_embeddings.shape[0] == 1:
            sparse_embeddings = sparse_embeddings.expand(batch_size, -1, -1)
        if dense_embeddings.shape[0] == 1:
            dense_embeddings = dense_embeddings.expand(batch_size, -1, -1, -1)
        low_resolution_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return F.interpolate(
            low_resolution_masks,
            size=images.shape[2:],
            mode="bilinear",
            align_corners=False,
        )


def load_sam_model(
    model_path: Union[str, Path] = MODEL_PATH,
    device: Union[str, torch.device] = "cpu",
) -> SAMWrapper:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден файл весов SAM: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    clean_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key
        for prefix in ("module.", "sam_model.", "model.", "sam."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        clean_state_dict[clean_key] = value

    sam_model = sam_model_registry["vit_b"](checkpoint=None)
    sam_model.load_state_dict(clean_state_dict, strict=True)
    model = SAMWrapper(sam_model).to(device)
    model.eval()
    return model


def get_image_patches(
    image: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> list[tuple[np.ndarray, int, int]]:
    patches = []
    height, width = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    fill_value = 255 if image.dtype == np.uint8 else 1.0

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + window_size, height)
            x_end = min(x + window_size, width)
            shape = (window_size, window_size, channels) if channels > 1 else (window_size, window_size)
            patch = np.full(shape, fill_value, dtype=image.dtype)
            patch[: y_end - y, : x_end - x] = image[y:y_end, x:x_end]
            patches.append((patch, x, y))
    return patches


def assemble_mask(
    predictions: list[tuple[np.ndarray, int, int]],
    original_shape: tuple[int, int],
    use_voting: bool = True,
) -> np.ndarray:
    height, width = original_shape
    mask_sum = np.zeros((height, width), dtype=np.int32)
    count = np.zeros((height, width), dtype=np.int32)
    for predicted_patch, x, y in predictions:
        y_end = min(y + predicted_patch.shape[0], height)
        x_end = min(x + predicted_patch.shape[1], width)
        mask_sum[y:y_end, x:x_end] += predicted_patch[: y_end - y, : x_end - x]
        count[y:y_end, x:x_end] += 1
    if use_voting:
        return (mask_sum > count / 2).astype(np.uint8)
    return mask_sum


def postprocess_mask(mask: np.ndarray, dilation_radius: int = DILATION_RADIUS, min_area: int = MIN_AREA) -> np.ndarray:
    kernel_size = dilation_radius if dilation_radius % 2 != 0 else dilation_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    merged_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    number_of_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged_mask, connectivity=8)
    cleaned_mask = np.zeros_like(merged_mask, dtype=bool)
    for label in range(1, number_of_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned_mask[labels == label] = True
    return cleaned_mask


def create_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    blue = np.array([30, 120, 255], dtype=np.float32)
    rgb[mask] = (1 - alpha) * rgb[mask] + alpha * blue
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def predict_sam(
    image: Image.Image,
    model: SAMWrapper,
    device: Union[str, torch.device] = "cpu",
    threshold: float = THRESHOLD,
) -> SAMPrediction:
    image_rgb = np.asarray(image.convert("RGB"))
    patches = get_image_patches(image_rgb)
    predictions = []
    device = torch.device(device)
    model.eval()
    with torch.inference_mode():
        for patch, x, y in patches:
            tensor = torch.as_tensor(patch).permute(2, 0, 1).float().unsqueeze(0).to(device)
            inference_precision = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with inference_precision:
                output = model(tensor)
            probabilities = torch.sigmoid(output).squeeze().cpu().numpy()
            predictions.append(((probabilities > threshold).astype(np.uint8), x, y))

    mask = assemble_mask(predictions, image_rgb.shape[:2], use_voting=True)
    mask = postprocess_mask(mask)
    positive_pixels = int(np.count_nonzero(mask))
    total_pixels = int(mask.size)
    percentage = positive_pixels / total_pixels * 100 if total_pixels else 0.0
    return SAMPrediction(
        mask=mask,
        overlay=create_overlay(image, mask),
        positive_percentage=float(percentage),
        positive_pixels=positive_pixels,
        total_pixels=total_pixels,
    )
