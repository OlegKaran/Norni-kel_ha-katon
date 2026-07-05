from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import models


MODEL_PATH = Path(__file__).resolve().parent / "models" / "resnet18_epoch_10_state_dict.pt"
CLASS_NAMES = {
    0: "Рядовая руда",
    1: "Труднообрабатываемая руда",
}


@dataclass(frozen=True)
class OrePrediction:
    class_index: int
    class_name: str
    vote_percentage: float
    ordinary_tiles: int
    difficult_tiles: int
    total_tiles: int


def split_to_tiles(
    image: Union[str, Path, Image.Image],
    tile_size: int = 1024,
    overlap: int = 256,
) -> list[torch.Tensor]:
    """Разбивает изображение на тайлы так же, как код обучения из ноутбука."""
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap должен быть неотрицательным и меньше tile_size")

    if isinstance(image, Image.Image):
        image_rgb = image.convert("RGB")
    else:
        image_rgb = Image.open(image).convert("RGB")

    image_array = np.array(image_rgb)
    height, width = image_array.shape[:2]
    stride = tile_size - overlap
    tiles = []

    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = image_array[y : y + tile_size, x : x + tile_size]
            tiles.append(torch.from_numpy(tile.copy()).permute(2, 0, 1))

    if not tiles:
        raise ValueError(
            f"Для классификации изображение должно быть не меньше {tile_size}×{tile_size} пикселей; "
            f"получено {width}×{height}"
        )

    return tiles


def load_resnet18_model(
    model_path: Union[str, Path] = MODEL_PATH,
    device: Union[str, torch.device] = "cpu",
) -> torch.nn.Module:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Не найден файл весов модели: {model_path}")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 2)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def model_predict(
    image: Union[str, Path, Image.Image],
    model: torch.nn.Module,
    device: Union[str, torch.device] = "cpu",
    ind_to_class: dict[int, str] | None = None,
    batch_size: int = 4,
) -> OrePrediction:
    """Классифицирует каждый тайл и агрегирует результат большинством голосов."""
    class_names = ind_to_class or CLASS_NAMES
    tiles = split_to_tiles(image)
    predictions = []

    model.eval()
    model.to(device)
    with torch.inference_mode():
        for start in range(0, len(tiles), batch_size):
            batch = torch.stack(tiles[start : start + batch_size]).float().div_(255.0)
            logits = model(batch.to(device))
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())

    total_tiles = len(predictions)
    difficult_tiles = int(sum(predictions))
    ordinary_tiles = total_tiles - difficult_tiles
    difficult_share = difficult_tiles / total_tiles

    # В исходном ноутбуке класс 1 выбирается только при строгом большинстве.
    class_index = 1 if difficult_share > 0.5 else 0
    winning_tiles = difficult_tiles if class_index == 1 else ordinary_tiles

    return OrePrediction(
        class_index=class_index,
        class_name=class_names[class_index],
        vote_percentage=winning_tiles / total_tiles * 100,
        ordinary_tiles=ordinary_tiles,
        difficult_tiles=difficult_tiles,
        total_tiles=total_tiles,
    )
