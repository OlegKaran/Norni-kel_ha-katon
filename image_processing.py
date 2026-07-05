from __future__ import annotations

import heapq
import os
from dataclasses import dataclass

os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_multiotsu
from skimage.segmentation import slic
from sklearn.cluster import KMeans


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def safe_multiotsu(values: np.ndarray, classes: int = 3) -> np.ndarray:
    """Multi-Otsu с устойчивым результатом для почти однородных областей."""
    values = np.asarray(values)
    if values.size == 0:
        raise ValueError("Область анализа не содержит пикселей")
    try:
        return threshold_multiotsu(values, classes=classes)
    except ValueError:
        quantiles = np.linspace(0, 1, classes + 1)[1:-1]
        return np.asarray(np.quantile(values.astype(np.float32), quantiles), dtype=np.float32)


def calculate_dark_score(image_bgr, local_window=51):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_mean = cv2.boxFilter(gray, -1, (local_window, local_window), normalize=True, borderType=cv2.BORDER_REFLECT)
    local_mean_square = cv2.boxFilter(gray**2, -1, (local_window, local_window), normalize=True, borderType=cv2.BORDER_REFLECT)
    local_std = np.sqrt(np.maximum(local_mean_square - local_mean**2, 0))
    return np.maximum((local_mean - gray) / (local_std + 1.0), 0)


def calculate_inclusion_density(
    image_bgr,
    dark_score,
    dark_threshold=85,
    soft_width=30,
    texture_window=21,
    texture_threshold=6,
    density_windows=(31, 61, 121),
    z_normalization=2.0,
):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    relative_darkness = np.clip(dark_score / z_normalization, 0, 1)
    absolute_darkness = np.clip((dark_threshold - gray) / soft_width, 0, 1)
    local_mean = cv2.boxFilter(gray, -1, (texture_window, texture_window), borderType=cv2.BORDER_REFLECT)
    local_mean_square = cv2.boxFilter(gray**2, -1, (texture_window, texture_window), borderType=cv2.BORDER_REFLECT)
    local_std = np.sqrt(np.maximum(local_mean_square - local_mean**2, 0))
    texture_score = np.clip(local_std / texture_threshold, 0, 1)
    absolute_response = absolute_darkness * (0.25 + 0.75 * texture_score)
    inclusion_score = np.maximum(relative_darkness, absolute_response)
    density_maps = [
        cv2.boxFilter(inclusion_score.astype(np.float32), -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
        for window in density_windows
    ]
    return np.maximum.reduce(density_maps)


def get_adjacency_edges(labels):
    edge_parts = []
    left, right = labels[:, :-1], labels[:, 1:]
    difference = left != right
    if np.any(difference):
        edge_parts.append(np.column_stack([left[difference], right[difference]]))
    top, bottom = labels[:-1, :], labels[1:, :]
    difference = top != bottom
    if np.any(difference):
        edge_parts.append(np.column_stack([top[difference], bottom[difference]]))
    if not edge_parts:
        return np.empty((0, 2), dtype=np.int32)
    edges = np.vstack(edge_parts).astype(np.int32)
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def merge_superpixels_by_density(labels, density, mean_tolerance=0.07, max_region_range=0.10):
    number_of_superpixels = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=number_of_superpixels).astype(np.float64)
    density_sums = np.bincount(labels.ravel(), weights=density.ravel(), minlength=number_of_superpixels).astype(np.float64)
    means = density_sums / np.maximum(areas, 1)
    parent = np.arange(number_of_superpixels)
    minimum_mean, maximum_mean = means.copy(), means.copy()
    neighbors = [set() for _ in range(number_of_superpixels)]
    edges = get_adjacency_edges(labels)
    for first, second in edges:
        first, second = int(first), int(second)
        neighbors[first].add(second)
        neighbors[second].add(first)

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    queue = []
    for first, second in edges:
        heapq.heappush(queue, (abs(means[first] - means[second]), int(first), int(second)))
    while queue:
        _, first, second = heapq.heappop(queue)
        first, second = find(first), find(second)
        if first == second:
            continue
        first_mean = density_sums[first] / max(areas[first], 1)
        second_mean = density_sums[second] / max(areas[second], 1)
        combined_minimum = min(minimum_mean[first], minimum_mean[second])
        combined_maximum = max(maximum_mean[first], maximum_mean[second])
        if abs(first_mean - second_mean) > mean_tolerance or combined_maximum - combined_minimum > max_region_range:
            continue
        if areas[first] < areas[second]:
            first, second = second, first
        parent[second] = first
        areas[first] += areas[second]
        density_sums[first] += density_sums[second]
        minimum_mean[first], maximum_mean[first] = combined_minimum, combined_maximum
        means[first] = density_sums[first] / max(areas[first], 1)
        new_neighbors = neighbors[first] | neighbors[second]
        new_neighbors.discard(first)
        new_neighbors.discard(second)
        neighbors[first] = set()
        for neighbor in new_neighbors:
            neighbor = find(neighbor)
            if neighbor == first:
                continue
            neighbors[first].add(neighbor)
            neighbors[neighbor].discard(second)
            neighbors[neighbor].discard(first)
            neighbors[neighbor].add(first)
            heapq.heappush(queue, (abs(means[first] - means[neighbor]), first, neighbor))
        neighbors[second].clear()

    roots = np.array([find(index) for index in range(number_of_superpixels)])
    _, region_indices = np.unique(roots, return_inverse=True)
    region_map = region_indices[labels]
    number_of_regions = int(region_map.max()) + 1
    region_areas = np.bincount(region_map.ravel(), minlength=number_of_regions).astype(np.float64)
    region_density = np.bincount(region_map.ravel(), weights=density.ravel(), minlength=number_of_regions) / np.maximum(region_areas, 1)
    return region_map, region_density, region_areas


def segment_density_by_similarity(
    density,
    n_segments=3500,
    compactness=2.0,
    slic_sigma=2.0,
    mean_tolerance=0.07,
    max_region_range=0.10,
):
    density_smooth = cv2.GaussianBlur(density.astype(np.float32), (0, 0), sigmaX=2, sigmaY=2)
    superpixel_labels = slic(
        density_smooth,
        n_segments=n_segments,
        compactness=compactness,
        sigma=slic_sigma,
        start_label=0,
        channel_axis=None,
        convert2lab=False,
    )
    region_map, region_density, region_areas = merge_superpixels_by_density(
        superpixel_labels, density_smooth, mean_tolerance, max_region_range
    )
    if len(region_density) < 3:
        raise ValueError("После объединения осталось меньше трёх областей концентрации")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(region_density.reshape(-1, 1), sample_weight=region_areas)
    centers = kmeans.cluster_centers_.ravel()
    order = np.argsort(centers)
    cluster_to_class = np.zeros(3, dtype=np.uint8)
    for class_value, cluster in enumerate(order):
        cluster_to_class[cluster] = class_value
    return cluster_to_class[cluster_labels][region_map], centers[order]


def refine_concentration_boundaries(
    density,
    class_map,
    class_centers,
    spatial_sigma=12,
    density_sigma=0.08,
    spatial_weight=0.55,
    iterations=3,
):
    density = density.astype(np.float32)
    result = class_map.astype(np.uint8).copy()
    class_centers = np.asarray(class_centers, dtype=np.float32)
    for _ in range(iterations):
        class_scores = []
        for class_value in range(3):
            spatial_support = cv2.GaussianBlur(
                (result == class_value).astype(np.float32),
                (0, 0),
                sigmaX=spatial_sigma,
                sigmaY=spatial_sigma,
                borderType=cv2.BORDER_REFLECT,
            )
            density_support = np.exp(-((density - class_centers[class_value]) ** 2) / (2 * density_sigma**2))
            class_scores.append(spatial_weight * spatial_support + (1 - spatial_weight) * density_support)
        result = np.argmax(np.stack(class_scores, axis=0), axis=0).astype(np.uint8)
    return result


def filter_components_by_area(mask, min_area=2, max_area=None):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    keep = areas >= min_area
    if max_area is not None:
        keep &= areas <= max_area
    keep[0] = False
    return keep[labels]


def detect_talc_pixels(
    image_bgr,
    class_map,
    roi_classes=(2,),
    local_window=31,
    min_texture_std=10.0,
    min_relative_darkness=0.4,
    otsu_classes=3,
    max_dark_lightness=110,
    blackhat_sizes=(7, 13, 25, 51),
    min_blackhat_response=4,
    density_windows=(31, 61, 121),
    min_candidate_density=0.006,
    min_component_area=2,
    max_component_area=None,
):
    height, width = image_bgr.shape[:2]
    if class_map.shape != (height, width):
        class_map = cv2.resize(class_map.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    roi_mask = np.isin(class_map, roi_classes)
    if not np.any(roi_mask):
        return {"talc_pixel_mask": np.zeros((height, width), dtype=bool), "roi_mask": roi_mask}
    lightness = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    thresholds = safe_multiotsu(lightness[roi_mask], classes=otsu_classes)
    dark_limit = min(float(thresholds[0]), max_dark_lightness)
    absolute_dark_mask = lightness <= dark_limit
    local_mean = cv2.boxFilter(lightness, -1, (local_window, local_window), normalize=True, borderType=cv2.BORDER_REFLECT)
    local_mean_square = cv2.boxFilter(lightness**2, -1, (local_window, local_window), normalize=True, borderType=cv2.BORDER_REFLECT)
    local_std = np.sqrt(np.maximum(local_mean_square - local_mean**2, 0))
    relative_darkness = np.maximum((local_mean - lightness) / (local_std + 1.0), 0)
    texture_mask = local_std >= min_texture_std
    blackhat = np.maximum.reduce([
        cv2.morphologyEx(
            lightness.astype(np.uint8),
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(size) | 1, int(size) | 1)),
        )
        for size in blackhat_sizes
    ])
    talc_candidates = roi_mask & absolute_dark_mask & (
        ((relative_darkness >= min_relative_darkness) & texture_mask) | (blackhat >= min_blackhat_response)
    )
    candidate_density = np.maximum.reduce([
        cv2.boxFilter(talc_candidates.astype(np.float32), -1, (window, window), normalize=True, borderType=cv2.BORDER_REFLECT)
        for window in density_windows
    ])
    talc_pixel_mask = filter_components_by_area(
        talc_candidates & (candidate_density >= min_candidate_density), min_component_area, max_component_area
    )
    return {
        "talc_pixel_mask": talc_pixel_mask,
        "roi_mask": roi_mask,
        "texture_mask": texture_mask,
        "relative_darkness": relative_darkness,
        "blackhat": blackhat,
        "otsu_thresholds": thresholds,
        "dark_limit": dark_limit,
    }


def add_constrained_medium_pixels(
    image_bgr,
    talc_result,
    max_medium_lightness=120,
    max_distance_from_core=8,
    medium_min_relative_darkness=0.7,
    medium_min_blackhat_response=8,
    required_evidence_count=2,
    core_density_window=31,
    min_core_density=0.005,
    min_component_area=2,
):
    core_mask = talc_result["talc_pixel_mask"].astype(bool)
    if not np.any(core_mask) or len(talc_result.get("otsu_thresholds", [])) < 2:
        return core_mask
    roi_mask = talc_result["roi_mask"].astype(bool)
    lightness = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    medium_limit = min(float(talc_result["otsu_thresholds"][1]), max_medium_lightness)
    medium_intensity_mask = (lightness > talc_result["dark_limit"]) & (lightness <= medium_limit)
    evidence_count = (
        (talc_result["relative_darkness"] >= medium_min_relative_darkness).astype(np.uint8)
        + (talc_result["blackhat"] >= medium_min_blackhat_response).astype(np.uint8)
        + talc_result["texture_mask"].astype(np.uint8)
    )
    distance_from_core = cv2.distanceTransform((~core_mask).astype(np.uint8), cv2.DIST_L2, 5)
    core_density = cv2.boxFilter(core_mask.astype(np.float32), -1, (core_density_window, core_density_window), normalize=True, borderType=cv2.BORDER_REFLECT)
    medium_pixel_mask = (
        roi_mask
        & medium_intensity_mask
        & (evidence_count >= required_evidence_count)
        & (distance_from_core <= max_distance_from_core)
        & (core_density >= min_core_density)
    )
    return filter_components_by_area(core_mask | medium_pixel_mask, min_component_area)


def create_blue_overlay(image_bgr: np.ndarray, class_map: np.ndarray, alpha: float = 0.45) -> Image.Image:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = rgb.astype(np.float32)
    high_mask = class_map == 2
    blue = np.array([30, 120, 255], dtype=np.float32)
    result[high_mask] = (1 - alpha) * result[high_mask] + alpha * blue
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def process_talc_concentration(image: Image.Image):
    image_bgr = pil_to_bgr(image)
    dark_score = calculate_dark_score(image_bgr, local_window=51)
    density = calculate_inclusion_density(image_bgr, dark_score)
    class_map_raw, class_centers = segment_density_by_similarity(density)
    class_map = refine_concentration_boundaries(density, class_map_raw, class_centers)
    talc_result = detect_talc_pixels(image_bgr, class_map)
    talc_pixel_mask = add_constrained_medium_pixels(image_bgr, talc_result)
    talc_percentage_total = float(np.count_nonzero(talc_pixel_mask) / talc_pixel_mask.size * 100)
    return {
        "image": create_blue_overlay(image_bgr, class_map),
        "class_map": class_map,
        "talc_pixel_mask": talc_pixel_mask,
        "talc_percentage_total": talc_percentage_total,
    }


@dataclass
class SulfideParameters:
    blackhat_sizes: tuple = (7, 13, 25, 51)
    blackhat_threshold: float = 8.0
    bright_threshold: float = 100.0
    density_window: int = 81
    min_region_area: int = 5000


def ellipse(size):
    size = max(3, int(size)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def fill_internal_holes(mask):
    mask = mask.astype(bool)
    count, labels, _, _ = cv2.connectedComponentsWithStats((~mask).astype(np.uint8), connectivity=8)
    border_labels = np.unique(np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]))
    internal_labels = np.ones(count, dtype=bool)
    internal_labels[border_labels] = False
    internal_labels[0] = False
    return mask | internal_labels[labels]


def density_hysteresis_mask(structure_density, low_threshold=0.80, high_threshold=0.85, min_region_area=5000):
    weak_mask = structure_density >= low_threshold
    strong_mask = structure_density >= high_threshold
    count, labels, stats, _ = cv2.connectedComponentsWithStats(weak_mask.astype(np.uint8), connectivity=8)
    accepted_labels = np.zeros(count, dtype=bool)
    accepted_labels[np.unique(labels[strong_mask])] = True
    accepted_labels[0] = False
    accepted_labels &= stats[:, cv2.CC_STAT_AREA] >= min_region_area
    return accepted_labels[labels]


def detect_light_region_with_inclusions(image_bgr, params=None):
    params = params or SulfideParameters()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blackhat = np.maximum.reduce([
        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, ellipse(size)) for size in params.blackhat_sizes
    ])
    structure_pixels = (gray >= params.bright_threshold) | (blackhat >= params.blackhat_threshold)
    density_sigma = params.density_window / 6.0
    structure_density = cv2.GaussianBlur(
        structure_pixels.astype(np.float32), (0, 0), sigmaX=density_sigma, sigmaY=density_sigma, borderType=cv2.BORDER_REFLECT
    )
    region = density_hysteresis_mask(structure_density, min_region_area=params.min_region_area)
    region = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_CLOSE, ellipse(15)).astype(bool)
    return fill_internal_holes(region)


def detect_dark_inclusions_in_region(
    image_bgr,
    region_mask,
    otsu_classes=3,
    dark_classes=1,
    boundary_margin=5,
    min_area=3,
    max_area=None,
    ring_width=7,
    min_ring_contrast=20,
    close_size=3,
    use_clahe=True,
):
    region_mask = region_mask.astype(bool)
    if boundary_margin > 0:
        size = 2 * boundary_margin + 1
        inner_region = cv2.erode(region_mask.astype(np.uint8), ellipse(size)).astype(bool)
    else:
        inner_region = region_mask.copy()
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    if use_clahe:
        lightness_for_segmentation = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    else:
        lightness_for_segmentation = lightness
    roi_values = lightness_for_segmentation[inner_region]
    if roi_values.size == 0:
        return np.zeros_like(region_mask, dtype=bool)
    thresholds = safe_multiotsu(roi_values, classes=otsu_classes)
    if not 1 <= dark_classes < otsu_classes:
        raise ValueError("dark_classes должен быть от 1 до otsu_classes - 1")
    dark_candidates = (lightness_for_segmentation <= thresholds[dark_classes - 1]) & inner_region
    if close_size > 1:
        dark_candidates = cv2.morphologyEx(dark_candidates.astype(np.uint8), cv2.MORPH_CLOSE, ellipse(close_size)).astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark_candidates.astype(np.uint8), connectivity=8)
    inclusions_mask = np.zeros_like(dark_candidates, dtype=bool)
    ring_kernel = ellipse(2 * ring_width + 1)
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        if area < min_area or (max_area is not None and area > max_area):
            continue
        padding = ring_width + 1
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(labels.shape[1], x + width + padding), min(labels.shape[0], y + height + padding)
        component_crop = labels[y0:y1, x0:x1] == label
        region_crop = inner_region[y0:y1, x0:x1]
        lightness_crop = lightness[y0:y1, x0:x1]
        ring = cv2.dilate(component_crop.astype(np.uint8), ring_kernel).astype(bool) & ~component_crop & region_crop
        if not np.any(ring):
            continue
        contrast = float(np.mean(lightness_crop[ring]) - np.mean(lightness_crop[component_crop]))
        if contrast >= min_ring_contrast:
            inclusions_mask[y0:y1, x0:x1][component_crop] = True
    return inclusions_mask


def process_sulfide_inclusions(image: Image.Image):
    image_bgr = pil_to_bgr(image)
    region_mask = detect_light_region_with_inclusions(image_bgr)
    dark_inclusions_mask = detect_dark_inclusions_in_region(image_bgr, region_mask)
    dark_inclusions_inside = dark_inclusions_mask & region_mask
    region_area = int(np.count_nonzero(region_mask))
    inclusions_area = int(np.count_nonzero(dark_inclusions_inside))
    percentage = inclusions_area / region_area * 100 if region_area > 0 else 0.0
    mask_image = Image.fromarray((dark_inclusions_inside.astype(np.uint8) * 255), mode="L").convert("RGB")
    return {
        "image": mask_image,
        "region_mask": region_mask,
        "dark_inclusions_mask": dark_inclusions_inside,
        "dark_inclusions_percentage": float(percentage),
    }
