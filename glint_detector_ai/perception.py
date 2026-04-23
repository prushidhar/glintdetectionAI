from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from glint_detector_ai.config import DetectionConfig
from glint_detector_ai.models import FrameAnalysis, GlintDetection


class BaseGlintDetector(ABC):
    backend_name = "base"

    def __init__(self, config: DetectionConfig) -> None:
        self.config = config

    @abstractmethod
    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> tuple[FrameAnalysis, np.ndarray]:
        raise NotImplementedError


class BrightSpotDetector(BaseGlintDetector):
    """Detect compact bright reflections that may indicate a camera lens glint."""

    backend_name = "heuristic"

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> tuple[FrameAnalysis, np.ndarray]:
        processed_frame, scale = self._resize_for_processing(frame)
        grayscale = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        grayscale = self._enhance_contrast(grayscale)
        blurred = cv2.GaussianBlur(
            grayscale,
            (self.config.blur_kernel_size, self.config.blur_kernel_size),
            0,
        )

        _, mask = cv2.threshold(
            blurred,
            self.config.brightness_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        kernel = np.ones(
            (self.config.morphology_kernel_size, self.config.morphology_kernel_size),
            dtype=np.uint8,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[GlintDetection] = []
        scaled_min_area = self.config.min_area * scale * scale
        scaled_max_area = self.config.max_area * scale * scale
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < scaled_min_area or area > scaled_max_area:
                continue

            _, _, width, height = cv2.boundingRect(contour)
            shortest_side = max(1, min(width, height))
            aspect_ratio = max(width, height) / shortest_side
            if aspect_ratio > self.config.max_aspect_ratio:
                continue

            perimeter = cv2.arcLength(contour, closed=True)
            circularity = self._compute_circularity(area, perimeter)
            if circularity < self.config.circularity_threshold:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])

            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
            intensity = float(cv2.mean(grayscale, mask=contour_mask)[0])

            detections.append(
                GlintDetection(
                    x=int(round(center_x / scale)),
                    y=int(round(center_y / scale)),
                    area=area / max(scale * scale, 1e-6),
                    intensity=intensity,
                    circularity=circularity,
                    confidence=self._heuristic_confidence(intensity, circularity),
                )
            )

        display_mask = mask
        if scale != 1.0:
            display_mask = cv2.resize(
                mask,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        analysis = FrameAnalysis(
            frame_index=frame_index,
            timestamp=datetime.now(),
            detections=detections,
            max_intensity=max((detection.intensity for detection in detections), default=0.0),
            backend=self._backend_label(scale),
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        )
        return analysis, display_mask

    @staticmethod
    def _compute_circularity(area: float, perimeter: float) -> float:
        if perimeter <= 0:
            return 0.0
        return float((4.0 * np.pi * area) / (perimeter * perimeter))

    def _heuristic_confidence(self, intensity: float, circularity: float) -> float:
        span = max(1.0, 255.0 - float(self.config.brightness_threshold))
        brightness_score = max(
            0.0,
            min(1.0, (intensity - self.config.brightness_threshold) / span),
        )
        return max(0.0, min(1.0, (brightness_score * 0.75) + (circularity * 0.25)))

    def _resize_for_processing(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        scale = min(1.0, max(0.2, float(self.config.processing_scale)))
        if scale == 1.0:
            return frame, 1.0
        resized = cv2.resize(
            frame,
            (max(1, int(frame.shape[1] * scale)), max(1, int(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _enhance_contrast(self, grayscale: np.ndarray) -> np.ndarray:
        if not self.config.enable_clahe:
            return grayscale
        tile_size = max(2, int(self.config.clahe_tile_grid_size))
        clahe = cv2.createCLAHE(
            clipLimit=float(self.config.clahe_clip_limit),
            tileGridSize=(tile_size, tile_size),
        )
        return clahe.apply(grayscale)

    def _backend_label(self, scale: float) -> str:
        suffixes: list[str] = []
        if self.config.enable_clahe:
            suffixes.append("clahe")
        if scale != 1.0:
            suffixes.append(f"{scale:.2f}x")
        if not suffixes:
            return self.backend_name
        return f"{self.backend_name}+{'+'.join(suffixes)}"


class YoloGlintDetector(BaseGlintDetector):
    """Optional YOLO-based detector hook for future trained-model integration."""

    backend_name = "yolo"

    def __init__(self, config: DetectionConfig) -> None:
        super().__init__(config)
        if not config.yolo_model_path:
            raise RuntimeError(
                "YOLO backend selected but no model path was provided. "
                "Use --yolo-model-path path/to/model.pt."
            )
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YOLO backend requires the optional 'ultralytics' package. "
                "Install it with 'pip install ultralytics'."
            ) from exc

        self._target_labels = {label.casefold() for label in config.yolo_target_labels}
        self._model = YOLO(config.yolo_model_path)
        self._model_name = Path(config.yolo_model_path).stem

    def analyze_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
    ) -> tuple[FrameAnalysis, np.ndarray]:
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        results = self._model.predict(
            source=frame,
            conf=self.config.yolo_confidence_threshold,
            verbose=False,
        )

        detections: list[GlintDetection] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_index = int(box.cls[0].item()) if box.cls is not None else -1
                label = self._resolve_label(names, cls_index)
                if self._target_labels and label.casefold() not in self._target_labels:
                    continue

                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                region = grayscale[y1:y2, x1:x2]
                intensity = float(np.mean(region)) if region.size else 0.0
                width = x2 - x1
                height = y2 - y1
                area = float(width * height)
                confidence = float(box.conf[0].item()) if box.conf is not None else 0.0

                detections.append(
                    GlintDetection(
                        x=(x1 + x2) // 2,
                        y=(y1 + y2) // 2,
                        area=area,
                        intensity=intensity,
                        circularity=self._bbox_circularity(width, height),
                        label=label,
                        confidence=confidence,
                    )
                )
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)

        analysis = FrameAnalysis(
            frame_index=frame_index,
            timestamp=datetime.now(),
            detections=detections,
            max_intensity=max((detection.intensity for detection in detections), default=0.0),
            backend=f"{self.backend_name}:{self._model_name}",
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
        )
        return analysis, mask

    @staticmethod
    def _bbox_circularity(width: int, height: int) -> float:
        perimeter = max(1.0, float((width * 2) + (height * 2)))
        area = max(1.0, float(width * height))
        return min(1.0, (4.0 * np.pi * area) / (perimeter * perimeter))

    @staticmethod
    def _resolve_label(names: object, cls_index: int) -> str:
        if isinstance(names, dict):
            return str(names.get(cls_index, f"class_{cls_index}"))
        if isinstance(names, list) and 0 <= cls_index < len(names):
            return str(names[cls_index])
        return f"class_{cls_index}"


def build_detector(config: DetectionConfig) -> BaseGlintDetector:
    backend = config.backend.casefold()
    if backend == "heuristic":
        return BrightSpotDetector(config)
    if backend == "yolo":
        return YoloGlintDetector(config)
    raise ValueError(
        f"Unsupported detector backend '{config.backend}'. "
        "Expected one of: heuristic, yolo."
    )
