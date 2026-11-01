"""Minimal, inspectable camera-frame contract and synthetic target cue.

The red-blob detector is only a pipeline smoke example. It is not a robust
object detector and is not used by the balance policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    frame_id: str
    width: int
    height: int
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    @classmethod
    def from_vertical_fov(
        cls, *, frame_id: str, width: int, height: int, fovy_deg: float
    ) -> CameraCalibration:
        fy = height / (2.0 * np.tan(np.deg2rad(fovy_deg) / 2.0))
        return cls(frame_id, width, height, float(fy), float(fy), width / 2.0, height / 2.0)


@dataclass(frozen=True, slots=True)
class RGBFrame:
    timestamp_s: float
    calibration: CameraCalibration
    rgb: np.ndarray

    def __post_init__(self) -> None:
        expected = (self.calibration.height, self.calibration.width, 3)
        if self.rgb.shape != expected or self.rgb.dtype != np.uint8:
            raise ValueError(
                f"RGB image must be uint8 with shape {expected}, got {self.rgb.dtype} {self.rgb.shape}"
            )


@dataclass(frozen=True, slots=True)
class VisualCue:
    label: str
    normalized_x: float
    normalized_y: float
    area_fraction: float
    confidence: float
    frame_id: str
    timestamp_s: float


def detect_red_target(frame: RGBFrame, *, min_pixels: int = 12) -> VisualCue | None:
    """Return a red-dominant color blob centroid, or None when not visible."""
    rgb = frame.rgb.astype(np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mask = (red > 110) & (red > green * 1.55) & (red > blue * 1.45)
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None
    height, width = mask.shape
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    area = len(xs) / float(width * height)
    color_margin = np.clip((red[mask] - np.maximum(green[mask], blue[mask])) / 255.0, 0.0, 1.0)
    confidence = float(
        np.clip(color_margin.mean() * min(1.0, len(xs) / (min_pixels * 8)), 0.0, 1.0)
    )
    return VisualCue(
        label="red_target",
        normalized_x=(centroid_x - frame.calibration.cx_px) / max(1.0, width / 2.0),
        normalized_y=(centroid_y - frame.calibration.cy_px) / max(1.0, height / 2.0),
        area_fraction=area,
        confidence=confidence,
        frame_id=frame.calibration.frame_id,
        timestamp_s=frame.timestamp_s,
    )
