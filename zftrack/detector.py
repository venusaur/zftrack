"""Blob detection: turn a frame into a list of candidate fish.

The fish are darker than the background, so we use the *signed* difference
``background - frame`` (clipped at zero). This is more specific than an absolute
difference: it responds only to things that got darker than the empty arena
(i.e. a fish moving in) and ignores brighter artefacts such as glints.

Each detection also carries body-shape information (orientation, length, the two
body-axis tips and how dark each tip is) and a small appearance descriptor used
for re-identifying a fish after it is briefly lost.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    """A single detected blob in one frame."""

    centroid: tuple[float, float]
    bbox: tuple[int, int, int, int]  # x, y, w, h
    area: float
    contour: np.ndarray  # Nx1x2 int32, for drawing the outline

    # Body shape / orientation.
    length: float = 0.0          # extent along the principal (body) axis
    width: float = 0.0           # extent along the secondary axis
    axis_deg: float = 0.0        # undirected body-axis angle, 0-180
    end_a: tuple[float, float] = (0.0, 0.0)  # one body-axis tip
    end_b: tuple[float, float] = (0.0, 0.0)  # the other tip
    end_a_intensity: float = 255.0  # mean gray near end_a (lower = darker)
    end_b_intensity: float = 255.0  # mean gray near end_b

    descriptor: np.ndarray | None = None  # appearance vector for re-ID


def _patch_intensity(gray: np.ndarray, x: float, y: float, r: int = 4) -> float:
    """Mean grayscale intensity in a small square around (x, y)."""
    h, w = gray.shape
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 255.0
    return float(gray[y0:y1, x0:x1].mean())


def _shape_features(contour: np.ndarray):
    """Principal-axis orientation, length/width and the two body tips."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    if len(pts) < 2:
        cx, cy = float(pts[0, 0]), float(pts[0, 1])
        return 0.0, 0.0, 0.0, (cx, cy), (cx, cy)

    mean, eigvecs, _ = cv2.PCACompute2(pts, mean=None)
    axis = eigvecs[0]
    secondary = eigvecs[1]
    centered = pts - mean

    t = centered @ axis              # projection onto the body axis
    s = centered @ secondary         # projection onto the secondary axis
    length = float(t.max() - t.min())
    width = float(s.max() - s.min())
    axis_deg = float(np.degrees(np.arctan2(axis[1], axis[0])) % 180.0)

    end_a = (float(mean[0, 0] + axis[0] * t.max()),
             float(mean[0, 1] + axis[1] * t.max()))
    end_b = (float(mean[0, 0] + axis[0] * t.min()),
             float(mean[0, 1] + axis[1] * t.min()))
    return length, width, axis_deg, end_a, end_b


class BlobDetector:
    """Foreground blob detector against a fixed median background.

    Parameters
    ----------
    background:
        Grayscale background image (see :func:`zftrack.build_background`).
    threshold:
        Pixels darker than the background by more than this (0-255) are
        foreground. Lower = more sensitive.
    min_area, max_area:
        Accepted blob area range in pixels. ``min_area`` rejects debris/noise;
        ``max_area`` rejects large artefacts (and can let merged fish pairs
        through if set generously).
    blur:
        Gaussian blur kernel (odd) applied to the difference image to suppress
        speckle before thresholding.
    close_ksize, open_ksize:
        Morphological closing then opening kernel sizes. Closing fills gaps in a
        fish body; opening removes isolated specks.
    """

    def __init__(
        self,
        background: np.ndarray,
        threshold: int = 13,
        min_area: float = 200.0,
        max_area: float = 6000.0,
        blur: int = 5,
        close_ksize: int = 7,
        open_ksize: int = 3,
    ) -> None:
        self.background = background
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.blur = blur if blur % 2 == 1 else blur + 1
        self._close_kernel = np.ones((close_ksize, close_ksize), np.uint8)
        self._open_kernel = np.ones((open_ksize, open_ksize), np.uint8)

    def foreground_mask(self, gray: np.ndarray) -> np.ndarray:
        """Return the binary foreground mask for a grayscale frame."""
        diff = cv2.subtract(self.background, gray)
        if self.blur > 1:
            diff = cv2.GaussianBlur(diff, (self.blur, self.blur), 0)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open_kernel)
        return mask

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect fish-like blobs in a BGR frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self.foreground_mask(gray)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area <= area <= self.max_area):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                cx, cy = x + w / 2.0, y + h / 2.0
            else:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]

            length, width, axis_deg, end_a, end_b = _shape_features(contour)
            ia = _patch_intensity(gray, *end_a)
            ib = _patch_intensity(gray, *end_b)

            # Appearance descriptor: coarse size/shape/intensity signature.
            mean_intensity = _patch_intensity(gray, cx, cy, r=6)
            descriptor = np.array(
                [area, length, width, mean_intensity], dtype=np.float32
            )

            detections.append(
                Detection(
                    centroid=(cx, cy),
                    bbox=(x, y, w, h),
                    area=area,
                    contour=contour,
                    length=length,
                    width=width,
                    axis_deg=axis_deg,
                    end_a=end_a,
                    end_b=end_b,
                    end_a_intensity=ia,
                    end_b_intensity=ib,
                    descriptor=descriptor,
                )
            )
        return detections
