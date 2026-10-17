from __future__ import annotations

import cv2
import numpy as np
from .base import BaseFaceParser


class BiSeNetParser(BaseFaceParser):
    """
    BiSeNet Face Parsing Model with ONNX Runtime.
    Supports ResNet-18 and ResNet-34 backbones trained on CelebAMask-HQ.
    Input format: NCHW (1, 3, 512, 512) normalized with ImageNet mean/std.
    Output: 19 semantic facial classes.
    """

    SKIN_CLASS = 1
    FEATURE_CLASSES = [2, 3, 4, 5, 6, 10, 12, 13]  # brows, eyes, glasses, nose, upper/lower lips
    INNER_MOUTH_CLASS = 11  # oral cavity, teeth, tongue
    NECK_CLASS = 14

    def __init__(
        self,
        model_path: str,
        align_size: int = 512,
        blur_sigma: float = 0,
        providers: list[str] | None = None,
    ) -> None:
        super().__init__(
            model_path=model_path,
            align_size=align_size,
            blur_sigma=blur_sigma,
            providers=providers,
        )
        self.input_h = 512
        self.input_w = 512
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_crop, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - self.mean) / self.std
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)

    def postprocess(
        self,
        outputs: list[np.ndarray],
        crop_size: tuple[int, int],
        face_crop: np.ndarray | None = None,
        include_features: bool = True,
        include_neck: bool = False,
        exclude_tongue: bool = True,
    ) -> np.ndarray:
        logits = outputs[0]  # shape (1, 19, 512, 512)
        pred_map = np.argmax(logits, axis=1)[0]  # shape (512, 512)

        # Build mask of target classes
        target_classes = [self.SKIN_CLASS]
        if include_features:
            target_classes.extend(self.FEATURE_CLASSES)
            if not exclude_tongue:
                target_classes.append(self.INNER_MOUTH_CLASS)
        if include_neck:
            target_classes.append(self.NECK_CLASS)

        # Binary mask in crop space
        mask = np.isin(pred_map, target_classes).astype(np.float32)

        # When exclude_tongue is True, ensure mouth interior / tongue is strictly excluded
        if exclude_tongue:
            mask[pred_map == self.INNER_MOUTH_CLASS] = 0.0

        # Resize to crop size if different from 512x512
        if crop_size != (self.input_w, self.input_h):
            mask = cv2.resize(mask, crop_size, interpolation=cv2.INTER_LINEAR)

        if self.blur_sigma > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), self.blur_sigma)
            mask = np.clip(mask, 0.0, 1.0)

        return mask
