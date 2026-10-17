from __future__ import annotations

import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
from uniface import face_alignment


def setup_cuda_dlls():
    paths_to_add = []
    for p in sys.path:
        if "site-packages" in p:
            cudnn = os.path.join(p, "nvidia", "cudnn", "bin")
            cublas = os.path.join(p, "nvidia", "cublas", "bin")
            if os.path.exists(cudnn):
                paths_to_add.append(cudnn)
            if os.path.exists(cublas):
                paths_to_add.append(cublas)

    for path in paths_to_add:
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


setup_cuda_dlls()


class BaseFaceParser:
    """
    Base class for Face Parsing / Segmentation models using ONNX Runtime.
    Provides common landmark-based face alignment, session management,
    and inverse affine warping back to original image space.
    """

    def __init__(
        self,
        model_path: str,
        align_size: int = 512,
        blur_sigma: float = 0,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.align_size = align_size
        self.blur_sigma = blur_sigma
        self.providers = providers
        self._cuda_setting = (
            providers is not None and "CUDAExecutionProvider" in providers
        )
        self.session = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        if self.providers is None:
            available = ort.get_available_providers()
            self.providers = []
            if "CUDAExecutionProvider" in available:
                self.providers.append("CUDAExecutionProvider")
            self.providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # Errors and Fatal only

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=self.providers,
        )

        input_cfg = self.session.get_inputs()[0]
        self.input_name = input_cfg.name
        self.input_shape = input_cfg.shape
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def postprocess(
        self,
        outputs: list[np.ndarray],
        crop_size: tuple[int, int],
        face_crop: np.ndarray | None = None,
        include_features: bool = True,
        include_neck: bool = False,
        exclude_tongue: bool = True,
    ) -> np.ndarray:
        raise NotImplementedError

    def parse(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        include_features: bool = True,
        include_neck: bool = False,
        exclude_tongue: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        Align face using landmarks, run inference, and warp mask back to image space.

        Args:
            image (np.ndarray): Original BGR image (H, W, 3).
            landmarks (np.ndarray): 5-point facial landmarks (5, 2).
            include_features (bool): Include eyes, nose, lips in face contour mask.
            include_neck (bool): Include neck in the mask.
            exclude_tongue (bool): Mask out inner mouth, teeth, and tongue.

        Returns:
            np.ndarray: Float32 mask with values in [0, 1] in full image space (H, W).
        """
        if landmarks.shape != (5, 2):
            raise ValueError(f"Landmarks must have shape (5, 2), got {landmarks.shape}")

        face_crop, inverse_matrix = face_alignment(
            image, landmarks, image_size=self.align_size
        )
        crop_size = (face_crop.shape[1], face_crop.shape[0])

        input_tensor = self.preprocess(face_crop)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        mask = self.postprocess(
            outputs,
            crop_size,
            face_crop=face_crop,
            include_features=include_features,
            include_neck=include_neck,
            exclude_tongue=exclude_tongue,
        )

        h, w = image.shape[:2]
        warped_mask = cv2.warpAffine(
            mask,
            inverse_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped_mask
