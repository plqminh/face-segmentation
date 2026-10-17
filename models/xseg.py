from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort
from uniface import face_alignment


class XSeg:
    """
    XSeg: Face Occlusion Segmentation Model with ONNX Runtime.

    Standalone implementation without inheritance. XSeg outputs a binary soft mask
    indicating face regions. It requires 5-point facial landmarks for proper face
    alignment before inference.

    The model uses NHWC input format and outputs values in [0, 1] range.

    Args:
        model_path (str): Path to XSeg ONNX model file.
        align_size (int): Face alignment output size. Must be multiple of 112 or 128.
            Defaults to 256.
        blur_sigma (float): Gaussian blur sigma for mask smoothing.
            0 = raw output (no blur). Defaults to 0.
        providers (list[str] | None): ONNX Runtime execution providers. If None,
            auto-detects the best available provider.

    Attributes:
        align_size (int): Face alignment output size.
        blur_sigma (float): Blur sigma for post-processing.
        input_size (tuple[int, int]): Model input dimensions (width, height).

    Example:
        >>> from uniface import RetinaFace
        >>> detector = RetinaFace()
        >>> parser = XSeg('xseg.onnx', blur_sigma=5)
        >>>
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     if face.landmarks is not None:
        ...         mask = parser.parse(image, face.landmarks)
    """

    def __init__(
        self,
        model_path: str,
        align_size: int = 256,
        blur_sigma: float = 0,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.align_size = align_size
        self.blur_sigma = blur_sigma
        self.providers = providers

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the ONNX model from the stored model path."""
        # Auto-detect providers if not specified
        if self.providers is None:
            available = ort.get_available_providers()
            self.providers = []
            if "CoreMLExecutionProvider" in available:
                self.providers.append("CoreMLExecutionProvider")
            if "CUDAExecutionProvider" in available:
                self.providers.append("CUDAExecutionProvider")
            self.providers.append("CPUExecutionProvider")

        # Create session
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3  # Only ERROR and FATAL

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=self.providers,
        )

        # Get input configuration
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        self.input_name = input_cfg.name

        # NHWC format: (N, H, W, C) - height at index 1, width at index 2
        if isinstance(input_shape[1], int) and isinstance(input_shape[2], int):
            self.input_size = (input_shape[2], input_shape[1])  # (width, height)
        else:
            self.input_size = (256, 256)  # Default for dynamic shapes

        # Get output configuration
        outputs = self.session.get_outputs()
        self.output_names = [output.name for output in outputs]

    def preprocess(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Preprocess an aligned face crop for XSeg inference.

        Args:
            face_crop (np.ndarray): An aligned face crop in BGR format.

        Returns:
            np.ndarray: Preprocessed image tensor with shape (1, H, W, 3) - NHWC format.
        """
        # Resize to model input size
        image = cv2.resize(face_crop, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Keep NHWC format (batch, height, width, channels)
        image = np.expand_dims(image, axis=0)

        return image

    def postprocess(self, outputs: np.ndarray, crop_size: tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output to segmentation mask.

        Args:
            outputs (np.ndarray): Raw model output (NHWC format).
            crop_size (tuple[int, int]): Size to resize mask to (width, height).

        Returns:
            np.ndarray: Segmentation mask as float32 array in range [0, 1].
        """
        # Squeeze and clip to [0, 1]
        mask = outputs.squeeze().clip(0, 1).astype(np.float32)

        # Resize to crop size
        mask = cv2.resize(mask, crop_size, interpolation=cv2.INTER_LINEAR)

        # Apply optional Gaussian blur for smoothing
        if self.blur_sigma > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), self.blur_sigma)
            # Threshold and normalize
            # clip to [0.5, 1], subtract 0.5, multiply by 2 -> range [0, 1]
            mask = (mask.clip(0.5, 1) - 0.5) * 2

        return mask

    def parse(self, image: np.ndarray, landmarks: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform face segmentation using 5-point landmarks for alignment.

        This method aligns the face using landmarks, runs XSeg inference,
        and warps the mask back to the original image space.

        Args:
            image (np.ndarray): Input image in BGR format.
            landmarks (np.ndarray): 5-point facial landmarks with shape (5, 2).

        Returns:
            np.ndarray: Segmentation mask in original image space with shape (H, W),
                        values in range [0, 1].
        """
        if landmarks.shape != (5, 2):
            raise ValueError(f"Landmarks must have shape (5, 2), got {landmarks.shape}")

        # Align face using landmarks
        face_crop, inverse_matrix = face_alignment(image, landmarks, image_size=self.align_size)

        # Run inference
        crop_size = (face_crop.shape[1], face_crop.shape[0])  # (width, height)
        input_tensor = self.preprocess(face_crop)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess mask
        mask = self.postprocess(outputs[0], crop_size)

        # Warp mask back to original image space
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

    def parse_aligned(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Perform face segmentation on an already aligned face crop.

        Args:
            face_crop (np.ndarray): An aligned face crop in BGR format.

        Returns:
            np.ndarray: Segmentation mask with same size as input crop,
                        values in range [0, 1].
        """
        crop_size = (face_crop.shape[1], face_crop.shape[0])
        input_tensor = self.preprocess(face_crop)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        return self.postprocess(outputs[0], crop_size)

    def parse_with_inverse(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse face and return mask with alignment matrices for custom warping.

        Args:
            image (np.ndarray): Input image in BGR format.
            landmarks (np.ndarray): 5-point facial landmarks with shape (5, 2).

        Returns:
            tuple containing:
                - mask (np.ndarray): Segmentation mask in crop space.
                - face_crop (np.ndarray): Aligned face crop.
                - inverse_matrix (np.ndarray): Inverse affine matrix (2, 3) for warping back.
        """
        if landmarks.shape != (5, 2):
            raise ValueError(f"Landmarks must have shape (5, 2), got {landmarks.shape}")

        # Align face using landmarks
        face_crop, inverse_matrix = face_alignment(image, landmarks, image_size=self.align_size)

        # Run inference
        crop_size = (face_crop.shape[1], face_crop.shape[0])
        input_tensor = self.preprocess(face_crop)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess mask (in crop space)
        mask = self.postprocess(outputs[0], crop_size)

        return mask, face_crop, inverse_matrix

    def __call__(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Callable shortcut for the parse method."""
        return self.parse(image, landmarks)
