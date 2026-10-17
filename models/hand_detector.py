from __future__ import annotations

import os
import cv2
import numpy as np


class MPPalmDet:
    """
    OpenCV Zoo MediaPipe Palm Detector using OpenCV DNN.
    Detects hands/palms in an image with bounding boxes and 7 keypoints.
    """

    def __init__(
        self,
        model_path: str,
        nms_threshold: float = 0.3,
        score_threshold: float = 0.5,
        top_k: int = 5000,
    ) -> None:
        self.model_path = model_path
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.input_size = np.array([192, 192])

        self.model = cv2.dnn.readNet(self.model_path)
        self.anchors = self._load_anchors()

    def _load_anchors(self):
        # MediaPipe Palm Detector anchors (exactly 2016 anchors for 192x192)
        anchors = []
        for stride, count in [(8, 2), (16, 6)]:
            grid_size = 192 // stride
            for y in range(grid_size):
                for x in range(grid_size):
                    for _ in range(count):
                        anchors.append([(x + 0.5) / grid_size, (y + 0.5) / grid_size])
        return np.array(anchors, dtype=np.float32)

    def _preprocess(self, image: np.ndarray):
        h, w = image.shape[:2]
        scale = max(h, w)
        pad_image = np.zeros((scale, scale, 3), dtype=np.uint8)
        pad_image[:h, :w] = image

        blob = cv2.dnn.blobFromImage(
            pad_image,
            scalefactor=1.0 / 255.0,
            size=(192, 192),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )
        # OpenCV palm detector expects NHWC or NCHW depending on version
        # cv2.dnn handles blob format
        return blob, scale

    def infer(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = max(h, w)

        # Create padded square image
        pad_image = np.zeros((scale, scale, 3), dtype=np.uint8)
        pad_image[:h, :w] = image

        # Resized for model input (192, 192)
        resized = cv2.resize(pad_image, (192, 192)).astype(np.float32) / 255.0
        # Convert to RGB NHWC for model
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_blob = np.expand_dims(rgb, axis=0)

        self.model.setInput(input_blob)
        output_names = self.model.getUnconnectedOutLayersNames()
        output_blob = self.model.forward(output_names)

        return self._postprocess(output_blob, scale)

    def _postprocess(self, output_blob, scale: float) -> np.ndarray:
        # output_blob[0]: raw_boxes (1, 2016, 18), output_blob[1]: raw_scores (1, 2016, 1)
        raw_boxes = output_blob[0][0]
        raw_scores = output_blob[1][0, :, 0]

        # Sigmoid for scores
        scores = 1.0 / (1.0 + np.exp(-raw_scores.astype(np.float64)))
        valid_mask = scores > self.score_threshold

        if not np.any(valid_mask):
            return np.empty((0, 19))

        scores = scores[valid_mask]
        box_delta = raw_boxes[valid_mask, :4]
        landmark_delta = raw_boxes[valid_mask, 4:]
        anchors = self.anchors[valid_mask]

        # Decode boxes
        cxy = (box_delta[:, :2] / 192.0 + anchors) * scale
        wh = (box_delta[:, 2:] / 192.0) * scale
        xy1 = cxy - wh / 2.0
        xy2 = cxy + wh / 2.0
        boxes = np.hstack([xy1, xy2])

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.score_threshold, self.nms_threshold
        )
        if len(indices) == 0:
            return np.empty((0, 19))

        indices = np.array(indices).flatten()
        selected_boxes = boxes[indices]
        selected_scores = scores[indices]

        # Decode landmarks
        selected_landmarks = (
            landmark_delta[indices].reshape(-1, 7, 2) / 192.0 + anchors[indices, np.newaxis, :]
        ) * scale

        return np.c_[
            selected_boxes,
            selected_landmarks.reshape(-1, 14),
            selected_scores.reshape(-1, 1),
        ]


class MPHandPose:
    """
    OpenCV Zoo MediaPipe Hand Pose Estimator using OpenCV DNN.
    Estimates 21 3D hand keypoints per detected palm.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.input_size = np.array([224, 224])
        self.model = cv2.dnn.readNet(self.model_path)

    def infer(self, image: np.ndarray, palm: np.ndarray):
        h, w = image.shape[:2]
        palm_bbox = palm[:4].reshape(2, 2)
        palm_landmarks = palm[4:18].reshape(7, 2)

        # Center and size of palm
        center = np.mean(palm_bbox, axis=0)
        box_size = np.max(palm_bbox[1] - palm_bbox[0]) * 2.6
        box_size = max(box_size, 40)

        # Rotation from wrist to middle finger base (landmarks 0 and 2)
        p1 = palm_landmarks[0]
        p2 = palm_landmarks[2]
        radians = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        angle = np.rad2deg(radians)

        rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        # Shift to center in (224, 224)
        rot_mat[0, 2] += 112 - center[0]
        rot_mat[1, 2] += 112 - center[1]

        scale_factor = 224.0 / box_size
        scale_mat = cv2.getRotationMatrix2D((112, 112), 0, scale_factor)
        combined_mat = scale_mat @ np.vstack([rot_mat, [0, 0, 1]])

        crop = cv2.warpAffine(
            image, combined_mat[:2], (224, 224), flags=cv2.INTER_LINEAR
        )
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        input_blob = np.expand_dims(rgb_crop, axis=0)

        self.model.setInput(input_blob)
        output_names = self.model.getUnconnectedOutLayersNames()
        out = self.model.forward(output_names)

        # out[0]: landmarks (1, 63), out[1]: confidence (1, 1)
        conf = float(out[1][0][0]) if len(out) > 1 else 1.0
        if conf < self.conf_threshold:
            return None

        raw_landmarks = out[0][0].reshape(21, 3)[:, :2]  # (21, 2) in 224x224 coords

        # Invert affine transform back to original image space
        inv_mat = cv2.invertAffineTransform(combined_mat[:2])
        pts_homo = np.hstack([raw_landmarks, np.ones((21, 1), dtype=np.float32)])
        landmarks_orig = (pts_homo @ inv_mat.T)[:, :2]

        return landmarks_orig


class SelfieMulticlassSeg:
    """
    MediaPipe Multiclass Selfie Segmentation in ONNX.
    Provides dense pixel-accurate segmentation for:
    Class 2: Body-skin (Hands, arms, fingers)
    """

    def __init__(
        self,
        model_path: str = "weights/selfie_multiclass.onnx",
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = None
        self._initialize()

    def _initialize(self):
        if not os.path.exists(self.model_path):
            self._download_weights()
        if os.path.exists(self.model_path):
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                self.session = ort.InferenceSession(
                    self.model_path, opts, providers=self.providers
                )
            except Exception as e:
                print(f"Warning: Failed to load {self.model_path}: {e}")

    def _download_weights(self):
        url = "https://huggingface.co/FreeHugsForRobots/ps-skin-seg/resolve/main/selfie_multiclass_256x256.onnx"
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        try:
            import urllib.request

            urllib.request.urlretrieve(url, self.model_path)
        except Exception as e:
            print(f"Warning: Failed to download {self.model_path}: {e}")

    @property
    def is_available(self) -> bool:
        return self.session is not None

    def segment(self, image: np.ndarray) -> np.ndarray | None:
        if not self.is_available:
            return None
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        blob = np.expand_dims(resized, axis=0)

        input_name = self.session.get_inputs()[0].name
        out = self.session.run(None, {input_name: blob})[0]
        logits = out[0]
        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)


class YOLOHandSegmenter:
    """
    Ultralytics YOLO Hand Instance Segmentation in ONNX.
    Trained on the HaGRID hand gesture dataset with 32 prototype masks.
    Detects hands and generates instance segmentation masks for fingers and palms.
    """

    def __init__(
        self,
        model_path: str = "weights/yolo26m-seg-hand.onnx",
        providers: list[str] | None = None,
        conf_threshold: float = 0.35,
    ) -> None:
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]
        self.conf_threshold = conf_threshold
        self.session = None
        self._initialize()

    def _initialize(self):
        if not os.path.exists(self.model_path):
            self._download_weights()
        if os.path.exists(self.model_path):
            try:
                import onnxruntime as ort

                opts = ort.SessionOptions()
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                self.session = ort.InferenceSession(
                    self.model_path, opts, providers=self.providers
                )
            except Exception as e:
                print(f"Warning: Failed to load {self.model_path}: {e}")

    def _download_weights(self):
        url = "https://huggingface.co/NightingaleCen/YOLO26m-seg-hand/resolve/main/yolo26m-seg-hand.onnx"
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        try:
            import urllib.request

            urllib.request.urlretrieve(url, self.model_path)
        except Exception as e:
            print(f"Warning: Failed to download {self.model_path}: {e}")

    @property
    def is_available(self) -> bool:
        return self.session is not None

    def segment(self, image: np.ndarray) -> np.ndarray | None:
        if not self.is_available:
            return None
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        blob = np.transpose(resized, (2, 0, 1))[np.newaxis, ...]

        input_name = self.session.get_inputs()[0].name
        out0, out1 = self.session.run(None, {input_name: blob})

        boxes = out0[0, :, :4]
        scores = out0[0, :, 4]
        coeffs = out0[0, :, 6:]
        protos = out1[0]

        valid = scores > self.conf_threshold
        if not np.any(valid):
            return None

        hand_mask = np.zeros((h, w), dtype=np.float32)
        scale_x = 56.0 / 224.0
        scale_y = 56.0 / 224.0

        for i in np.where(valid)[0]:
            box = boxes[i]
            coeff = coeffs[i]

            logits = np.sum(coeff[:, None, None] * protos, axis=0)
            prob_56 = 1.0 / (1.0 + np.exp(-logits))

            bx1 = max(0, int(box[0] * scale_x))
            by1 = max(0, int(box[1] * scale_y))
            bx2 = min(56, int(np.ceil(box[2] * scale_x)))
            by2 = min(56, int(np.ceil(box[3] * scale_y)))

            cropped_prob = np.zeros_like(prob_56)
            cropped_prob[by1:by2, bx1:bx2] = prob_56[by1:by2, bx1:bx2]

            prob_orig = cv2.resize(cropped_prob, (w, h), interpolation=cv2.INTER_LINEAR)
            hand_mask = np.maximum(hand_mask, (prob_orig > 0.5).astype(np.float32))

        if np.sum(hand_mask) == 0:
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hand_mask = cv2.dilate(hand_mask, kernel, iterations=1)
        return hand_mask


class HandOcclusionDetector:
    """
    Hand Occlusion Detector using YOLO Hand Instance Segmentation (HaGRID-trained).
    Exclusively uses YOLO to detect and segment hands and fingers in pixel space.
    """

    def __init__(
        self,
        yolo_model_path: str = "weights/yolo26m-seg-hand.onnx",
        providers: list[str] | None = None,
        conf_threshold: float = 0.40,
        **kwargs,
    ) -> None:
        self.yolo_model_path = yolo_model_path
        self.providers = providers
        self.conf_threshold = conf_threshold
        self.yolo_seg = None
        self._initialize()

    def _initialize(self):
        self.yolo_seg = YOLOHandSegmenter(
            self.yolo_model_path,
            providers=self.providers,
            conf_threshold=self.conf_threshold,
        )

    @property
    def is_available(self) -> bool:
        return self.yolo_seg is not None and self.yolo_seg.is_available

    def detect_hand_mask(self, image: np.ndarray) -> np.ndarray | None:
        """
        Detect hands in image using YOLO and return a binary float32 mask [0, 1] of the hand area.
        Returns None if no hands are detected.
        """
        if not self.is_available:
            return None
        return self.yolo_seg.segment(image)

