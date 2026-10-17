from __future__ import annotations

from .xseg import XSeg
from .bisenet import BiSeNetParser
from .segformer import SegFormerParser
from .hand_detector import HandOcclusionDetector, YOLOHandSegmenter, SelfieMulticlassSeg

MODEL_REGISTRY = {
    "BiSeNet ResNet-34 (Sharp & Detailed)": {
        "class": BiSeNetParser,
        "default_weight": "weights/parsing_resnet34.onnx",
        "url": "https://huggingface.co/yakhyo/uniface-weights/resolve/main/parsing_resnet34.onnx",
        "align_size": 512,
        "description": "512x512 BiSeNet with ResNet-34 backbone. High edge precision (~295 FPS on GPU).",
    },
    "BiSeNet ResNet-18 (Ultra Fast)": {
        "class": BiSeNetParser,
        "default_weight": "weights/parsing_resnet18.onnx",
        "url": "https://huggingface.co/yakhyo/uniface-weights/resolve/main/parsing_resnet18.onnx",
        "align_size": 512,
        "description": "512x512 BiSeNet with ResNet-18 backbone. Lightweight and fast (~407 FPS on GPU).",
    },
    "SegFormer B5 (Transformer Quality)": {
        "class": SegFormerParser,
        "default_weight": "weights/segformer_face.onnx",
        "url": "https://huggingface.co/jonathandinu/face-parsing/resolve/main/onnx/model_quantized.onnx",
        "align_size": 512,
        "description": "512x512 SegFormer B5 Transformer. Deep attention features for challenging poses.",
    },
    "XSeg (Original Model)": {
        "class": XSeg,
        "default_weight": "weights/xseg.onnx",
        "url": "https://github.com/yakhyo/face-segmentation/releases/download/weights/xseg.onnx",
        "align_size": 256,
        "description": "256x256 original XSeg occlusion segmentation model.",
    },
}

__all__ = [
    "XSeg",
    "BiSeNetParser",
    "SegFormerParser",
    "HandOcclusionDetector",
    "YOLOHandSegmenter",
    "SelfieMulticlassSeg",
    "MODEL_REGISTRY",
]
