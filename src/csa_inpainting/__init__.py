"""Coherent Semantic Attention for Image Inpainting."""

from .attention import CoherentSemanticAttention
from .networks import InpaintingModel

__all__ = ["CoherentSemanticAttention", "InpaintingModel"]
__version__ = "0.2.0"
