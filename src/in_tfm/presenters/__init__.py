"""Turning a cluster's sampled hits into report artifacts.

The other modality-aware module besides `sources`. A presenter owns two decisions that differ
completely between images and text:

- how to collapse the attribution's feature axis (channels for an image, hidden for a token)
- what a rendered hit even is - a JPEG overlay, or highlighted text
"""

from .base import ClusterHit, ClusterPresenter, HadamardShape
from .image import ImagePresenter
from .text import TextPresenter

__all__ = ["ClusterHit", "ClusterPresenter", "HadamardShape", "ImagePresenter", "TextPresenter"]
