from .attnlrp import compute_attnlrp_relevance, patch_eager_attention
from .deeplift import compute_deeplift_relevance

__all__ = [
    "compute_attnlrp_relevance",
    "compute_deeplift_relevance",
    "patch_eager_attention",
]
