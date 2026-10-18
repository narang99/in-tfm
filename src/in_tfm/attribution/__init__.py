from .attnlrp import compute_attnlrp_relevance, patch_eager_attention
from .deeplift import compute_deeplift_relevance
from .integrated_gradients import compute_integrated_gradients_relevance

__all__ = [
    "compute_attnlrp_relevance",
    "compute_deeplift_relevance",
    "compute_integrated_gradients_relevance",
    "patch_eager_attention",
]
