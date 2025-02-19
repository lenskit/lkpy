"""
Recommender systems toolkit.
"""

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "UNSPECIFIED"


from .operations import predict, recommend, score
from .pipeline import Pipeline, RecPipelineBuilder, topn_pipeline

__all__ = ["predict", "recommend", "score", "Pipeline", "RecPipelineBuilder", "topn_pipeline"]
