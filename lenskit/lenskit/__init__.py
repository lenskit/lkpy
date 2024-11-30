"""
Recommender systems toolkit.
"""

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "UNSPECIFIED"
