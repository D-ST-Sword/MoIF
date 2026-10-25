"""MoIF: learned subset predictions with implicit interaction-order filtering."""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .model import MoIF, build, objective

__all__ = ["MoIF", "build", "objective"]
