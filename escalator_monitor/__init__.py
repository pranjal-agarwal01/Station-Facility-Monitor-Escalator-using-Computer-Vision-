"""Escalator operational-state monitoring from fixed CCTV footage."""

from .config import Config
from .geometry import Quad
from .pipeline import EscalatorMonitor, FrameResult
from .state import State

__version__ = "2.0.0"
__all__ = ["Config", "EscalatorMonitor", "FrameResult", "Quad", "State", "__version__"]
