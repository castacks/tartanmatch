"""TartanMatch: dense correspondence across RGB, depth, thermal, LiDAR, and event modalities."""

from tartanmatch.config import TartanMatchConfig
from tartanmatch.model import MODALITIES, Modality, TartanMatch, TartanMatchOutput

__all__ = ["MODALITIES", "Modality", "TartanMatch", "TartanMatchConfig", "TartanMatchOutput"]
__version__ = "1.0.0"
