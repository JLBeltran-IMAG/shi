"""SHI core package initialization."""
from .processor import SHIProcessor
from .cleaner import Cleaner
from .exceptions import SHIError
from .config import config

__version__ = "1.0.0"
__all__ = ["SHIProcessor", "Cleaner", "SHIError", "config"]
