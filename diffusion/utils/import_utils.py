import importlib.util
import logging
import warnings
import os

import importlib_metadata
from packaging import version

logger = logging.getLogger(__name__)

# Check if xformers is disabled via environment variable
_xformers_disabled = os.getenv("XFORMERS_DISABLED", "0").lower() in ("1", "true", "yes")
if _xformers_disabled:
    logger.info("xformers is disabled via environment variable XFORMERS_DISABLED")
    _xformers_available = False
else:
    _xformers_available = importlib.util.find_spec("xformers") is not None
    try:
        if _xformers_available:
            _xformers_version = importlib_metadata.version("xformers")
            _torch_version = importlib_metadata.version("torch")
            if version.Version(_torch_version) < version.Version("1.12"):
                raise ValueError("xformers is installed but requires PyTorch >= 1.12")
            logger.debug(f"Successfully imported xformers version {_xformers_version}")
    except importlib_metadata.PackageNotFoundError:
        _xformers_available = False

_triton_modules_available = importlib.util.find_spec("triton") is not None
try:
    if _triton_modules_available:
        _triton_version = importlib_metadata.version("triton")
        if version.Version(_triton_version) < version.Version("3.0.0"):
            raise ValueError("triton is installed but requires Triton >= 3.0.0")
        logger.debug(f"Successfully imported triton version {_triton_version}")
except ImportError:
    _triton_modules_available = False
    warnings.warn("TritonLiteMLA and TritonMBConvPreGLU with `triton` is not available on your platform.")


def is_xformers_available():
    return _xformers_available


def is_triton_module_available():
    return _triton_modules_available
