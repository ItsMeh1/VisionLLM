"""Set environment flags before PyTorch is imported anywhere."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

__version__ = "1.0.0"
