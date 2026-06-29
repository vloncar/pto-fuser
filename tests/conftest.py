"""Make the package importable without an install (mirrors the prototype style)."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.environ.setdefault("PTO_LIB_PATH", "/home/vloncar/work/einsum_workspace/pto-isa")
