"""Ensures the repo root is importable as a package root (so `from src...` works)
regardless of which directory pytest is invoked from."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
