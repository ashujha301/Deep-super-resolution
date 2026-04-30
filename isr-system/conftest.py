# conftest.py
# Shared pytest configuration. Placed at project root so
# pytest can find src/ as a package without installing it.

import sys
from pathlib import Path

# Add project root to sys.path so `from src.data.x import y` works
sys.path.insert(0, str(Path(__file__).parent))
