# tests/conftest.py
from pathlib import Path
import sys

# Ensure project root is on sys.path for `import src.*`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
