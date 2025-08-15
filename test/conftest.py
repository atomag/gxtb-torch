import sys
from pathlib import Path

# Ensure local src directory is importable as package root for gxtb
root = Path(__file__).resolve().parents[1]
src = root / 'src'
if str(src) not in sys.path:
    sys.path.insert(0, str(src))
