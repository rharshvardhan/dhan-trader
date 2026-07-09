import sys
from pathlib import Path

# Ensure the project root is importable so tests can `import scan` / `import strategy`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
