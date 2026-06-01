import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = PROJECT_ROOT / "venv" / "Lib" / "site-packages"

if VENV_SITE_PACKAGES.exists():
    sys.path.append(str(VENV_SITE_PACKAGES))
