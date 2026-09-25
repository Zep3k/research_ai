from pathlib import Path

from .errors import TheoryError

STATE_DIR = Path(".theory")
DB_PATH = STATE_DIR / "research.db"
PAPERS_DIR = STATE_DIR / "papers"
CONFIG_PATH = STATE_DIR / "config.json"


def require_workspace() -> None:
    if not DB_PATH.exists():
        raise TheoryError('No theory workspace found. Run `theory init "Project name"` first.')
