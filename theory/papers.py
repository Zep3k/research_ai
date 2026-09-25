import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from .paths import PAPERS_DIR


@dataclass(frozen=True)
class ImportedPaper:
    pdf_path: Path
    text_path: Path
    sha256: str
    page_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_pdf(source: Path, paper_id: int) -> ImportedPaper:
    if paper_id <= 0:
        raise ValueError("paper_id must be positive")
    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    dest = PAPERS_DIR / f"{paper_id:04d}-{source.name}"
    chunks = []
    with pymupdf.open(source) as doc:
        if not doc.is_pdf:
            raise ValueError(f"Not a valid PDF: {source}")
        page_count = doc.page_count
        for i, page in enumerate(doc):
            chunks += [f"\n\n--- PAGE {i+1} ---\n\n", page.get_text()]

    text_path = dest.with_suffix(".txt")
    temporary_text = Path(f"{text_path}.tmp")
    try:
        shutil.copy2(source, dest)
        temporary_text.write_text("".join(chunks), encoding="utf-8", errors="ignore")
        temporary_text.replace(text_path)
    except Exception:
        dest.unlink(missing_ok=True)
        temporary_text.unlink(missing_ok=True)
        text_path.unlink(missing_ok=True)
        raise

    return ImportedPaper(dest, text_path, _sha256(dest), page_count)
