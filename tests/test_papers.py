import pymupdf
import pytest

from theory.papers import import_pdf


def test_import_pdf_extracts_text_and_hash(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "A theorem statement")
        doc.save(source)

    imported = import_pdf(source, 1)

    assert imported.page_count == 1
    assert len(imported.sha256) == 64
    assert imported.pdf_path.exists()
    assert "A theorem statement" in imported.text_path.read_text(encoding="utf-8")


def test_import_pdf_leaves_no_partial_files_on_invalid_pdf(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "broken.pdf"
    source.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(Exception):
        import_pdf(source, 1)

    assert not list((tmp_path / ".theory" / "papers").glob("0001-*"))
