"""Safe extraction of user-provided literature text."""

from __future__ import annotations

from io import BytesIO

from pypdf import PdfReader


MAX_EXTRACTED_CHARACTERS = 250_000
MAX_PDF_PAGES = 100


def extract_literature_text(content: bytes, content_type: str) -> str:
    """Extract bounded text from supported user literature formats."""
    if content_type == "application/pdf":
        reader = PdfReader(BytesIO(content), strict=False)
        chunks: list[str] = []
        for page in reader.pages[:MAX_PDF_PAGES]:
            chunks.append(page.extract_text() or "")
        text = "\n\n".join(chunks)
    elif content_type in {"text/plain", "text/markdown"}:
        text = content.decode("utf-8-sig", errors="replace")
    else:
        raise ValueError("Unsupported literature content type")
    return text[:MAX_EXTRACTED_CHARACTERS].strip()
