from typing import List
import re

def _clean_text(s: str) -> str:
    s = s.replace('\x00', '')
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

def extract_text_from_pdf(path: str) -> str:
    """Try PyMuPDF first; fallback to pdfminer if needed."""
    try:
        import fitz  # PyMuPDF
        text_parts: List[str] = []
        with fitz.open(path) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        text = "\n".join(text_parts)
        if len(text.strip()) > 0:
            return _clean_text(text)
    except Exception:
        pass

    # fallback to pdfminer
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        text = pdfminer_extract_text(path)
        return _clean_text(text)
    except Exception as e:
        raise RuntimeError(f"Failed to extract text: {e}")
