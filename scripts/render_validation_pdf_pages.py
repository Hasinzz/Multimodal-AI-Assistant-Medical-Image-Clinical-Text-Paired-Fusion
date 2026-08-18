from __future__ import annotations

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "validation pdf.pdf"
OUT = ROOT / "outputs" / "validation_pdf_page_screens"
PAGES = [
    5,
    24,
    26,
    27,
    28,
    33,
    34,
    35,
    37,
    42,
    44,
    45,
    46,
    47,
    48,
    50,
    51,
    52,
    53,
    54,
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    for page_number in PAGES:
        pix = doc.load_page(page_number - 1).get_pixmap(matrix=fitz.Matrix(2, 2))
        path = OUT / f"page_{page_number:02d}.png"
        pix.save(path)
        print(path)


if __name__ == "__main__":
    main()
