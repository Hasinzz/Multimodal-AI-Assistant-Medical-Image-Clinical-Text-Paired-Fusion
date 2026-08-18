from __future__ import annotations

import re
import argparse
from dataclasses import dataclass
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "new validation pdf.pdf"
PDF = DEFAULT_PDF
OUT_DIR = ROOT / "outputs" / "new_validation_pdf_audit"
EXTRACTED = OUT_DIR / "new_validation_pdf_extracted_pages.md"
INVENTORY = OUT_DIR / "new_validation_pdf_claim_inventory.md"
ISSUE_SCAN = OUT_DIR / "new_validation_pdf_issue_scan.md"
SCREEN_DIR = OUT_DIR / "page_screens"

CLAIM_PATTERNS = re.compile(
    r"(accuracy|macro|micro|F1|AUROC|AUC|mAP|precision|recall|100|failed|"
    r"checkpoint|final_v2|large_v2|retrain|BioBERT|BERT|YOLO|clinical|doctor|"
    r"cross-modal|attention|Model-4|model-4|V4|validation|knowledge base|knowledge graph)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Check:
    label: str
    pattern: str
    expected: str
    severity: str


CHECKS = [
    Check("Model-4 overclaim", r"\bmodel-4\b|\bModel-4\b", "Should only appear in negated wording such as 'not a separate Model-4'.", "high"),
    Check("Bad model-4 enhancement phrase", r"enhancement to the model-4", "Must be removed.", "high"),
    Check("V3 knowledge base", r"V3 knowledge base", "Should be V4 knowledge base / data/rag_kb_v4.", "high"),
    Check("Knowledge graph wording", r"knowledge graph", "Should be RAG knowledge base unless a real graph exists.", "medium"),
    Check("Broken NER phrase", r"BERT-based weak in the art of NER", "Should be BERT-based weakly supervised NER.", "medium"),
    Check("Purchased typo", r"Model-3 was purchased|purchased and evaluated", "Should be implemented/developed and evaluated.", "medium"),
    Check("Breach X-ray typo", r"Breach X-ray", "Should be Chest X-ray.", "low"),
    Check("Uncompressed V4 typo", r"uncompressed V4", "Should be current/experimental V4.", "low"),
    Check("Foreign doctors wording", r"Foreign doctors", "Should be external doctors/clinicians/domain experts.", "low"),
    Check("Howeverer typo", r"Howeverer", "Should be However.", "low"),
    Check("Reseach typo", r"\breseach\b", "Should be research.", "low"),
    Check("Into spacing", r"\bin to\b", "Usually should be into.", "low"),
    Check("Pseudolabeled style", r"pseudolabeled", "Prefer pseudo-labeled.", "low"),
    Check("Medical photographs wording", r"medical photographs", "Use medical images.", "low"),
    Check("Take care of wording", r"take care of", "Use process/handle.", "low"),
    Check("Figure 4.2 AUC caption", r"distribution of Test per-class AUC|distribution of test per-class AUC", "AUC is a model result, not dataset distribution.", "low"),
]


RESULT_CHECKS = [
    ("Brain MRI accuracy", r"0\.9375"),
    ("Brain MRI macro F1", r"0\.9359"),
    ("X-ray macro AUROC", r"0\.8133"),
    ("X-ray micro AUROC", r"0\.8377"),
    ("X-ray tuned macro F1", r"0\.2862|0\.286"),
    ("X-ray tuned micro F1", r"0\.3366|0\.336"),
    ("100 completed / 0 failed", r"100.*completed|completed.*100|0 failed|failed.*0"),
    ("Model-2 60/60", r"60/60"),
    ("Model-3 100/100", r"100/100"),
    ("YOLO mAP50", r"0\.8357|0\.836"),
    ("BERT NER entity F1", r"0\.9963|0\.996"),
]

PAGES_TO_RENDER = [5, 23, 33, 35, 37, 42, 48, 50, 51, 52, 53, 54, 56, 58]


def configure_paths(pdf_arg: str | None) -> None:
    global PDF, OUT_DIR, EXTRACTED, INVENTORY, ISSUE_SCAN, SCREEN_DIR

    PDF = Path(pdf_arg) if pdf_arg else DEFAULT_PDF
    if not PDF.is_absolute():
        PDF = ROOT / PDF
    safe_stem = re.sub(r"[^A-Za-z0-9]+", "_", PDF.stem).strip("_").lower()
    OUT_DIR = ROOT / "outputs" / f"{safe_stem}_audit"
    EXTRACTED = OUT_DIR / f"{safe_stem}_extracted_pages.md"
    INVENTORY = OUT_DIR / f"{safe_stem}_claim_inventory.md"
    ISSUE_SCAN = OUT_DIR / f"{safe_stem}_issue_scan.md"
    SCREEN_DIR = OUT_DIR / "page_screens"


def extract_pages() -> list[tuple[int, str]]:
    doc = fitz.open(PDF)
    pages: list[tuple[int, str]] = []
    lines = [f"# Extracted text from {PDF.name}", ""]
    for index, page in enumerate(doc, start=1):
        text = page.get_text("text")
        pages.append((index, text))
        lines.append(f"## Page {index}")
        lines.append("")
        lines.append(text.rstrip())
        lines.append("")
    EXTRACTED.write_text("\n".join(lines), encoding="utf-8")
    return pages


def write_inventory(pages: list[tuple[int, str]]) -> None:
    lines = [f"# Claim Inventory for {PDF.name}", ""]
    for page_no, text in pages:
        clean_lines = [line.strip() for line in text.splitlines() if line.strip()]
        hits = [line for line in clean_lines if CLAIM_PATTERNS.search(line)]
        opening = " | ".join(clean_lines[:4])
        lines.append(f"## Page {page_no}")
        lines.append("")
        lines.append(f"Opening text: {opening}")
        lines.append("")
        if hits:
            lines.append("Claim/metric/risk lines:")
            for hit in hits:
                lines.append(f"- {hit[:260]}")
        else:
            lines.append("No direct metric/claim keywords found.")
        lines.append("")
    INVENTORY.write_text("\n".join(lines), encoding="utf-8")


def find_hits(text: str, pattern: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for page_match in re.finditer(r"\n## Page (\d+)\n\n(.*?)(?=\n## Page \d+\n\n|\Z)", text, flags=re.S):
        page_no = int(page_match.group(1))
        body = page_match.group(2)
        for line in body.splitlines():
            if re.search(pattern, line, flags=re.I):
                hits.append((page_no, line.strip()))
    return hits


def write_issue_scan() -> None:
    extracted = EXTRACTED.read_text(encoding="utf-8")
    lines = [f"# Issue Scan for {PDF.name}", ""]
    lines.append("## Prior Fix Checks")
    lines.append("")
    for check in CHECKS:
        hits = find_hits(extracted, check.pattern)
        status = "FOUND" if hits else "clear"
        lines.append(f"### {check.label}")
        lines.append("")
        lines.append(f"- Severity: {check.severity}")
        lines.append(f"- Status: {status}")
        lines.append(f"- Expected: {check.expected}")
        if hits:
            lines.append("- Hits:")
            for page_no, line in hits[:12]:
                lines.append(f"  - Page {page_no}: {line[:260]}")
            if len(hits) > 12:
                lines.append(f"  - ... {len(hits) - 12} more")
        lines.append("")

    lines.append("## Result Value Presence Checks")
    lines.append("")
    for label, pattern in RESULT_CHECKS:
        hits = find_hits(extracted, pattern)
        lines.append(f"- {label}: {'FOUND' if hits else 'MISSING'}")
        for page_no, line in hits[:5]:
            lines.append(f"  - Page {page_no}: {line[:220]}")
    ISSUE_SCAN.write_text("\n".join(lines), encoding="utf-8")


def render_pages() -> None:
    SCREEN_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    for page_no in PAGES_TO_RENDER:
        if page_no < 1 or page_no > len(doc):
            continue
        pix = doc.load_page(page_no - 1).get_pixmap(matrix=fitz.Matrix(2, 2))
        pix.save(SCREEN_DIR / f"page_{page_no:02d}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit validation PDF claims and metrics.")
    parser.add_argument("pdf", nargs="?", help="PDF file to audit. Defaults to new validation pdf.pdf.")
    args = parser.parse_args()
    configure_paths(args.pdf)
    if not PDF.exists():
        raise FileNotFoundError(PDF)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pages = extract_pages()
    write_inventory(pages)
    write_issue_scan()
    render_pages()
    print(EXTRACTED)
    print(INVENTORY)
    print(ISSUE_SCAN)
    print(SCREEN_DIR)


if __name__ == "__main__":
    main()
