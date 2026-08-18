from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "outputs" / "validation_pdf_extracted_pages.md"
OUT = ROOT / "outputs" / "validation_pdf_claim_inventory.md"

PATTERNS = re.compile(
    r"(accuracy|macro|micro|F1|AUROC|AUC|mAP|precision|recall|100|failed|"
    r"checkpoint|final_v2|large_v2|retrain|BioBERT|BERT|YOLO|clinical|doctor|"
    r"cross-modal|attention|Model-4|model-4|V4|validation)",
    re.IGNORECASE,
)


def main() -> None:
    text = EXTRACTED.read_text(encoding="utf-8")
    chunks = re.split(r"\n## Page (\d+)\n\n", text)[1:]
    lines_out: list[str] = ["# Validation PDF Claim Inventory", ""]

    for idx in range(0, len(chunks), 2):
        page = chunks[idx]
        body = chunks[idx + 1]
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        heading = " | ".join(lines[:4])
        hits = [line for line in lines if PATTERNS.search(line)]

        lines_out.append(f"## Page {page}")
        lines_out.append("")
        lines_out.append(f"Opening text: {heading}")
        lines_out.append("")
        if hits:
            lines_out.append("Claim/metric/risk lines:")
            for hit in hits:
                clipped = hit[:240]
                lines_out.append(f"- {clipped}")
        else:
            lines_out.append("No direct metric/claim keywords found.")
        lines_out.append("")

    OUT.write_text("\n".join(lines_out), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
