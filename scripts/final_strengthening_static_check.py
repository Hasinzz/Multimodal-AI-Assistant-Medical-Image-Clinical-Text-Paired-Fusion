from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report_source"
OUTPUT = ROOT / "outputs" / "final_research_strengthening"
MIKTEX_BIN = Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and compile the final strengthened thesis report.")
    parser.add_argument("--static-only", action="store_true")
    return parser.parse_args()


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


def tex_files() -> list[Path]:
    return sorted(REPORT.rglob("*.tex"))


def bibliography_keys() -> set[str]:
    bib = REPORT / "bibliography" / "references.bib"
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", read(bib)))


def duplicate_bibliography_records() -> list[str]:
    source = read(REPORT / "bibliography" / "references.bib")
    keys = re.findall(r"@\w+\s*\{\s*([^,\s]+)", source)
    issues = [f"duplicate BibTeX key: {key}" for key, count in Counter(keys).items() if count > 1]
    current_key: str | None = None
    dois: dict[str, list[str]] = {}
    for line in source.splitlines():
        entry = re.match(r"@\w+\s*\{\s*([^,\s]+)", line)
        if entry:
            current_key = entry.group(1)
        doi = re.match(r"\s*doi\s*=\s*\{([^}]+)\}", line, re.IGNORECASE)
        if doi and current_key:
            normalized = doi.group(1).strip().lower().removeprefix("https://doi.org/")
            dois.setdefault(normalized, []).append(current_key)
    issues.extend(
        f"duplicate DOI {doi}: {', '.join(keys)}"
        for doi, keys in sorted(dois.items())
        if len(keys) > 1
    )
    return issues


def resolve_input(name: str) -> Path:
    path = REPORT / name
    return path if path.suffix else path.with_suffix(".tex")


def check_inputs() -> list[str]:
    missing: list[str] = []
    for tex in tex_files():
        for name in re.findall(r"\\input\{([^}]+)\}", read(tex)):
            if not resolve_input(name).exists():
                missing.append(f"{tex.relative_to(REPORT)} -> {name}")
    return sorted(set(missing))


def check_graphics() -> list[str]:
    missing: list[str] = []
    for tex in tex_files():
        for name in re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", read(tex)):
            target = REPORT / name
            candidates = [target] if target.suffix else [target.with_suffix(ext) for ext in (".png", ".jpg", ".jpeg", ".pdf")]
            if not any(candidate.exists() for candidate in candidates):
                missing.append(f"{tex.relative_to(REPORT)} -> {name}")
    return sorted(set(missing))


def check_citations() -> list[str]:
    known = bibliography_keys()
    missing: list[str] = []
    for tex in tex_files():
        for cite_group in re.findall(r"\\(?:textcite|parencite|cite)\{([^}]+)\}", read(tex)):
            for key in (part.strip() for part in cite_group.split(",")):
                if key and key not in known:
                    missing.append(f"{tex.relative_to(REPORT)} -> {key}")
    return sorted(set(missing))


def check_braces() -> list[str]:
    issues: list[str] = []
    for tex in tex_files():
        source = re.sub(r"\\[{}]", "", read(tex))
        if source.count("{") != source.count("}"):
            issues.append(f"{tex.relative_to(REPORT)}: left={source.count('{')}, right={source.count('}')}")
    return issues


def line_hits(pattern: re.Pattern[str]) -> list[str]:
    hits: list[str] = []
    files = tex_files() + [REPORT / "bibliography" / "references.bib"]
    for path in files:
        for number, line in enumerate(read(path).splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(REPORT)}:{number}: {line.strip()}")
    return hits


def duplicate_labels() -> list[str]:
    locations: dict[str, list[str]] = {}
    for tex in tex_files():
        for number, line in enumerate(read(tex).splitlines(), 1):
            for label in re.findall(r"\\label\{([^}]+)\}", line):
                locations.setdefault(label, []).append(f"{tex.relative_to(REPORT)}:{number}")
    return [f"{label}: {', '.join(paths)}" for label, paths in sorted(locations.items()) if len(paths) > 1]


def table_caption_issues() -> list[str]:
    issues: list[str] = []
    for tex in tex_files():
        source = read(tex)
        for environment in ("table", "longtable"):
            block_pattern = re.compile(
                rf"\\begin\{{{environment}\}}.*?\\end\{{{environment}\}}",
                re.DOTALL,
            )
            for index, block in enumerate(block_pattern.findall(source), 1):
                if "\\caption*" in block:
                    issues.append(
                        f"{tex.relative_to(REPORT)} {environment} {index}: "
                        "starred caption is absent from List of Tables"
                    )
                elif "\\caption{" not in block:
                    issues.append(f"{tex.relative_to(REPORT)} {environment} {index}: missing caption")
                if "\\label{" not in block:
                    issues.append(f"{tex.relative_to(REPORT)} {environment} {index}: missing label")
    return issues


def markdown_hits() -> list[str]:
    pattern = re.compile(r"^\s*(?:```|#{1,6}\s|\|\s*:?-{3})")
    hits: list[str] = []
    for tex in tex_files():
        for number, line in enumerate(read(tex).splitlines(), 1):
            if pattern.search(line):
                hits.append(f"{tex.relative_to(REPORT)}:{number}: {line.strip()}")
    return hits


def find_tool(name: str) -> str | None:
    local = MIKTEX_BIN / f"{name}.exe"
    if local.exists():
        return str(local)
    return shutil.which(name)


def compile_report() -> tuple[str, str]:
    pdflatex = find_tool("pdflatex")
    biber = find_tool("biber")
    if not pdflatex or not biber:
        return "compiler_not_available", f"pdflatex={pdflatex!r}; biber={biber!r}"
    commands = [
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        [biber, "main"],
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
    ]
    logs: list[str] = []
    for command in commands:
        process = subprocess.run(
            command,
            cwd=REPORT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
        logs.append(f"$ {' '.join(command)}\n{process.stdout}")
        if process.returncode != 0:
            return "failed", "\n\n".join(logs)
    return "compiled_with_pdflatex_biber", "\n\n".join(logs)


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    missing_inputs = check_inputs()
    missing_graphics = check_graphics()
    missing_citations = check_citations()
    brace_issues = check_braces()
    todo_hits = line_hits(re.compile(r"TODO(?:_CITATION)?", re.IGNORECASE))
    unsafe_hits = line_hits(
        re.compile(
            r"doctor[- ]oriented feedback|doctor feedback|physician feedback|doctor advice|"
            r"treatment recommendation|diagnostic recommendation|medical advice|clinical recommendation",
            re.IGNORECASE,
        )
    )
    conflict_hits = line_hits(re.compile(r"^(?:<{7}|={7}|>{7})"))
    malformed_bib_hits = line_hits(re.compile(r"\band\s+et\s+al\.?", re.IGNORECASE))
    duplicate_bib_hits = duplicate_bibliography_records()
    duplicate_label_hits = duplicate_labels()
    table_issues = table_caption_issues()
    markdown_source_hits = markdown_hits()
    status, compile_log = ("static_only", "Compilation not requested.") if args.static_only else compile_report()
    log_path = OUTPUT / "final_latex_compile.log"
    log_path.write_text(compile_log, encoding="utf-8")
    latex_log = read(REPORT / "main.log") if (REPORT / "main.log").exists() else ""
    unresolved_references = sorted(set(re.findall(r"LaTeX Warning: Reference `([^']+)'", latex_log)))
    unresolved_citations = sorted(set(re.findall(r"LaTeX Warning: Citation '([^']+)'", latex_log)))
    overfull_boxes = len(re.findall(r"Overfull \\[hv]box", latex_log))
    lot_text = read(REPORT / "main.lot") if (REPORT / "main.lot").exists() else ""
    lof_text = read(REPORT / "main.lof") if (REPORT / "main.lof").exists() else ""
    source_table_captions = 0
    for path in tex_files():
        source = read(path)
        for environment in ("table", "longtable"):
            blocks = re.findall(
                rf"\\begin\{{{environment}\}}.*?\\end\{{{environment}\}}",
                source,
                re.DOTALL,
            )
            source_table_captions += sum("\\caption{" in block for block in blocks)
    source_figure_captions = sum(
        len(re.findall(r"\\begin\{figure\}.*?\\caption\{", read(path), re.DOTALL))
        for path in tex_files()
    )
    lot_entries = len(re.findall(r"\\contentsline \{table\}", lot_text))
    lof_entries = len(re.findall(r"\\contentsline \{figure\}", lof_text))
    list_entry_issues: list[str] = []
    if status == "compiled_with_pdflatex_biber":
        if lot_entries != source_table_captions:
            list_entry_issues.append(
                f"List of Tables entries={lot_entries}, source table captions={source_table_captions}"
            )
        if lof_entries != source_figure_captions:
            list_entry_issues.append(
                f"List of Figures entries={lof_entries}, source figure captions={source_figure_captions}"
            )
        if "Cross-modal input-combination technical validation" not in lot_text:
            list_entry_issues.append("cross-modal validation table is absent from the List of Tables")
    static_pass = not any(
        (
            missing_inputs,
            missing_graphics,
            missing_citations,
            brace_issues,
            todo_hits,
            unsafe_hits,
            conflict_hits,
            malformed_bib_hits,
            duplicate_bib_hits,
            duplicate_label_hits,
            table_issues,
            markdown_source_hits,
            unresolved_references,
            unresolved_citations,
            list_entry_issues,
        )
    ) and overfull_boxes == 0
    compile_pass = status == "compiled_with_pdflatex_biber" or args.static_only
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "compile_status": status,
        "compile_log": str(log_path.relative_to(ROOT)),
        "compiled_pdf": str((REPORT / "main.pdf").relative_to(ROOT)) if (REPORT / "main.pdf").exists() else None,
        "static_pass": static_pass,
        "compile_pass": compile_pass,
        "counts": {
            "missing_inputs": len(missing_inputs),
            "missing_graphics": len(missing_graphics),
            "missing_citations": len(missing_citations),
            "brace_issues": len(brace_issues),
            "todo_hits": len(todo_hits),
            "unsafe_hits": len(unsafe_hits),
            "conflict_hits": len(conflict_hits),
            "malformed_bib_hits": len(malformed_bib_hits),
            "duplicate_bibliography_records": len(duplicate_bib_hits),
            "duplicate_labels": len(duplicate_label_hits),
            "table_caption_or_label_issues": len(table_issues),
            "markdown_source_hits": len(markdown_source_hits),
            "unresolved_references": len(unresolved_references),
            "unresolved_citations": len(unresolved_citations),
            "overfull_boxes": overfull_boxes,
            "list_entry_issues": len(list_entry_issues),
        },
        "details": {
            "missing_inputs": missing_inputs,
            "missing_graphics": missing_graphics,
            "missing_citations": missing_citations,
            "brace_issues": brace_issues,
            "todo_hits": todo_hits,
            "unsafe_hits": unsafe_hits,
            "conflict_hits": conflict_hits,
            "malformed_bib_hits": malformed_bib_hits,
            "duplicate_bibliography_records": duplicate_bib_hits,
            "duplicate_labels": duplicate_label_hits,
            "table_caption_or_label_issues": table_issues,
            "markdown_source_hits": markdown_source_hits,
            "unresolved_references": unresolved_references,
            "unresolved_citations": unresolved_citations,
            "list_entry_issues": list_entry_issues,
        },
    }
    (OUTPUT / "final_latex_compile_and_static_check.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Final LaTeX Compile and Static Check",
        "",
        f"- Compile status: `{status}`",
        f"- Compile pass: `{compile_pass}`",
        f"- Static pass: `{static_pass}`",
        f"- PDF: `{payload['compiled_pdf']}`",
        f"- Compile log: `{payload['compile_log']}`",
        "",
        "| Check | Count |",
        "|---|---:|",
        *[f"| {name.replace('_', ' ')} | {count} |" for name, count in payload["counts"].items()],
        "",
        "Full findings are recorded in `final_latex_compile_and_static_check.json`.",
    ]
    (OUTPUT / "final_latex_compile_and_static_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not static_pass or not compile_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
