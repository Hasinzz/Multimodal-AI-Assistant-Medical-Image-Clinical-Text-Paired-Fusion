from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report_source"
OUT = ROOT / "outputs" / "final_revision"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


def all_tex() -> list[Path]:
    return sorted(REPORT.rglob("*.tex"))


def bib_keys() -> set[str]:
    bib = REPORT / "bibliography" / "references.bib"
    if not bib.exists():
        return set()
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", read(bib)))


def resolve_input(name: str) -> Path:
    path = REPORT / name
    if path.suffix:
        return path
    return path.with_suffix(".tex")


def check_inputs() -> list[str]:
    missing: list[str] = []
    main = REPORT / "main.tex"
    for name in re.findall(r"\\input\{([^}]+)\}", read(main)):
        if not resolve_input(name).exists():
            missing.append(name)
    for tex in all_tex():
        for name in re.findall(r"\\input\{([^}]+)\}", read(tex)):
            if not resolve_input(name).exists():
                missing.append(f"{tex.relative_to(REPORT)} -> {name}")
    return sorted(set(missing))


def check_graphics() -> list[str]:
    missing: list[str] = []
    for tex in all_tex():
        for name in re.findall(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", read(tex)):
            target = REPORT / name
            if target.exists():
                continue
            if target.suffix:
                missing.append(f"{tex.relative_to(REPORT)} -> {name}")
                continue
            found = any((REPORT / f"{name}{ext}").exists() for ext in [".png", ".jpg", ".jpeg", ".pdf"])
            if not found:
                missing.append(f"{tex.relative_to(REPORT)} -> {name}")
    return sorted(set(missing))


def check_citations() -> list[str]:
    keys = bib_keys()
    missing: list[str] = []
    for tex in all_tex():
        for cite_group in re.findall(r"\\cite\{([^}]+)\}", read(tex)):
            for key in [part.strip() for part in cite_group.split(",") if part.strip()]:
                if key not in keys:
                    missing.append(f"{tex.relative_to(REPORT)} -> {key}")
    return sorted(set(missing))


def brace_balance() -> list[str]:
    issues: list[str] = []
    for tex in all_tex():
        text = re.sub(r"\\[{}]", "", read(tex))
        left = text.count("{")
        right = text.count("}")
        if left != right:
            issues.append(f"{tex.relative_to(REPORT)}: {{={left}, }}={right}")
    return issues


def todo_markers() -> list[str]:
    hits: list[str] = []
    for tex in all_tex():
        for i, line in enumerate(read(tex).splitlines(), 1):
            if "TODO_CITATION" in line:
                hits.append(f"{tex.relative_to(REPORT)}:{i}: {line.strip()}")
    return hits


def unsafe_report_hits() -> list[str]:
    patterns = [
        r"doctor feedback",
        r"doctor-oriented",
        r"Doctor-Oriented",
        r"clinical recommendation",
        r"diagnostic recommendation",
        r"100% accuracy",
        r"100% accurate",
        r"doctor-validated",
    ]
    regex = re.compile("|".join(patterns), re.IGNORECASE)
    hits: list[str] = []
    for tex in all_tex():
        for i, line in enumerate(read(tex).splitlines(), 1):
            if regex.search(line):
                hits.append(f"{tex.relative_to(REPORT)}:{i}: {line.strip()}")
    return hits


def maybe_compile() -> tuple[str, str]:
    latexmk = shutil.which("latexmk")
    pdflatex = shutil.which("pdflatex")
    biber = shutil.which("biber")

    def compile_with_pdflatex_biber() -> tuple[str, str]:
        log_parts: list[str] = []
        commands = [
            [pdflatex, "-interaction=nonstopmode", "main.tex"],
            [biber, "main"],
            [pdflatex, "-interaction=nonstopmode", "main.tex"],
            [pdflatex, "-interaction=nonstopmode", "main.tex"],
        ]
        status = "compiled_with_pdflatex_biber"
        for command in commands:
            proc = subprocess.run(command, cwd=REPORT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            log_parts.append(proc.stdout[-1200:])
            if proc.returncode != 0:
                status = "pdflatex_biber_failed"
                break
        return status, "\n".join(log_parts)[-4000:]

    if latexmk:
        proc = subprocess.run(
            [latexmk, "-pdf", "-interaction=nonstopmode", "main.tex"],
            cwd=REPORT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=180,
        )
        if proc.returncode == 0:
            return "compiled_with_latexmk", proc.stdout[-4000:]
        latexmk_log = proc.stdout[-1200:]
        if pdflatex and biber:
            status, compile_log = compile_with_pdflatex_biber()
            return status, f"latexmk fallback: {latexmk_log}\n{compile_log}"[-4000:]
        return "latexmk_failed", proc.stdout[-4000:]
    if pdflatex and biber:
        return compile_with_pdflatex_biber()
    tools = {
        "latexmk": bool(latexmk),
        "pdflatex": bool(pdflatex),
        "biber": bool(biber),
    }
    return "compiler_not_available", str(tools)


def write_reports() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    missing_inputs = check_inputs()
    missing_graphics = check_graphics()
    missing_cites = check_citations()
    brace_issues = brace_balance()
    todos = todo_markers()
    unsafe_hits = unsafe_report_hits()
    compile_status, compile_log = maybe_compile()

    static_pass = not (missing_inputs or missing_graphics or missing_cites or brace_issues or unsafe_hits)

    check_md = f"""# LaTeX Compile or Static Check

Generated: {generated}

## Compile Status

- Status: `{compile_status}`
- Compiler detail: `{compile_log.strip()}`

## Static Checks

- Missing `\\input` files: {len(missing_inputs)}
- Missing `\\includegraphics` files: {len(missing_graphics)}
- Missing bibliography keys: {len(missing_cites)}
- Brace-balance issues: {len(brace_issues)}
- Unsafe report-facing phrase hits: {len(unsafe_hits)}
- Citation TODO markers: {len(todos)}
- Static check pass: `{static_pass}`

## Details

### Missing Inputs
{chr(10).join(f"- {item}" for item in missing_inputs) if missing_inputs else "- None"}

### Missing Graphics
{chr(10).join(f"- {item}" for item in missing_graphics) if missing_graphics else "- None"}

### Missing Citations
{chr(10).join(f"- {item}" for item in missing_cites) if missing_cites else "- None"}

### Brace Balance
{chr(10).join(f"- {item}" for item in brace_issues) if brace_issues else "- None"}

### Unsafe Phrase Hits
{chr(10).join(f"- {item}" for item in unsafe_hits) if unsafe_hits else "- None"}

### Citation TODOs
{chr(10).join(f"- {item}" for item in todos) if todos else "- None"}
"""
    (OUT / "latex_compile_or_static_check.md").write_text(check_md, encoding="utf-8")

    final_summary = f"""# Final Report Update Summary

Generated: {generated}

## Backup Path

`outputs/final_revision/report_backup_before_careful_update/report_source`

## Model-1 Cross-Validation Status

Status B: overfitting/generalization evidence exists, but no completed Model-1 k-fold cross-validation artefacts were found. The report now states held-out technical performance and lists k-fold cross-validation as future work.

## Training Run/Skipped

No new training was run for this report update. A new Model-1 k-fold run was skipped because no prior fold infrastructure/output was present and launching a new fold study during the report rewrite would introduce unverified results.

## Report Files Edited

- `report_source/core/abstract.tex`
- `report_source/core/approval.tex`
- `report_source/core/titlepage.tex`
- `report_source/chapters/chapter_1.tex`
- `report_source/chapters/chapter_2.tex`
- `report_source/chapters/Chapter_3.tex`
- `report_source/chapters/Chapter_4.tex`
- `report_source/chapters/Chapter_5.tex`
- `report_source/chapters/Chapter_6.tex`
- `report_source/chapters/Chapter_7.tex`
- `report_source/chapters/Chapter_8.tex`
- `report_source/tables/revised_final_results_tables.tex`
- `app.py`

## Figures Added/Replaced/Removed

- Added `report_source/figures/revised_system_architecture.png`.
- Replaced the Chapter 4 architecture reference with the revised architecture figure.
- Added `report_source/figures/fig15_best_version_claim_status_summary_revised.png`.
- Replaced the Chapter 7 claim-status reference with the revised claim-status figure.
- Removed no physical figure files; older figures remain for traceability.

## Doctor-Feedback Wording Removal Status

Report-facing wording was changed to generated follow-up note, retrieval-supported follow-up note, or retrieval-supported review text. The Streamlit visible label was changed to `Generated follow-up note`. Legacy backend keys remain only for compatibility in code/data fields.

## Citation TODOs

- `TODO_CITATION_MTSAMPLES` remains in Chapters 5 and 7 until a verified MTSamples dataset citation/source entry is added.
- No fabricated citation was added.

## Compile/Static-Check Status

- Compile status: `{compile_status}`
- Static check pass: `{static_pass}`
- Missing inputs: {len(missing_inputs)}
- Missing figures: {len(missing_graphics)}
- Missing citations: {len(missing_cites)}
- Brace issues: {len(brace_issues)}
- Unsafe report-facing phrase hits: {len(unsafe_hits)}

## Final Summary Path

`outputs/final_revision/FINAL_REPORT_UPDATE_SUMMARY.md`

## Remaining Manual Tasks

- Add a verified MTSamples bibliography entry and replace the TODO markers.
- Run a full LaTeX compile on a machine with `latexmk` or `pdflatex`/`biber` available.
- Manually proofread final prose for thesis style and supervisor formatting requirements.
"""
    (OUT / "FINAL_REPORT_UPDATE_SUMMARY.md").write_text(final_summary, encoding="utf-8")

    print(check_md)


if __name__ == "__main__":
    write_reports()
