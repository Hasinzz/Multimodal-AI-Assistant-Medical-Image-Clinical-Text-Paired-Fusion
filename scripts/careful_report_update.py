from __future__ import annotations

import json
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report_source"
OUT = ROOT / "outputs" / "final_revision"
FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def write_text(path: Path, text: str, changed: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text(encoding="utf-8") if path.exists() else None
    text = text.replace("\r\n", "\n")
    if old != text:
        path.write_text(text, encoding="utf-8")
        changed.append(str(path.relative_to(ROOT)))


def replace_between(text: str, start: str, end: str, block: str, anchor: str | None = None, before: bool = False) -> str:
    block = block.strip("\n")
    new_block = f"{start}\n{block}\n{end}"
    if start in text and end in text:
        prefix, rest = text.split(start, 1)
        _, suffix = rest.split(end, 1)
        return prefix + new_block + suffix
    if anchor is None:
        return text.rstrip() + "\n\n" + new_block + "\n"
    if anchor not in text:
        return text.rstrip() + "\n\n" + new_block + "\n"
    if before:
        return text.replace(anchor, new_block + "\n\n" + anchor, 1)
    return text.replace(anchor, anchor + "\n\n" + new_block, 1)


def safe_report_replacements(text: str) -> str:
    replacements = {
        "doctor-oriented feedback": "retrieval-supported follow-up note",
        "Doctor-oriented feedback": "Retrieval-supported follow-up note",
        "Doctor-oriented Feedback": "Retrieval-Supported Follow-up Note",
        "Doctor-Oriented Feedback": "Retrieval-Supported Follow-up Note",
        "doctor oriented feedback": "retrieval-supported follow-up note",
        "Doctor oriented feedback": "Retrieval-supported follow-up note",
        "doctor feedback": "generated follow-up note",
        "Doctor feedback": "Generated follow-up note",
        "feedback for the doctor": "generated follow-up note for review",
        "feedback for the doctors": "generated follow-up note for review",
        "feedback to doctors": "generated follow-up notes for review",
        "feedback to the physicians": "generated follow-up notes for review",
        "feedback to the physician": "generated follow-up note for review",
        "feedback given to doctors": "generated follow-up note",
        "feedback that is directed at the doctor": "generated follow-up note",
        "feedback directed to the physician": "generated follow-up note",
        "physician-targeted feedback": "generated follow-up note",
        "doctor-targeted feedback": "generated follow-up note",
        "Doctor-Oriented Advice Generation": "Retrieval-Supported Follow-up Note Generation",
        "Doctor-Oriented Report Generation": "Retrieval-Supported Follow-up Note Generation",
        "decision support advice": "decision-support review text",
        "advice that is structured": "structured review text",
        "increased health record accuracy": "more consistent health record organization",
        "qualitative comments of clinicians": "technical artefact review and model-specific metrics",
        "medical advice": "clinical instruction",
        "recommended treatments": "treatment directions",
        "summary with feedback for the physician": "summary with a generated follow-up note for review",
        "summary generation and a Streamlit interface": "summary generation, follow-up note generation, and a Streamlit interface",
        "RAG-based feedback generation": "RAG-based follow-up note generation",
        "RAG feedback pipeline": "RAG follow-up note pipeline",
        "feedback generation system": "summarization and follow-up note generation system",
        "feedback generation in V4 advanced mode": "follow-up note generation in V4 advanced mode",
        "feedback analysis": "follow-up note analysis",
        "Feedback Analysis": "Follow-up Note Analysis",
        "health record feedback": "follow-up note outputs",
        "Retrieval-Augmented Summary and Feedback": "Retrieval-Augmented Summary and Follow-up Note",
        "feedback geared toward clinicians": "generated follow-up notes geared toward reviewer oversight",
        "Evidence, summary, feedback generated": "Evidence, summary, follow-up note generated",
        "generated summaries and feedback": "generated summaries and follow-up notes",
        "summaries and feedback": "summaries and follow-up notes",
        "offer feedback": "produce retrieval-supported follow-up notes",
        "structured feedback and clinician-oriented summary": "structured follow-up notes and patient-oriented summaries",
        "ultimate feedback cannot exceed": "generated note cannot exceed",
        "medical recommendations and not the conclusive ones": "medical decision-making and not a conclusive clinical output",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.replace("reasoning.`", "reasoning.")
    text = text.replace("lab reports.The", "lab reports. The")
    text = text.replace("data.Rather", "data. Rather")
    text = text.replace("report.The", "report. The")
    text = text.replace("inputs.The", "inputs. The")
    text = text.replace("analysis.Its", "analysis. Its")
    text = text.replace("clinical pictures", "medical images")
    text = text.replace("scanned clinical notes", "scanned clinical documents")
    text = text.replace("display clinical summaries and feedback", "display patient summaries and generated follow-up notes")
    text = text.replace("summarization and summarization and follow-up note generation system", "summarization and follow-up note generation system")
    text = text.replace("with retrieval-supported follow-up note", "with a retrieval-supported follow-up note")
    text = text.replace("and produces patient summaries and retrieval-supported follow-up note through", "and produces patient summaries and retrieval-supported follow-up notes through")
    text = text.replace("follow-up notess", "follow-up notes")
    text = text.replace("as a fully-supervised cross-modal attention training using a paired image-text-label dataset was not available.", "because fully supervised cross-modal attention training would require a paired image-text-label dataset that was not available.")
    text = text.replace("produces patient summaries and retrieval-supported follow-up note", "produces patient summaries and retrieval-supported follow-up notes")
    text = text.replace("Rather Model-3", "Rather, Model-3")
    text = text.replace("not available.Rather", "not available. Rather")
    text = text.replace("Thus, the approach of is", "Thus, the approach is")
    text = text.replace("patient summaries and generated follow-up note are generated", "patient summaries and generated follow-up notes are produced")
    text = text.replace("The final output of Model-3 includes retrieval-supported follow-up note.", "The final output of Model-3 includes a retrieval-supported follow-up note.")
    text = text.replace("in addition to generated follow-up note", "in addition to a generated follow-up note")
    text = text.replace("fully trained cross-modal attention model", "supervised cross-modal-attention model")
    text = text.replace("feedback is solely generated from the model predictions", "generated text is produced solely from model predictions")
    text = text.replace("legacy backend feedback field", "legacy backend output field")
    text = text.replace("structured follow-up note and patient-oriented summary", "structured follow-up notes and patient-oriented summaries")
    text = text.replace("final summaries, and generated follow-up note", "final summaries, and generated follow-up notes")
    text = text.replace("research validation", "research demonstration")
    text = text.replace("technical summarization and decision-support system", "technical summarization assistant")
    text = text.replace("Model-2 provides the functionality to analyze clinical notes.", "Model-2 provides the functionality to analyze scanned documents and doctor-note text.")
    text = text.replace("provide feedback to the physicians", "generate retrieval-supported follow-up notes for review")
    text = text.replace("provide feedback to the physician", "generate a retrieval-supported follow-up note for review")
    text = text.replace("The feedback and patient summary generated", "The follow-up note and patient summary generated")
    text = text.replace("supports the summary and feedback", "supports the summary and follow-up note")
    text = text.replace("This feedback is not intended", "This generated note is not intended")
    text = text.replace("summary and feedback", "summary and follow-up note")
    text = text.replace("summary, feedback, and evidence", "summary, follow-up note, and evidence")
    text = text.replace("feedback, and evidence availability", "follow-up note, and evidence availability")
    return text


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, max_width: int, fnt: ImageFont.ImageFont) -> list[str]:
    lines: list[str] = []
    for raw in text.split("\n"):
        words = raw.split()
        if not words:
            lines.append("")
            continue
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if draw.textbbox((0, 0), candidate, font=fnt)[2] <= max_width:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    fill: str,
    outline: str,
    title_size: int = 30,
    body_size: int = 24,
) -> None:
    x0, y0, x1, y1 = xy
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=5)
    title_font = font(title_size, bold=True)
    body_font = font(body_size)
    y = y0 + 24
    for line in wrap_text(draw, title, x1 - x0 - 48, title_font):
        w = draw.textbbox((0, 0), line, font=title_font)[2]
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=title_font, fill="#24313f")
        y += title_size + 8
    y += 6
    for line in wrap_text(draw, body, x1 - x0 - 48, body_font):
        w = draw.textbbox((0, 0), line, font=body_font)[2]
        draw.text((x0 + (x1 - x0 - w) / 2, y), line, font=body_font, fill="#24313f")
        y += body_size + 8


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill: str = "#607080", width: int = 6) -> None:
    draw.line([start, end], fill=fill, width=width)
    sx, sy = start
    ex, ey = end
    if ex >= sx:
        pts = [(ex, ey), (ex - 22, ey - 12), (ex - 22, ey + 12)]
    else:
        pts = [(ex, ey), (ex + 22, ey - 12), (ex + 22, ey + 12)]
    draw.polygon(pts, fill=fill)


def generate_architecture_diagram(path: Path) -> None:
    img = Image.new("RGB", (2400, 1450), "#fbfcfd")
    d = ImageDraw.Draw(img)
    d.text((420, 58), "Final Multimodal AI Assistant Architecture", font=font(54, True), fill="#24313f")
    d.text((650, 124), "Stable prototype with explicit sub-branches and an optional V4 experimental layer", font=font(27), fill="#55606d")

    blue_fill, blue = "#eaf3fb", "#3675ad"
    teal_fill, teal = "#e7f5f2", "#229f92"
    green_fill, green = "#eaf6ea", "#4c9141"
    amber_fill, amber = "#fff2d9", "#d88a22"
    rose_fill, rose = "#fbe8ea", "#c95d63"
    gray_fill, gray = "#eef2f5", "#6f7a85"

    box(d, (80, 230, 420, 380), "Brain MRI", "image input", blue_fill, blue, 28, 24)
    box(d, (80, 430, 420, 580), "Chest X-ray", "image input", blue_fill, blue, 28, 24)
    box(d, (80, 700, 420, 850), "Scanned Document", "prescription or lab report", teal_fill, teal, 28, 24)
    box(d, (80, 900, 420, 1050), "Doctor Note Text", "clinical note fields", teal_fill, teal, 28, 24)

    box(d, (560, 260, 1020, 550), "Model-1\nMedical Image Analysis", "1A Brain MRI classifier\n1B Chest X-ray multi-label classifier\nDenseNet-121 checkpoints", blue_fill, blue, 30, 24)
    box(d, (560, 710, 1020, 1040), "Model-2\nClinical Text and Document Understanding", "2A Scanned document OCR/extraction\n2B Doctor-note clinical text\nRule extraction plus weak NER evidence", teal_fill, teal, 29, 23)
    box(d, (1190, 475, 1680, 830), "Model-3\nMultimodal Fusion and RAG-based Summarization", "3A Late fusion of available outputs\n3B Evidence retrieval and summary generation\nGenerated follow-up note for review", green_fill, green, 28, 22)

    box(d, (1840, 270, 2260, 410), "Patient Summary", "retrieval-supported narrative", amber_fill, amber, 28, 23)
    box(d, (1840, 500, 2260, 640), "Generated Follow-up Note", "review text, not clinical advice", amber_fill, amber, 27, 22)
    box(d, (1840, 730, 2260, 870), "Structured JSON", "machine-readable output", amber_fill, amber, 28, 23)

    box(d, (780, 1160, 1710, 1345), "V4 Experimental Enhancement Layer", "V4A YOLO ROI on pseudo-labels   |   V4B weakly supervised NER   |   V4C expanded RAG KB\nOptional enhancement within the three-model architecture", rose_fill, rose, 31, 24)
    box(d, (1830, 1090, 2260, 1265), "cross_attention_v4", "code/readiness only\nfuture paired-data work", gray_fill, gray, 28, 23)

    arrow(d, (420, 305), (560, 355))
    arrow(d, (420, 505), (560, 455))
    arrow(d, (420, 775), (560, 825))
    arrow(d, (420, 975), (560, 930))
    arrow(d, (1020, 405), (1190, 580))
    arrow(d, (1020, 875), (1190, 700))
    arrow(d, (1680, 560), (1840, 340))
    arrow(d, (1680, 650), (1840, 570))
    arrow(d, (1680, 740), (1840, 800))
    arrow(d, (1250, 1160), (1280, 830), fill="#c95d63")
    arrow(d, (1710, 1245), (1830, 1180), fill="#6f7a85")

    d.text((650, 1390), "Technical validation reports execution and component metrics only; no clinical validation or paired-patient claim is made.", font=font(24), fill="#a6464c")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def generate_claim_status_figure(path: Path) -> None:
    img = Image.new("RGB", (2200, 1200), "#fbfcfd")
    d = ImageDraw.Draw(img)
    d.text((520, 70), "Best Available Versions and Claim Status", font=font(52, True), fill="#24313f")

    cards = [
        ((110, 225, 545, 435), "Model-1A\nBrain MRI", "Accuracy 0.9375\nMacro F1 0.9359\nStable final", "#e7f5f2", "#229f92"),
        ((635, 225, 1070, 435), "Model-1B\nChest X-ray", "Macro AUROC 0.8133\nTuned macro F1 0.2862\nStable final", "#e7f5f2", "#229f92"),
        ((1160, 225, 1595, 435), "Model-2A\nOCR/Extraction", "241 document records\nOCR success 1.0000\nStable final", "#eaf3fb", "#3675ad"),
        ((1685, 225, 2120, 435), "Model-2B\nDoctor-note text", "Classifier accuracy 0.3209\nWeak-label Entity-F1 0.0148\nStable branch", "#eaf3fb", "#3675ad"),
        ((250, 645, 685, 860), "Model-3\nFusion + RAG", "Summary generation 1.0000\nFollow-up note generation 1.0000\nPipeline-based", "#eaf6ea", "#4c9141"),
        ((885, 645, 1320, 860), "V4A\nYOLO ROI", "mAP50 0.8357\nPseudo-label evidence\nExperimental", "#fbe8ea", "#c95d63"),
        ((1520, 645, 1955, 860), "V4B / V4C\nNER + KB", "Weak-label Entity-F1 0.99628\nExpanded KB 12,000 records\nExperimental", "#fbe8ea", "#c95d63"),
    ]
    for coords, title, body, fill, outline in cards:
        box(d, coords, title, body, fill, outline, 31, 26)
    d.text((270, 1040), "Stable final system: Model-1 + Model-2 + Model-3. V4 components are optional experimental evidence; cross_attention_v4 remains future work.", font=font(29), fill="#a6464c")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def build_tables() -> str:
    return r"""
% Generated by scripts/careful_report_update.py from verified final-revision metrics.
\begin{table}[H]
\centering
\caption{Selected final technical results and safe interpretation.}
\label{tab:revised_final_results_summary}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|p{0.25\textwidth}|p{0.34\textwidth}|p{0.34\textwidth}|}
\hline
\textbf{Component} & \textbf{Verified result} & \textbf{Safe interpretation} \\
\hline
Model-1A Medical Image Analysis: Brain MRI & Accuracy 0.9375; macro F1 0.9359; train-validation accuracy gap 0.0514 & Held-out technical classification result. No completed k-fold cross-validation artefact was found in the repository. \\
\hline
Model-1B Medical Image Analysis: Chest X-ray & Macro AUROC 0.8133; micro AUROC 0.8377; tuned macro F1 0.2862; tuned micro F1 0.3366 & Multi-label technical image-classification result on imbalanced labels. AUROC and tuned F1 are reported instead of a single accuracy claim. \\
\hline
Model-2A Clinical Text and Document Understanding: OCR/Extraction & 241 scanned-document records; OCR success 1.0000; entity extraction success 0.8755; structured JSON completion 1.0000 & Technical document-processing evidence from scanned prescriptions and lab reports. This is not a trained OCR neural model claim. \\
\hline
Model-2B Clinical Text and Document Understanding: Doctor-note Clinical Text & 4,966 usable notes; 29 retained specialties; classifier accuracy 0.3209; macro F1 0.3468; weak NER entity F1 0.0148 & Doctor-note text branch with classification and weak-label NER evidence. The weak NER result is not expert-annotated clinical NER. \\
\hline
Model-3A/3B Multimodal Fusion and RAG-based Summarization & Fusion output success 1.0000; patient summary generation 1.0000; follow-up note generation 1.0000; evidence availability 1.0000; average evidence count 3.71 & Pipeline-based late fusion plus retrieval-supported summarization. No end-to-end paired multimodal training artefact is claimed. \\
\hline
100-case technical validation & 100 completed; 0 failed; technical execution success 1.0000 & Execution coverage across planned input combinations. This is not diagnostic accuracy and not clinical validation. \\
\hline
\end{tabular}%
}
\end{table}

\begin{table}[H]
\centering
\caption*{Table 7.1 (continued): experimental and future-work evidence.}
\resizebox{\textwidth}{!}{%
\begin{tabular}{|p{0.25\textwidth}|p{0.34\textwidth}|p{0.34\textwidth}|}
\hline
\textbf{Component} & \textbf{Verified result} & \textbf{Safe interpretation} \\
\hline
V4 Experimental Enhancement Layer & V4A YOLO ROI mAP50 0.8357; V4B weak-label Entity-F1 0.99628; V4C expanded RAG KB 12,000 records & Optional experimental evidence based on pseudo-labels, weak labels, and an expanded retrieval resource within the three-model architecture. \\
\hline
cross\_attention\_v4 & Code/readiness checks only; no non-empty trained checkpoint and no paired same-patient image-text-label dataset & Future-work candidate, not a final trained component. \\
\hline
\end{tabular}%
}
\end{table}
""".strip() + "\n"


def update_report_sources(changed: list[str]) -> None:
    abstract = REPORT / "core" / "abstract.tex"
    abstract_text = r"""\section*{Abstract}

Manual interpretation of patient records, medical images, prescribing information, laboratory reports, and clinical notes is time consuming, especially when healthcare information is fragmented across multiple formats. The aim of this thesis was to develop and implement a localized research prototype of a Multimodal AI Assistant for automated health record summarization and follow-up note analysis. The prototype processes medical images, scanned clinical documents, and doctor-note text to produce structured patient summaries and retrieval-supported follow-up notes for review.

The proposed system consists of three main model groups. Model-1, Medical Image Analysis, contains Model-1A for Brain MRI classification and Model-1B for Chest X-ray multi-label classification. Model-2, Clinical Text and Document Understanding, contains Model-2A for scanned document OCR/extraction and Model-2B for doctor-note clinical text processing. Model-3, Multimodal Fusion and RAG-based Summarization, performs late fusion of available outputs and uses local retrieval to support patient-summary and follow-up-note generation. A Streamlit interface was built to enable local demonstration of image, document, doctor-note, and fusion workflows.

The stable mode is the main final working prototype. A V4 Experimental Enhancement Layer was also built to investigate pseudo-labeled YOLO ROI detection, weakly supervised NER using BERT, and an expanded RAG knowledge base. V4 is reported only as an optional experimental layer within the three-model architecture. The system was tested technically using image-model metrics, threshold tuning, document-pipeline testing, Model-2B text-branch evaluation, Model-3 output-completion checks, and a 100-case end-to-end execution run. These results show that the prototype can run in image-only, document-only, doctor-note-only, and fusion cases, but they do not constitute clinical validation and the system must not be used instead of medical expertise.

\vspace{1cm}

\textbf{Keywords:} Multimodal Artificial Intelligence, Health Record Summarization, Optical Character Recognition (OCR), Natural Language Processing (NLP), Clinical Data Fusion, Retrieval-Supported Follow-up Note Generation, Biomedical Deep Learning
\pagebreak
"""
    write_text(abstract, abstract_text, changed)

    for tex_path in REPORT.glob("**/*.tex"):
        if "report_backup_before_careful_update" in tex_path.parts:
            continue
        if tex_path == abstract:
            continue
        write_text(tex_path, safe_report_replacements(read_text(tex_path)), changed)

    # Chapter 1 objective and scope wording.
    ch1 = REPORT / "chapters" / "chapter_1.tex"
    text = read_text(ch1)
    text = text.replace(
        "    \\item Image-based Prediction and Evidence Integration: To predict from medical images and integrate these prediction results with document-derived evidence for assistant-style summarisation and feedback generation.\n"
        "    \\item Retrieval-Supported Follow-up Note Generation: To generate synthesized findings and decision-support review text in the form of factual, structured reports and hence less manual data entry as well as more consistent health record organization. The general objective of the system is to deliver structured review text in a report format focusing on doctors. This method of working with images aids in minimizing hand data entry.\n"
        "    \\item Evaluation \\& Reporting: To strictly evaluate the system performance with the help of applicable quantitative indicators (entity-F1, AUC) and technical artefact review and model-specific metrics.\n",
        "    \\item Image-based Prediction and Evidence Integration: To generate technical image predictions and integrate these prediction results with document-derived evidence for assistant-style summarization and follow-up note generation.\n"
        "    \\item Retrieval-Supported Follow-up Note Generation: To generate factual, structured review text that organizes model findings and retrieved evidence without providing diagnosis, treatment direction, or clinical instruction.\n"
        "    \\item Evaluation \\& Reporting: To evaluate system performance with model-specific quantitative indicators, technical artefact checks, and cautious claim boundaries.\n",
    )
    ch1_block = r"""
\section{Final Prototype Scope Clarification}

The final thesis contribution is a local research prototype, not a deployed clinical product. The stable system is reported as three model groups: Model-1 Medical Image Analysis, Model-2 Clinical Text and Document Understanding, and Model-3 Multimodal Fusion and RAG-based Summarization. Model-2 is split into Model-2A Scanned Document OCR/Extraction and Model-2B Doctor-Note Clinical Text. Model-3 is split into Model-3A Late Fusion and Model-3B Evidence Retrieval and Summary Generation.

The V4 layer is reported separately as an experimental enhancement layer containing V4A YOLO ROI, V4B weakly supervised NER, and V4C expanded RAG knowledge base work. It does not alter the three-model architecture. Cross-modal attention remains future work because the available data do not provide verified same-patient image-text-label pairs for supervised multimodal training.
"""
    text = replace_between(
        text,
        "% FINAL_CAREFUL_UPDATE_CH1_SCOPE_START",
        "% FINAL_CAREFUL_UPDATE_CH1_SCOPE_END",
        ch1_block,
        anchor="\\section{Scopes and Challenges}",
        before=True,
    )
    write_text(ch1, text, changed)

    # Chapter 3 targeted safety wording.
    ch3 = REPORT / "chapters" / "Chapter_3.tex"
    text = read_text(ch3)
    text = text.replace(
        "    \\item The system shall generate patient summaries and retrieval-supported follow-up note for review.",
        "    \\item The system shall generate patient summaries and retrieval-supported follow-up notes for review, without presenting them as diagnosis, treatment direction, or clinical instruction.",
    )
    text = text.replace(
        "The system also has health and safety concerns. Although the system aids information review, it may introduce new concerns where users may place too much trust in the system. The reports and the interface state that the system aids information review, and offers no clinical conclusions, neither offered diagnoses, treatment directions, nor offers clinical instruction of an urgent or emergent nature.",
        "The system also has health and safety concerns. Although the system aids information review, it may introduce new concerns if users place too much trust in the generated outputs. The report and the interface state that the system supports technical information organization only and does not provide clinical conclusions, diagnosis, treatment direction, or urgent clinical instruction.",
    )
    text = text.replace(
        "Professional responsibility requires transparency, caution, and documentation. The system should clearly communicate its limitations, avoid unsupported clinical instruction, and preserve logs or structured outputs for review. Human oversight is necessary throughout the workflow.",
        "Professional responsibility requires transparency, caution, and documentation. The system should clearly communicate its limitations, avoid unsupported clinical instruction, and preserve logs or structured outputs for review. Human oversight is necessary throughout the workflow.",
    )
    text = text.replace(
        "Over-trust in generated feedback & Users may treat prototype output as clinical instruction & Clearly state that the system is for decision-support and technical demonstration only \\\\",
        "Over-trust in generated outputs & Users may treat prototype output as clinical instruction & Clearly state that the system is for technical information organization and demonstration only \\\\",
    )
    write_text(ch3, text, changed)

    # Chapter 4 architecture and naming block.
    ch4 = REPORT / "chapters" / "Chapter_4.tex"
    text = read_text(ch4)
    text = text.replace("figures/fig01_complete_system_architecture.png", "figures/revised_system_architecture.png")
    text = text.replace(
        "\\caption{Complete architecture of the proposed multimodal AI assistant}",
        "\\caption{Revised final architecture of the multimodal AI assistant, showing the stable model branches and optional V4 experimental layer.}",
    )
    ch4_block = r"""
\subsection{Final Model Naming and Architecture Boundary}

For final reporting, Model-1 is named \textit{Medical Image Analysis} and is split into Model-1A Brain MRI and Model-1B Chest X-ray. Model-2 is named \textit{Clinical Text and Document Understanding} and is split into Model-2A Scanned Document OCR/Extraction and Model-2B Doctor-Note Clinical Text. Model-3 is named \textit{Multimodal Fusion and RAG-based Summarization} and is split into Model-3A Late Fusion and Model-3B Evidence Retrieval and Summary Generation.

The V4 Experimental Enhancement Layer is separate from the stable model numbering. It contains V4A YOLO ROI, V4B weakly supervised NER, and V4C expanded RAG knowledge base work. The cross\_attention\_v4 code path is retained as future-work readiness only, because a verified same-patient paired image-text-label dataset and a trained checkpoint were not available.
"""
    text = replace_between(
        text,
        "% FINAL_CAREFUL_UPDATE_CH4_NAMING_START",
        "% FINAL_CAREFUL_UPDATE_CH4_NAMING_END",
        ch4_block,
        anchor="\\label{fig:complete_system_architecture}\n\\end{figure}",
    )
    write_text(ch4, text, changed)

    # Chapter 5 dataset branch insertion.
    ch5 = REPORT / "chapters" / "Chapter_5.tex"
    text = read_text(ch5)
    ch5_block = r"""
\subsection{Doctor-Note Clinical Text Dataset for Model-2B}

Model-2B uses doctor-note clinical text as a separate branch from scanned document OCR. The local processed doctor-note dataset contains 4,999 raw MTSamples rows, from which 4,966 usable notes were retained after filtering. The specialty classifier used 4,841 rows across 29 retained specialties, split into 2,904 training rows, 968 validation rows, and 969 test rows. The doctor-note inference evaluation used 250 processed cases and produced structured outputs for the tested records.

% TODO_CITATION_MTSAMPLES: Add a verified citation or dataset-access reference for the MTSamples clinical-note dataset before final submission.
This branch should not be interpreted as an expert-annotated NER dataset. Its NER labels are weak labels created by automated rules and dictionaries, and the reported NER score is therefore a weak-label technical result.
"""
    text = replace_between(
        text,
        "% FINAL_CAREFUL_UPDATE_CH5_MODEL2B_START",
        "% FINAL_CAREFUL_UPDATE_CH5_MODEL2B_END",
        ch5_block,
        anchor="\\subsection{Multimodal Data Usage for Model-3}",
        before=True,
    )
    write_text(ch5, text, changed)

    # Chapter 6 implementation branch insertion.
    ch6 = REPORT / "chapters" / "Chapter_6.tex"
    text = read_text(ch6)
    ch6_block = r"""
\subsection{Model-2B Doctor-Note Clinical Text Branch}

Model-2B processes structured doctor-note text fields separately from scanned document OCR. The branch accepts chief complaint, note text, relevant history, current medication or allergy information, report-related concern, and optional urgency level. It then produces extracted weak-label entities, a patient-summary text field, and structured JSON for downstream fusion.

The doctor-note classifier and weak NER components are reported as technical text-processing modules. The classifier is evaluated by specialty classification metrics, while the weak NER module is evaluated against automatically produced weak labels. These labels were not created or verified by clinical experts, so the result is not reported as expert clinical NER.
"""
    text = replace_between(
        text,
        "% FINAL_CAREFUL_UPDATE_CH6_MODEL2B_START",
        "% FINAL_CAREFUL_UPDATE_CH6_MODEL2B_END",
        ch6_block,
        anchor="\\section{Model-3 Fusion and RAG Implementation}",
        before=True,
    )
    ch6_compat = r"""
\subsection{Output Field Compatibility}

Some saved JSON outputs and backend code paths retain legacy field names for compatibility with earlier validation artefacts. In the report-facing text and interface labels, this output is described as a generated follow-up note or retrieval-supported review text. This naming avoids implying external validation, clinical instruction generation, or replacement of medical judgment.
"""
    text = replace_between(
        text,
        "% FINAL_CAREFUL_UPDATE_CH6_COMPAT_START",
        "% FINAL_CAREFUL_UPDATE_CH6_COMPAT_END",
        ch6_compat,
        anchor="\\section{Streamlit Interface and Validation Workflow}",
        before=True,
    )
    write_text(ch6, text, changed)

    # Chapter 7 result table and additional branch sections.
    ch7 = REPORT / "chapters" / "Chapter_7.tex"
    text = read_text(ch7)
    text = text.replace("figures/fig15_best_version_claim_status_summary.png", "figures/fig15_best_version_claim_status_summary_revised.png")
    text = text.replace("\\subsection{Retrieval-supported Follow-up Note}", "\\subsection{Retrieval-Supported Follow-up Note}")
    tables_block = r"""
\section{Verified Final Result Tables}

The tables below consolidate the verified final metrics used in this revision. They separate Model-2A from Model-2B, separate stable Model-3 pipeline evidence from V4 experiments, and keep the 100-case result framed as technical execution success rather than accuracy.

\input{tables/revised_final_results_tables}
"""
    text = replace_between(
        text,
        "% FINAL_CAREFUL_UPDATE_CH7_TABLES_START",
        "% FINAL_CAREFUL_UPDATE_CH7_TABLES_END",
        tables_block,
        anchor="\\section{Model-1 Image Classification Results}",
        before=True,
    )
    cv_block = r"""
\subsection{Cross-Validation Audit Status}

A repository audit was completed before this report revision to check whether Model-1 k-fold cross-validation artefacts were available. The audit found held-out performance metrics, checkpoint evidence, and overfitting/generalization analysis, but no completed fold-level checkpoints, fold-level metric files, or aggregate k-fold cross-validation result for Model-1. Therefore, this report does not claim completed k-fold cross-validation for Model-1. The Model-1 evidence is described as held-out technical performance with overfitting/generalization checks, and k-fold cross-validation remains future work.
"""
    text = replace_between(
        text,
        "% FINAL_CAREFUL_UPDATE_CH7_CV_START",
        "% FINAL_CAREFUL_UPDATE_CH7_CV_END",
        cv_block,
        anchor="Results from the classifiers are reported separately as they are both used to complete different tasks related to images.",
    )
    model2b_block = r"""
\subsection{Model-2B Doctor-Note Clinical Text Results}

Model-2B evaluates the doctor-note clinical text branch separately from scanned document OCR. The processed text dataset contains 4,966 usable notes from 4,999 raw MTSamples rows, with 4,841 rows used for the specialty classifier across 29 retained specialties. The train, validation, and test split was 2,904/968/969. The specialty classifier achieved a test accuracy of 0.3209 and a macro F1 score of 0.3468.

% TODO_CITATION_MTSAMPLES: Add a verified citation or dataset-access reference for the MTSamples clinical-note dataset before final submission.
The doctor-note weak NER evaluation used a transformer BERT mode, with token accuracy of 0.9937 and entity F1 of 0.0148. This weak NER result should be interpreted cautiously because the labels were generated from rules and dictionaries rather than expert clinical annotation.
"""
    text = replace_between(
        text,
        "% FINAL_CAREFUL_UPDATE_CH7_MODEL2B_START",
        "% FINAL_CAREFUL_UPDATE_CH7_MODEL2B_END",
        model2b_block,
        anchor="\\section{Model-3 Fusion and RAG Results}",
        before=True,
    )
    model3_boundary = r"""
The metric formerly stored under a legacy backend feedback field is reported here as follow-up note generation. This is a report-facing naming correction only; it does not imply external validation, diagnostic direction, treatment direction, or clinical approval.
"""
    text = replace_between(
        text,
        "% FINAL_CAREFUL_UPDATE_CH7_MODEL3_BOUNDARY_START",
        "% FINAL_CAREFUL_UPDATE_CH7_MODEL3_BOUNDARY_END",
        model3_boundary,
        anchor="\\subsection{Retrieval-Supported Follow-up Note}",
    )
    write_text(ch7, text, changed)

    # Chapter 8 final claim boundary.
    ch8 = REPORT / "chapters" / "Chapter_8.tex"
    text = read_text(ch8)
    ch8_block = r"""
\subsection{Final Claim Boundary}

No part of this thesis claims externally validated generated notes, clinical deployment readiness, treatment direction, diagnostic direction, or replacement of doctors. The completed work is a technical prototype with component-level evidence, output-completion checks, and clearly documented limitations. Model-3 uses late fusion and retrieval-supported summarization, while cross-modal attention remains future work until a true paired patient-level multimodal dataset and a trained checkpoint are available.
"""
    text = replace_between(
        text,
        "% FINAL_CAREFUL_UPDATE_CH8_BOUNDARY_START",
        "% FINAL_CAREFUL_UPDATE_CH8_BOUNDARY_END",
        ch8_block,
        anchor="\\subsection{Dataset, Pairing, and Annotation Limitations}",
        before=True,
    )
    write_text(ch8, text, changed)


def update_outputs_and_ui(changed: list[str]) -> None:
    # Report-facing generated summaries only; backup and raw compatibility keys are preserved.
    report_files = [
        OUT / "final_model_results_summary.md",
        OUT / "final_model_results_summary.json",
        OUT / "model2_model3_result_check.md",
        OUT / "final_model_results_tables.tex",
        OUT / "full_thesis_recheck_audit.md",
        OUT / "full_thesis_recheck_audit.json",
        OUT / "component_truth_audit.md",
        OUT / "component_truth_audit.json",
        OUT / "what_to_put_in_report.md",
        OUT / "FINAL_THESIS_RECHECK_MASTER_REPORT.md",
    ]
    for path in report_files:
        if not path.exists():
            continue
        text = read_text(path)
        text = safe_report_replacements(text)
        text = text.replace("Doctor feedback generation", "Follow-up note generation")
        text = text.replace("doctor feedback generation", "follow-up note generation")
        text = text.replace("Generated follow-up note generation", "Follow-up note generation")
        text = text.replace("generated follow-up note generation", "follow-up note generation")
        text = text.replace("Feedback generation", "Follow-up note generation")
        text = text.replace("Feedback |", "Follow-up note |")
        write_text(path, text, changed)

    app = ROOT / "app.py"
    text = read_text(app)
    text = text.replace('_render_value("Doctor feedback", fusion_output.get("doctor_oriented_feedback", fusion_output.get("doctor_feedback")))', '_render_value("Generated follow-up note", fusion_output.get("doctor_oriented_feedback", fusion_output.get("doctor_feedback")))')
    write_text(app, text, changed)


def write_audits(changed: list[str]) -> None:
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_text(
        OUT / "revised_system_architecture_caption.txt",
        "Revised final architecture of the multimodal AI assistant, showing Model-1A/1B, Model-2A/2B, Model-3A/3B, and the optional V4 experimental enhancement layer. The figure explicitly frames V4 as experimental and cross_attention_v4 as future-work readiness only.\n",
        changed,
    )

    figure_audit = f"""# Figure Update Audit

Generated: {generated}

## Added or Replaced

- Added `report_source/figures/revised_system_architecture.png` and `outputs/final_revision/revised_system_architecture.png`.
- Replaced the Chapter 4 top-level architecture reference from `figures/fig01_complete_system_architecture.png` to `figures/revised_system_architecture.png`.
- Added `report_source/figures/fig15_best_version_claim_status_summary_revised.png` and `outputs/final_revision/fig15_best_version_claim_status_summary_revised.png`.
- Replaced the Chapter 7 claim-status reference from `figures/fig15_best_version_claim_status_summary.png` to `figures/fig15_best_version_claim_status_summary_revised.png`.

## Retained

- Model-specific result figures such as Brain MRI confusion matrix, X-ray AUROC, threshold/version comparisons, 100-case distribution, and component output success were retained.
- `fig14_model_component_output_success.png` was retained because the image itself states that the counts are execution coverage, not shared diagnostic accuracy.

## Removed

- No figure files were deleted. Older figure files remain in `report_source/figures` for traceability, but the report no longer references the replaced architecture or claim-status images.
"""
    write_text(OUT / "figure_update_audit.md", figure_audit, changed)

    citation_audit = f"""# Citation Safety Audit

Generated: {generated}

## Existing Citations Reused

- Brain MRI dataset: `msoud_nickparvar_2026`
- NIH Chest X-ray dataset: `nih_chestxray_dataset`, `wang2017chestxray8`
- OCR/prescription datasets: `nadaarfaoui_ocr_prescriptions`, `mamun1113_prescription_bd`
- Biomedical/clinical NLP context: `Gao2024`, `Lee2020`
- Multimodal and cross-modal context: `Huang2020`, `Ghosh2024`, `AlSaad2024`
- Safety and trust context: `Wu2024`, `Yang2024`, `Schouten2025`

## TODO Citation Markers Added

- `TODO_CITATION_MTSAMPLES`: add a verified citation or dataset-access reference for the MTSamples clinical-note dataset before final submission.

## Fabrication Check

- No new bibliography entries were invented.
- No invented DOI, author list, venue, or year was added.
- Unverified dataset references were marked as TODO rather than cited with fabricated metadata.
"""
    write_text(OUT / "citation_safety_audit.md", citation_audit, changed)

    language_audit = f"""# Thesis Language Sweep Report

Generated: {generated}

## Report-Facing Wording

- Replaced legacy report-facing feedback wording with `generated follow-up note`, `retrieval-supported follow-up note`, or `retrieval-supported review text`.
- Updated the Streamlit visible label to `Generated follow-up note`.
- Preserved doctor-note wording where it refers to the Model-2B input branch, because the final naming requires `Doctor-Note Clinical Text`.

## Claim Boundaries

- Added explicit statements that the 100-case run is technical execution success, not accuracy.
- Added explicit Model-1 cross-validation status: no completed k-fold cross-validation artefacts found.
- Added explicit V4 status as an optional experimental enhancement layer within the three-model architecture.
- Added explicit cross_attention_v4 status: code/readiness only and future work.

## Compatibility Note

- Legacy backend keys such as `doctor_feedback` or `doctor_feedback_generation` may remain inside code, saved JSON, or audit-key names for compatibility with existing validation artefacts. These are not used as report-facing labels.
"""
    write_text(OUT / "thesis_language_sweep_report.md", language_audit, changed)


def update_change_log(changed: list[str]) -> None:
    log = OUT / "careful_update_change_log.md"
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entries = "\n".join(f"- `{item}`" for item in changed)
    text = f"""# Careful Thesis Report Update Change Log

Started: 2026-08-02
Last updated: {generated}

## Backup

- Pre-edit report backup: `outputs/final_revision/report_backup_before_careful_update/report_source`

## Files Edited or Generated

{entries}

## Major Changes

- Replaced legacy report-facing feedback wording with generated follow-up note / retrieval-supported review wording.
- Added Model-1 cross-validation audit status to report results wording.
- Added Model-2A/Model-2B split and Model-3A/Model-3B split to report source.
- Replaced the architecture figure and claim-status figure references with revised images.
- Added final result tables to `report_source/tables/revised_final_results_tables.tex`.
- Added citation TODO marker for MTSamples instead of fabricating a source.
- Preserved old figure files and backup source for traceability.
"""
    write_text(log, text, [])


def main() -> None:
    changed: list[str] = []
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    arch_report = FIGURES / "revised_system_architecture.png"
    arch_out = OUT / "revised_system_architecture.png"
    claim_report = FIGURES / "fig15_best_version_claim_status_summary_revised.png"
    claim_out = OUT / "fig15_best_version_claim_status_summary_revised.png"
    generate_architecture_diagram(arch_report)
    generate_claim_status_figure(claim_report)
    shutil.copy2(arch_report, arch_out)
    shutil.copy2(claim_report, claim_out)
    changed.extend(
        [
            str(arch_report.relative_to(ROOT)),
            str(arch_out.relative_to(ROOT)),
            str(claim_report.relative_to(ROOT)),
            str(claim_out.relative_to(ROOT)),
        ]
    )

    write_text(TABLES / "revised_final_results_tables.tex", build_tables(), changed)
    update_report_sources(changed)
    update_outputs_and_ui(changed)
    write_audits(changed)
    update_change_log(changed)

    print(json.dumps({"changed": changed}, indent=2))


if __name__ == "__main__":
    main()
