from __future__ import annotations

import shutil
import struct
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "outputs" / "thesis_figures"
OUT_DIR = ROOT / "outputs" / "thesis_slides"
PPTX_PATH = OUT_DIR / "Multimodal_AI_Thesis_Defense_7slides.pptx"
GUIDE_PATH = OUT_DIR / "SLIDE_SPEAKER_GUIDE.md"
MEDIA_DIR = OUT_DIR / "_pptx_media"

EMU_PER_INCH = 914400
SLIDE_W = int(13.333333 * EMU_PER_INCH)
SLIDE_H = int(7.5 * EMU_PER_INCH)

INK = "24323F"
MUTED = "5E6A75"
BLUE = "3A6EA8"
TEAL = "2A9D8F"
GREEN = "4C8F45"
ORANGE = "DC9228"
RED = "C85C5C"
BG = "F7FAFC"
LIGHT_BLUE = "E8F1FA"
LIGHT_TEAL = "E4F4F1"
LIGHT_ORANGE = "FFF0D8"
LIGHT_RED = "F8E6E6"
LIGHT_GRAY = "EDF2F7"
WHITE = "FFFFFF"


def emu(inches: float) -> int:
    return int(inches * EMU_PER_INCH)


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", header[16:24])


def fit_image(path: Path, x: float, y: float, w: float, h: float) -> tuple[int, int, int, int]:
    img_w, img_h = png_size(path)
    box_w, box_h = emu(w), emu(h)
    scale = min(box_w / img_w, box_h / img_h)
    draw_w = int(img_w * scale)
    draw_h = int(img_h * scale)
    draw_x = emu(x) + int((box_w - draw_w) / 2)
    draw_y = emu(y) + int((box_h - draw_h) / 2)
    return draw_x, draw_y, draw_w, draw_h


def solid_fill(color: str | None) -> str:
    if color is None:
        return "<a:noFill/>"
    return f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'


def line_xml(color: str | None, width: int = 19050) -> str:
    if color is None:
        return "<a:ln><a:noFill/></a:ln>"
    return f'<a:ln w="{width}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln>'


def run_xml(text: str, size: int, color: str, bold: bool = False) -> str:
    b = ' b="1"' if bold else ""
    return (
        f'<a:r><a:rPr lang="en-US" sz="{size * 100}"{b}>'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        '<a:latin typeface="Aptos"/></a:rPr>'
        f"<a:t>{escape(text)}</a:t></a:r>"
    )


def paragraph_xml(text: str, size: int, color: str, bold: bool, align: str) -> str:
    return f'<a:p><a:pPr algn="{align}"/>{run_xml(text, size, color, bold)}</a:p>'


@dataclass
class ImageRef:
    path: Path
    media_name: str
    rel_id: str


@dataclass
class Slide:
    title: str
    speaker: str
    shapes: list[str] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
    shape_id: int = 2

    def next_id(self) -> int:
        current = self.shape_id
        self.shape_id += 1
        return current

    def add_text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        lines: list[str],
        size: int = 18,
        color: str = INK,
        bold: bool = False,
        align: str = "l",
        fill: str | None = None,
        line: str | None = None,
        radius: str = "rect",
        margin: int = 91440,
    ) -> None:
        shape_id = self.next_id()
        paragraphs = "".join(paragraph_xml(line_text, size, color, bold, align) for line_text in lines)
        self.shapes.append(
            f"""
            <p:sp>
              <p:nvSpPr>
                <p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/>
                <p:cNvSpPr txBox="1"/>
                <p:nvPr/>
              </p:nvSpPr>
              <p:spPr>
                <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
                <a:prstGeom prst="{radius}"><a:avLst/></a:prstGeom>
                {solid_fill(fill)}
                {line_xml(line)}
              </p:spPr>
              <p:txBody>
                <a:bodyPr wrap="square" lIns="{margin}" tIns="{margin}" rIns="{margin}" bIns="{margin}" anchor="mid"/>
                <a:lstStyle/>
                {paragraphs}
              </p:txBody>
            </p:sp>
            """
        )

    def add_card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        title: str,
        body: list[str],
        fill: str,
        line: str,
        title_size: int = 16,
        body_size: int = 13,
    ) -> None:
        self.add_text(x, y, w, h, [title], title_size, INK, True, "ctr", fill, line, "roundRect", margin=emu(0.08))
        if body:
            self.add_text(x + 0.08, y + 0.42, w - 0.16, h - 0.5, body, body_size, INK, False, "ctr", None, None, "rect", margin=emu(0.04))

    def add_image(self, path: Path, x: float, y: float, w: float, h: float, media_name: str, rel_id: str) -> None:
        draw_x, draw_y, draw_w, draw_h = fit_image(path, x, y, w, h)
        shape_id = self.next_id()
        self.images.append(ImageRef(path=path, media_name=media_name, rel_id=rel_id))
        self.shapes.append(
            f"""
            <p:pic>
              <p:nvPicPr>
                <p:cNvPr id="{shape_id}" name="{escape(path.name)}"/>
                <p:cNvPicPr/>
                <p:nvPr/>
              </p:nvPicPr>
              <p:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </p:blipFill>
              <p:spPr>
                <a:xfrm><a:off x="{draw_x}" y="{draw_y}"/><a:ext cx="{draw_w}" cy="{draw_h}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </p:spPr>
            </p:pic>
            """
        )

    def add_title(self) -> None:
        self.add_text(0.45, 0.22, 12.45, 0.65, [self.title], 28, INK, True, "ctr", None, None, margin=emu(0.02))
        self.add_text(10.65, 7.04, 2.25, 0.28, [self.speaker], 8, MUTED, False, "r", None, None, margin=0)

    def xml(self) -> str:
        self.add_title()
        shape_xml = "\n".join(self.shapes)
        return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{BG}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {shape_xml}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


class DeckBuilder:
    def __init__(self) -> None:
        self.slides: list[Slide] = []
        self.media_counter = 1

    def new_slide(self, title: str, speaker: str) -> Slide:
        slide = Slide(title=title, speaker=speaker)
        self.slides.append(slide)
        return slide

    def add_image(self, slide: Slide, figure_name: str, x: float, y: float, w: float, h: float) -> None:
        path = FIGURES / figure_name
        media_name = f"image{self.media_counter}.png"
        rel_id = f"rId{len(slide.images) + 2}"
        self.media_counter += 1
        slide.add_image(path, x, y, w, h, media_name, rel_id)


def build_deck() -> DeckBuilder:
    deck = DeckBuilder()

    s1 = deck.new_slide("Multimodal AI Assistant for Automated Health Record Summarization and Feedback Analysis", "Opening: Syed Enamul Karim")
    s1.add_text(0.7, 1.05, 11.95, 0.55, ["Thesis Defense Presentation"], 22, BLUE, True, "ctr", None, None)
    team = [
        ("Syed Enamul Karim", "23241052"),
        ("Md. Hasin Saleh Alvi", "24141161"),
        ("Sabbir Bhuyan", "22101731"),
        ("Golam Mohammad Rahi", "24141248"),
    ]
    for i, (name, sid) in enumerate(team):
        x = 0.75 + i * 3.08
        s1.add_card(x, 2.0, 2.65, 1.2, name, [sid], LIGHT_BLUE if i % 2 == 0 else LIGHT_TEAL, BLUE if i % 2 == 0 else TEAL, 13, 14)
    s1.add_card(1.1, 4.05, 3.25, 1.25, "Stable System", ["Model-1 image analysis", "Model-2 OCR/rules", "Model-3 RAG fusion"], LIGHT_TEAL, TEAL)
    s1.add_card(5.05, 4.05, 3.25, 1.25, "Advanced Layer", ["V4 YOLO ROI", "BERT weak-label NER", "Experimental evidence"], LIGHT_RED, RED)
    s1.add_card(9.0, 4.05, 3.25, 1.25, "Validation Scope", ["100-case technical run", "Not clinical validation", "Decision-support prototype"], LIGHT_ORANGE, ORANGE)
    s1.add_text(0.9, 6.32, 11.6, 0.38, ["Department of Computer Science and Engineering, Brac University | Fall 2025"], 13, MUTED, False, "ctr", None, None)

    s2 = deck.new_slide("Problem, Objective and System Architecture", "Presenter: Syed Enamul Karim")
    s2.add_text(0.55, 1.05, 3.35, 4.9, [
        "Problem",
        "- Patient records are multimodal: MRI, X-ray, prescriptions, lab reports.",
        "- Manual review is slow and error-prone.",
        "",
        "Objective",
        "- Build a localized assistant for summarization and feedback analysis.",
        "- Combine image prediction, document extraction, and RAG fusion.",
        "",
        "Scope",
        "- Technical decision-support prototype.",
        "- Not a replacement for doctors."
    ], 13, INK, False, "l", WHITE, BLUE)
    deck.add_image(s2, "fig01_complete_system_architecture.png", 4.15, 1.05, 8.55, 5.35)
    s2.add_text(4.25, 6.55, 8.35, 0.34, ["Stable final system: Model-1 + Model-2 + Model-3. V4 is optional experimental evidence."], 12, RED, False, "ctr", None, None)

    s3 = deck.new_slide("Model-1: Brain MRI and Chest X-ray Analysis", "Presenter: Md. Hasin Saleh Alvi")
    deck.add_image(s3, "fig02_model1_image_pipeline.png", 0.55, 1.0, 12.2, 3.65)
    s3.add_card(0.8, 5.0, 3.35, 1.25, "Brain MRI final_v2", ["Accuracy 0.9375", "Macro F1 0.9359", "4-class softmax"], LIGHT_TEAL, TEAL)
    s3.add_card(4.95, 5.0, 3.35, 1.25, "Chest X-ray large_v2", ["Macro AUROC 0.8133", "Micro AUROC 0.8377", "14-label sigmoid"], LIGHT_BLUE, BLUE)
    s3.add_card(9.1, 5.0, 3.35, 1.25, "Threshold tuning", ["Tuned macro F1 0.2862", "Tuned micro F1 0.3366", "Improved precision tradeoff"], LIGHT_ORANGE, ORANGE)

    s4 = deck.new_slide("Model-2: OCR and Document Understanding", "Presenter: Sabbir Bhuyan")
    deck.add_image(s4, "fig05_model2_document_pipeline.png", 0.55, 1.0, 7.35, 3.45)
    s4.add_text(8.1, 1.08, 4.55, 2.1, [
        "Stable Model-2",
        "- OCR text extraction",
        "- Text cleaning",
        "- Rule-based entity extraction",
        "- Structured JSON + document summary",
        "",
        "Final run: 60/60 applicable document outputs"
    ], 13, INK, False, "l", WHITE, TEAL)
    s4.add_card(0.8, 4.85, 3.65, 1.35, "Stable vs V4 sample", ["OCR length: 1345 vs 1249", "Entities: 2 vs 41", "Runtime: 0.48s vs 6.15s"], LIGHT_BLUE, BLUE)
    s4.add_card(4.85, 4.85, 3.65, 1.35, "V4 document layer", ["YOLO ROI pseudo-labels", "BERT weak-label NER", "Advanced comparison only"], LIGHT_RED, RED)
    s4.add_card(8.9, 4.85, 3.65, 1.35, "Defense caveat", ["No stable Model-2 ROC/AUC/F1", "No expert-labeled NER accuracy", "More entities is not automatically better"], LIGHT_ORANGE, ORANGE)

    s5 = deck.new_slide("Model-3: Fusion, RAG and User Interface", "Presenter: Golam Mohammad Rahi")
    deck.add_image(s5, "fig06_model3_rag_fusion_pipeline.png", 0.65, 1.02, 7.25, 3.55)
    s5.add_text(8.15, 1.05, 4.5, 3.55, [
        "Fusion strategy",
        "- Combines Model-1 image output and Model-2 document output.",
        "- Builds a fused query from structured evidence.",
        "- Retrieves local knowledge-base chunks using TF-IDF.",
        "- Generates a patient summary and non-validated follow-up note."
    ], 13, INK, False, "l", WHITE, BLUE)
    s5.add_card(0.9, 5.0, 3.4, 1.2, "Model-3 validation", ["100/100 final outputs", "Technical output success", "Not clinical correctness"], LIGHT_TEAL, TEAL)
    s5.add_card(4.95, 5.0, 3.4, 1.2, "Streamlit UI", ["Upload image/document", "View prediction + entities", "Download JSON output"], LIGHT_BLUE, BLUE)
    s5.add_card(9.0, 5.0, 3.4, 1.2, "RAG evidence", ["Retrieved evidence count: 5", "Local KB, no graph claim", "Evidence should be manually checked"], LIGHT_ORANGE, ORANGE)

    s6 = deck.new_slide("Validation and Result Summary", "Presenter: Hasin + Sabbir")
    deck.add_image(s6, "fig11_technical_validation_case_distribution.png", 0.55, 1.0, 5.95, 3.1)
    deck.add_image(s6, "fig14_model_component_output_success.png", 6.75, 1.0, 5.95, 3.1)
    s6.add_card(0.85, 4.55, 2.9, 1.25, "Final validation", ["100 requested", "100 completed", "0 failed"], LIGHT_TEAL, TEAL)
    s6.add_card(4.1, 4.55, 2.9, 1.25, "Model-1 metrics", ["MRI macro F1 0.9359", "X-ray macro AUROC 0.8133", "Tuned macro F1 0.2862"], LIGHT_BLUE, BLUE)
    s6.add_card(7.35, 4.55, 2.9, 1.25, "Model-2/3 outputs", ["Model-2: 60/60", "Model-3: 100/100", "Output success only"], LIGHT_ORANGE, ORANGE)
    s6.add_card(10.6, 4.55, 2.05, 1.25, "Interpretation", ["Technical pipeline validation", "Not clinical validation"], LIGHT_RED, RED, 14, 12)

    s7 = deck.new_slide("V4 Experiments, Limitations and Final Claim", "Presenter: Rahi + Team Q&A")
    deck.add_image(s7, "fig19_v4_experimental_metrics.png", 0.55, 1.0, 6.35, 3.25)
    s7.add_text(7.15, 1.0, 5.35, 3.25, [
        "Honest V4 claims",
        "- YOLO ROI: mAP50 0.8357 on pseudo-labeled regions.",
        "- BERT NER: entity F1 0.9963 on weak labels.",
        "- V4 is an advanced improvement layer, not Model-4.",
        "",
        "Main limitations",
        "- No clinical validation.",
        "- No expert-labeled YOLO/NER ground truth.",
        "- No trained cross-modal attention without paired data."
    ], 13, INK, False, "l", WHITE, RED)
    s7.add_text(0.8, 4.75, 11.75, 1.05, [
        "Final defense claim",
        "This thesis presents a working multimodal AI assistant that integrates image analysis, OCR-based document understanding, local RAG fusion, summarization, and a Streamlit interface for technical health-record summarization and feedback support."
    ], 14, INK, True, "ctr", LIGHT_TEAL, TEAL)
    s7.add_card(1.0, 6.08, 3.4, 0.65, "Future work", ["Manual annotation + expert evaluation"], LIGHT_BLUE, BLUE, 13, 11)
    s7.add_card(4.95, 6.08, 3.4, 0.65, "Clinical safety", ["Doctor-in-the-loop validation"], LIGHT_ORANGE, ORANGE, 13, 11)
    s7.add_card(8.9, 6.08, 3.4, 0.65, "Advanced fusion", ["Paired image-text-label dataset"], LIGHT_RED, RED, 13, 11)

    return deck


def slide_rels(slide: Slide) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
    ]
    for image in slide.images:
        rels.append(
            f'<Relationship Id="{image.rel_id}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{image.media_name}"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>"""


def content_types(slide_count: int) -> str:
    slide_overrides = "\n".join(
        f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, slide_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
  {slide_overrides}
</Types>"""


def presentation_xml(slide_count: int) -> str:
    ids = "\n".join(
        f'<p:sldId id="{255 + i}" r:id="rId{i + 1}"/>' for i in range(1, slide_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
                xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
                xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst>{ids}</p:sldIdLst>
  <p:sldSz cx="{SLIDE_W}" cy="{SLIDE_H}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
  <p:defaultTextStyle/>
</p:presentation>"""


def presentation_rels(slide_count: int) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>'
    ]
    for i in range(1, slide_count + 1):
        rels.append(
            f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(rels)}
</Relationships>"""


ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def app_xml(slide_count: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Codex</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>{slide_count}</Slides>
  <Notes>0</Notes>
  <HiddenSlides>0</HiddenSlides>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant><vt:lpstr>Slides</vt:lpstr></vt:variant>
      <vt:variant><vt:i4>{slide_count}</vt:i4></vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="{slide_count}" baseType="lpstr">
      {''.join(f'<vt:lpstr>Slide {i}</vt:lpstr>' for i in range(1, slide_count + 1))}
    </vt:vector>
  </TitlesOfParts>
</Properties>"""


def core_xml() -> str:
    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
                   xmlns:dc="http://purl.org/dc/elements/1.1/"
                   xmlns:dcterms="http://purl.org/dc/terms/"
                   xmlns:dcmitype="http://purl.org/dc/dcmitype/"
                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Multimodal AI Thesis Defense Slides</dc:title>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>"""


SLIDE_LAYOUT = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
             type="blank" preserve="1">
  <p:cSld name="Blank">
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""

SLIDE_LAYOUT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""

SLIDE_MASTER = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
             xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
             xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{BG}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    </p:spTree>
  </p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
  <p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>
</p:sldMaster>"""

SLIDE_MASTER_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""

THEME = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="ThesisTheme">
  <a:themeElements>
    <a:clrScheme name="Thesis">
      <a:dk1><a:srgbClr val="24323F"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="5E6A75"/></a:dk2>
      <a:lt2><a:srgbClr val="F7FAFC"/></a:lt2>
      <a:accent1><a:srgbClr val="3A6EA8"/></a:accent1>
      <a:accent2><a:srgbClr val="2A9D8F"/></a:accent2>
      <a:accent3><a:srgbClr val="DC9228"/></a:accent3>
      <a:accent4><a:srgbClr val="C85C5C"/></a:accent4>
      <a:accent5><a:srgbClr val="4C8F45"/></a:accent5>
      <a:accent6><a:srgbClr val="7A8691"/></a:accent6>
      <a:hlink><a:srgbClr val="3A6EA8"/></a:hlink>
      <a:folHlink><a:srgbClr val="2A9D8F"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Clean"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
  <a:objectDefaults/>
  <a:extraClrSchemeLst/>
</a:theme>"""


def write_pptx(deck: DeckBuilder) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if MEDIA_DIR.exists():
        shutil.rmtree(MEDIA_DIR)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)

    media_sources: dict[str, Path] = {}
    for slide in deck.slides:
        for image in slide.images:
            media_sources[image.media_name] = image.path

    with zipfile.ZipFile(PPTX_PATH, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", content_types(len(deck.slides)))
        package.writestr("_rels/.rels", ROOT_RELS)
        package.writestr("docProps/app.xml", app_xml(len(deck.slides)))
        package.writestr("docProps/core.xml", core_xml())
        package.writestr("ppt/presentation.xml", presentation_xml(len(deck.slides)))
        package.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(deck.slides)))
        package.writestr("ppt/slideMasters/slideMaster1.xml", SLIDE_MASTER)
        package.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", SLIDE_MASTER_RELS)
        package.writestr("ppt/slideLayouts/slideLayout1.xml", SLIDE_LAYOUT)
        package.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", SLIDE_LAYOUT_RELS)
        package.writestr("ppt/theme/theme1.xml", THEME)
        for index, slide in enumerate(deck.slides, start=1):
            package.writestr(f"ppt/slides/slide{index}.xml", slide.xml())
            package.writestr(f"ppt/slides/_rels/slide{index}.xml.rels", slide_rels(slide))
        for media_name, source in media_sources.items():
            package.write(source, f"ppt/media/{media_name}")


def write_guide() -> None:
    GUIDE_PATH.write_text(
        """# Thesis Defense Slide Speaker Guide

Deck: `Multimodal_AI_Thesis_Defense_7slides.pptx`

## Presenter Split

| Slide | Presenter | Focus |
| --- | --- | --- |
| 1 | Syed Enamul Karim | Title, team, project scope |
| 2 | Syed Enamul Karim | Problem, objective, full architecture |
| 3 | Md. Hasin Saleh Alvi | Model-1 image pipeline and image-model metrics |
| 4 | Sabbir Bhuyan | Model-2 OCR/document pipeline and V4 document comparison |
| 5 | Golam Mohammad Rahi | Model-3 fusion, RAG, UI workflow |
| 6 | Hasin + Sabbir | Final validation and result interpretation |
| 7 | Rahi + full team | V4 caveats, limitations, final claim, Q&A |

## Defense-Safe Closing Claim

This thesis presents a working multimodal AI assistant that integrates image analysis, OCR-based document understanding, local RAG fusion, summarization, and a Streamlit interface for technical health-record summarization and feedback support.

## Claims To Avoid

- Do not say Model-2 or Model-3 have ROC/AUC/F1.
- Do not say V4 is Model-4.
- Do not say YOLO/BERT V4 metrics are clinical validation.
- Do not say cross-modal attention was trained, because no true paired image-text-label dataset exists.
""",
        encoding="utf-8",
    )


def main() -> None:
    deck = build_deck()
    write_pptx(deck)
    write_guide()
    print(PPTX_PATH)
    print(GUIDE_PATH)


if __name__ == "__main__":
    main()
