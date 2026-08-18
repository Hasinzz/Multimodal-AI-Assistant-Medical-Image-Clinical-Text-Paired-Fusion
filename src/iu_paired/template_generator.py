import re
from collections import Counter

from .config import TARGET_LABELS

ALIASES = {
    "Cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged cardiac"],
    "Pulmonary Atelectasis": ["atelectasis", "atelectatic"],
    "Calcified Granuloma": ["calcified granuloma", "granuloma"],
    "Cicatrix": ["cicatrix", "scar", "scarring"],
    "Pleural Effusion": ["pleural effusion", "effusion"],
    "Atherosclerosis": ["atherosclerosis", "atherosclerotic"],
    "Airspace Disease": ["airspace disease", "airspace opacity", "airspace opacities"],
    "Scoliosis": ["scoliosis", "scoliotic"],
    "Granulomatous Disease": ["granulomatous disease", "granulomatous"],
    "Nodule": ["nodule", "nodular"],
}
NEGATIONS = ("no ", "not ", "without ", "negative for ", "no evidence of ")
UNSUPPORTED = re.compile(r"\b(left|right|bilateral|mild|moderate|severe)\b|\b\d+(?:\.\d+)?\s*(?:cm|mm)\b", re.I)

def clean_sentence(text: str) -> str:
    text = re.sub(r"X{2,}", "", str(text), flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" ,;:-")
    return text if not text or text.endswith(('.', '!', '?')) else text + "."

def sentences(text: str) -> list[str]:
    return [clean_sentence(s) for s in re.split(r"(?<=[.!?])\s+|[\r\n]+", str(text)) if clean_sentence(s)]

def concepts(text: str) -> set[str]:
    low = text.lower()
    return {label for label, aliases in ALIASES.items() if any(a in low for a in aliases)}

def negated(text: str, aliases: list[str]) -> bool:
    low = text.lower()
    for alias in aliases:
        pos = low.find(alias)
        if pos >= 0 and any(n in low[max(0, pos - 35):pos] for n in NEGATIONS):
            return True
    return False

def derive_library(train_records: list[dict]) -> dict:
    library = {"labels": {}, "normal": {}}
    for label in TARGET_LABELS:
        candidates = []
        for row in train_records:
            if label not in row["labels"]:
                continue
            for sent in sentences(row["findings"]):
                if not any(a in sent.lower() for a in ALIASES[label]) or negated(sent, ALIASES[label]):
                    continue
                unsafe = bool(UNSUPPORTED.search(sent))
                multi = len(concepts(sent)) > 1
                score = (unsafe, multi, abs(len(sent.split()) - 9), len(sent), sent.lower(), str(row["uid"]))
                candidates.append((score, row, sent))
        safe = [x for x in candidates if not x[0][0]]
        if safe:
            score, row, sent = sorted(safe, key=lambda x: x[0])[0]
            library["labels"][label] = {"template": sent, "source_uid": str(row["uid"]),
                "original_sentence": sent, "fallback": False,
                "ranking_reason": "safe; deterministic preference for single-concept, concise sentence"}
        else:
            library["labels"][label] = {"template": f"{label} is present.", "source_uid": None,
                "original_sentence": None, "fallback": True,
                "ranking_reason": "no safe affirmative training-derived sentence"}
    normals = []
    for row in train_records:
        if row["problems_exact_normal"]:
            cleaned = clean_sentence(row["findings"])
            if cleaned and not concepts(cleaned) and not UNSUPPORTED.search(cleaned) and "xxxx" not in cleaned.lower():
                normals.append((cleaned.lower(), str(row["uid"]), cleaned))
    if normals:
        counts = Counter(x[0] for x in normals)
        chosen = sorted(normals, key=lambda x: (-counts[x[0]], len(x[2]), x[0], x[1]))[0]
        library["normal"] = {"template": chosen[2], "source_uid": chosen[1], "source_text": chosen[2],
            "fallback": False, "ranking_reason": "most frequent safe training finding, then shortest"}
    else:
        library["normal"] = {"template": "No selected abnormality is identified.", "source_uid": None,
            "source_text": None, "fallback": True, "ranking_reason": "no safe training-derived normal finding"}
    return library
