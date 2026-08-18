from typing import Dict, Optional

from src.config import KB_DIR
from src.model3.fusion import build_fused_query, generate_final_summary
from src.model3.retriever import LocalTfidfRetriever


def run_fusion_pipeline(
    case_id: str,
    model1_output: Optional[Dict] = None,
    model2_output: Optional[Dict] = None,
    doctor_note_output: Optional[Dict] = None,
    kb_dir: str = str(KB_DIR),
    top_k: int = 5,
    use_rag: bool = True,
) -> Dict:
    fused_query = build_fused_query(
        model1_output=model1_output,
        model2_output=model2_output,
        doctor_note_output=doctor_note_output,
    )

    if use_rag:
        retriever = LocalTfidfRetriever(kb_dir=kb_dir)
        retrieved_evidence = retriever.retrieve(
            query=fused_query,
            top_k=top_k,
        )
    else:
        retrieved_evidence = []

    output = generate_final_summary(
        case_id=case_id,
        model1_output=model1_output,
        model2_output=model2_output,
        doctor_note_output=doctor_note_output,
        retrieved_evidence=retrieved_evidence,
    )

    output["fused_query"] = fused_query
    output["kb_used"] = str(kb_dir)
    output["rag_enabled"] = bool(use_rag)

    return output
