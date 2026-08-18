from src.schemas import Entity, Model1Output, Model2Output


def test_model1_schema_serializes():
    result = Model1Output(
        case_id="synthetic-1",
        modality="brain_mri",
        top_predictions=["no tumor"],
        probabilities={"no tumor": 0.8},
        embedding_path="outputs/synthetic.npy",
        patient_summary_text="Synthetic test only.",
    )
    assert result.model_dump()["case_id"] == "synthetic-1"


def test_model2_schema_serializes_entities():
    result = Model2Output(
        case_id="synthetic-2",
        source_file="synthetic.txt",
        raw_text="synthetic",
        raw_text_preview="synthetic",
        entities=[Entity(text="example", label="TEST", confidence=0.5)],
        patient_summary="Synthetic test only.",
    )
    assert result.model_dump()["entities"][0]["label"] == "TEST"
