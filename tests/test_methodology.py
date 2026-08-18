import pytest

from src.iu_paired.methodology import late_fuse, validate_disjoint_ids, validate_locked_predictor_columns


def test_selected_late_fusion_equation():
    assert late_fuse([0.8, 0.2], [0.4, 0.6]) == pytest.approx([0.7, 0.3])


def test_late_fusion_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        late_fuse([0.5], [0.5, 0.2])
    with pytest.raises(ValueError):
        late_fuse([1.2], [0.5])


def test_locked_predictors_exclude_hidden_reference_text():
    validate_locked_predictor_columns(["uid", "image_filenames", "indication"])
    with pytest.raises(ValueError, match="findings"):
        validate_locked_predictor_columns(["uid", "image_filenames", "indication", "findings"])
    with pytest.raises(ValueError, match="impression"):
        validate_locked_predictor_columns(["uid", "image_filenames", "indication", "impression"])


def test_development_and_locked_studies_are_disjoint():
    validate_disjoint_ids(["dev-1", "dev-2"], ["test-1", "test-2"])
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_ids(["study-1"], ["study-1"])
