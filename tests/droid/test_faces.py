from pathlib import Path

import numpy as np

from reachy_mini.vision.face_detector import Face
from reachy_mini_conversation_app.droid.faces import (
    _TEMPLATE,
    MATCH_THRESHOLD,
    MAX_EMBEDDINGS_PER_PERSON,
    FaceRegistry,
    align_face,
    _similarity_transform,
)


def _unit(seed: int) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=128).astype(np.float32)
    return vector / np.linalg.norm(vector)


def test_similarity_transform_recovers_scale_rotation_and_shift() -> None:
    """Landmarks that are an exact similarity of the template map back onto it."""
    angle = np.deg2rad(12)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    src = (_TEMPLATE @ rotation.T) * 2.5 + np.array([300.0, 180.0])

    matrix = _similarity_transform(src, _TEMPLATE)

    np.testing.assert_allclose(src @ matrix[:, :2].T + matrix[:, 2], _TEMPLATE, atol=1e-6)


def test_align_face_crops_the_face_region() -> None:
    """The aligned crop is 112x112 and samples pixels around the landmarks, not the background."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[150:300, 250:400] = 200
    face = Face(bbox=(250, 150, 150, 150), right_eye=(290, 200), left_eye=(360, 200), nose=(325, 240))

    crop = align_face(frame, face)

    assert crop.shape == (112, 112, 3)
    assert crop[56, 56].tolist() == [200, 200, 200]


def test_registry_identifies_enrolled_people(tmp_path: Path) -> None:
    """Enrolled embeddings match; unrelated vectors do not."""
    registry = FaceRegistry(tmp_path)
    owner = _unit(1)
    registry.enroll("Pasha", [owner], is_owner=True)
    registry.enroll("Anna", [_unit(2)])

    noisy_owner = owner + 0.02 * _unit(3)
    person, score = registry.identify(noisy_owner / np.linalg.norm(noisy_owner))
    stranger, stranger_score = registry.identify(_unit(4))

    assert person is not None and person.name == "Pasha" and score > MATCH_THRESHOLD
    assert stranger is None and stranger_score < MATCH_THRESHOLD
    assert registry.owner_enrolled()
    assert [p.name for p in registry.people()] == ["Pasha", "Anna"]


def test_registry_persists_caps_and_forgets(tmp_path: Path) -> None:
    """Embeddings survive a reload, are capped per person, and can be forgotten."""
    registry = FaceRegistry(tmp_path)
    stored = registry.enroll("Anna", [_unit(i) for i in range(MAX_EMBEDDINGS_PER_PERSON + 5)])

    reloaded = FaceRegistry(tmp_path)

    assert stored == MAX_EMBEDDINGS_PER_PERSON
    assert len(reloaded.people()[0].embeddings) == MAX_EMBEDDINGS_PER_PERSON
    assert reloaded.forget("anna") is True
    assert FaceRegistry(tmp_path).people() == []
