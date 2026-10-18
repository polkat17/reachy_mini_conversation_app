"""Face recognition: SFace embeddings (Apache-2.0, OpenCV Zoo) on ONNX Runtime, matched against enrolled people.

Only 128-number embeddings are stored, never images.
"""

import json
import logging
import threading
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from scipy.ndimage import affine_transform
from huggingface_hub import hf_hub_download

from reachy_mini.vision.face_detector import Face


logger = logging.getLogger(__name__)

_MODEL_REPO = "opencv/face_recognition_sface"
_MODEL_FILE = "face_recognition_sface_2021dec.onnx"
_MODEL_REVISION = "3d7082438a6e4551e840c9b2bb60b71e8da4b524"
_CROP_SIZE = 112
# SFace's canonical positions of the right eye, left eye and nose tip in the 112x112 crop.
_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366]], dtype=np.float64)

MATCH_THRESHOLD = 0.40  # cosine similarity; the model's published threshold is 0.363
FACES_FILENAME = "faces.json"
MAX_EMBEDDINGS_PER_PERSON = 20


def _similarity_transform(src: NDArray[np.float64], dst: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the 2x3 least-squares similarity transform mapping ``src`` points onto ``dst`` (Umeyama)."""
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    covariance = dst_c.T @ src_c / len(src)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[1, 1] = -1
    rotation = u @ sign @ vt
    scale = float(np.trace(np.diag(singular) @ sign)) / float(src_c.var(axis=0).sum())
    translation = dst_mean - scale * rotation @ src_mean
    return np.hstack([scale * rotation, translation[:, None]])


def align_face(frame_bgr: NDArray[np.uint8], face: Face) -> NDArray[np.uint8]:
    """Warp the face into SFace's 112x112 canonical crop using the eye and nose landmarks."""
    landmarks = np.array([face.right_eye, face.left_eye, face.nose], dtype=np.float64)
    forward = _similarity_transform(landmarks, _TEMPLATE)
    # affine_transform maps output coordinates to input coordinates, in (row, col) order.
    inverse = np.linalg.inv(np.vstack([forward, [0.0, 0.0, 1.0]]))
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    matrix = swap @ inverse @ swap
    crop = np.empty((_CROP_SIZE, _CROP_SIZE, 3), dtype=np.uint8)
    for channel in range(3):
        crop[..., channel] = affine_transform(
            frame_bgr[..., channel].astype(np.float32),
            matrix[:2, :2],
            offset=matrix[:2, 2],
            output_shape=(_CROP_SIZE, _CROP_SIZE),
            order=1,
            mode="constant",
        ).clip(0, 255)
    return crop


class FaceEmbedder:
    """Turn a detected face into a unit-length 128-d SFace embedding."""

    def __init__(self) -> None:
        """Download (once) and load the SFace model on one CPU thread."""
        model_path = hf_hub_download(_MODEL_REPO, _MODEL_FILE, revision=_MODEL_REVISION)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.log_severity_level = 3  # the published model triggers harmless initializer warnings
        self._session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def embed(self, frame_bgr: NDArray[np.uint8], face: Face) -> NDArray[np.float32]:
        """Return the normalised embedding of ``face`` in ``frame_bgr``."""
        crop_rgb = align_face(frame_bgr, face)[..., ::-1]
        blob = crop_rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        (features,) = self._session.run(None, {self._input_name: blob})
        vector = np.asarray(features, dtype=np.float32).reshape(-1)
        return (vector / max(float(np.linalg.norm(vector)), 1e-9)).astype(np.float32)


@dataclass
class KnownPerson:
    """An enrolled person: a name, whether they are the owner, and their face embeddings."""

    name: str
    is_owner: bool = False
    embeddings: list[list[float]] = field(default_factory=list)


class FaceRegistry:
    """Enrolled faces stored as embeddings in ``faces.json``; thread-safe."""

    def __init__(self, data_dir: Path) -> None:
        """Load enrolled people from ``data_dir``."""
        self._path = data_dir / FACES_FILENAME
        self._lock = threading.Lock()
        self._people: dict[str, KnownPerson] = self._load()

    def _load(self) -> dict[str, KnownPerson]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read enrolled faces at %s: %s", self._path, e)
            return {}
        people: dict[str, KnownPerson] = {}
        for entry in raw.get("people", []) if isinstance(raw, dict) else []:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                person = KnownPerson(entry["name"], bool(entry.get("is_owner")), list(entry.get("embeddings", [])))
                people[person.name.lower()] = person
        return people

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"people": [person.__dict__ for person in self._people.values()]}
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(self._path)

    def people(self) -> list[KnownPerson]:
        """Return enrolled people, owner first."""
        with self._lock:
            return sorted(self._people.values(), key=lambda person: (not person.is_owner, person.name.lower()))

    def owner_enrolled(self) -> bool:
        """Return whether the owner has at least one embedding."""
        return any(person.is_owner and person.embeddings for person in self.people())

    def enroll(self, name: str, embeddings: list[NDArray[np.float32]], *, is_owner: bool = False) -> int:
        """Add embeddings for ``name`` (keeping the most recent ones); returns how many are stored."""
        with self._lock:
            key = name.strip().lower()
            person = self._people.get(key) or KnownPerson(name.strip(), is_owner)
            person.is_owner = person.is_owner or is_owner
            person.embeddings = (person.embeddings + [vector.tolist() for vector in embeddings])[
                -MAX_EMBEDDINGS_PER_PERSON:
            ]
            self._people[key] = person
            self._save()
            return len(person.embeddings)

    def forget(self, name: str) -> bool:
        """Delete everything stored about ``name``'s face."""
        with self._lock:
            removed = self._people.pop(name.strip().lower(), None) is not None
            if removed:
                self._save()
            return removed

    def clear(self) -> None:
        """Delete all enrolled faces."""
        with self._lock:
            self._people = {}
            self._save()

    def identify(self, embedding: NDArray[np.float32]) -> tuple[KnownPerson | None, float]:
        """Return the best-matching person and similarity, or None when nobody passes the threshold."""
        best: KnownPerson | None = None
        best_score = -1.0
        for person in self.people():
            if not person.embeddings:
                continue
            score = float(np.max(np.asarray(person.embeddings, dtype=np.float32) @ embedding))
            if score > best_score:
                best, best_score = person, score
        if best_score < MATCH_THRESHOLD:
            return None, best_score
        return best, best_score
