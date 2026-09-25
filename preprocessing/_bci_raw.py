"""BCI-IV-2a GDF/MAT preprocessing with session-wise EA."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SOURCE_FS = 250.0
TARGET_FS = 200.0
TRIAL_WINDOW_S = (2.0, 6.0)
BANDPASS_HZ = (0.5, 99.5)
NUM_CHANNELS = 22
NUM_SAMPLES = 800
TRIALS_PER_SESSION = 288
SESSIONS = ("T", "E")
SESSION_TO_ID = {"T": 0, "E": 1}
ID_TO_SESSION = {value: key for key, value in SESSION_TO_ID.items()}
REVE_SPLIT_SUBJECTS: Mapping[str, tuple[int, ...]] = {
    "train": (1, 2, 3, 4, 5),
    "val": (6, 7),
    "test": (8, 9),
}
EXPECTED_SPLIT_COUNTS = {
    split: len(subjects) * len(SESSIONS) * TRIALS_PER_SESSION
    for split, subjects in REVE_SPLIT_SUBJECTS.items()
}
EXPECTED_TOTAL_TRIALS = sum(EXPECTED_SPLIT_COUNTS.values())
TRAIN_CUE_CODES = ("769", "770", "771", "772")
EVALUATION_CUE_CODE = "783"
TRIAL_START_CODE = "768"
ARTIFACT_CODE = "1023"
RUN_START_CODE = "32766"
CACHE_FORMAT = "bciciv2a_reve_paper_v1"
CHANNEL_NAMES = (
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
)


@dataclass(frozen=True)
class REVESession:
    """One fully preprocessed subject/session block."""

    subject: int
    session: str
    samples: np.ndarray
    labels: np.ndarray
    artifact_flags: np.ndarray
    run_ids: np.ndarray
    alignment_operator: np.ndarray

    @property
    def split(self) -> str:
        return split_for_subject(self.subject)


def normalize_subject(subject: int | str) -> int:
    """Normalize ``1``, ``"01"`` or ``"A01"`` to an integer subject id."""

    text = str(subject).upper().removeprefix("A")
    if not text.isdigit() or int(text) not in range(1, 10):
        raise ValueError(f"BCIC-IV-2a subject must be A01..A09, got {subject!r}")
    return int(text)


def split_for_subject(subject: int | str) -> str:
    """Return the immutable REVE split for one subject."""

    number = normalize_subject(subject)
    for split, subjects in REVE_SPLIT_SUBJECTS.items():
        if number in subjects:
            return split
    raise AssertionError(f"No REVE split for subject {number}")


def _load_labels(path: Path) -> np.ndarray:
    try:
        from scipy.io import loadmat
    except ImportError as error:  # pragma: no cover - focused dependency error
        raise ImportError("Reading BCIC-IV-2a MAT labels requires scipy.") from error

    payload = loadmat(path)
    if "classlabel" not in payload:
        raise AssertionError(f"Missing classlabel in {path}")
    labels = np.asarray(payload["classlabel"]).reshape(-1)
    if labels.shape != (TRIALS_PER_SESSION,):
        raise AssertionError(f"Expected 288 labels in {path}, got {labels.shape}")
    labels = labels.astype(np.int64)
    expected = Counter({1: 72, 2: 72, 3: 72, 4: 72})
    if Counter(labels.tolist()) != expected:
        raise AssertionError(f"Unexpected label counts in {path}: {Counter(labels.tolist())}")
    return labels - 1


def _selected_events(
    descriptions: np.ndarray,
    onsets: np.ndarray,
    accepted: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isin(descriptions, tuple(accepted))
    selected_onsets = onsets[mask]
    selected_descriptions = descriptions[mask]
    order = np.argsort(selected_onsets, kind="stable")
    return selected_onsets[order], selected_descriptions[order]


def _artifact_flags(start_onsets: np.ndarray, artifact_onsets: np.ndarray) -> np.ndarray:
    flags = np.zeros(TRIALS_PER_SESSION, dtype=np.bool_)
    tolerance_s = 0.5 / SOURCE_FS
    for onset in artifact_onsets:
        matches = np.flatnonzero(np.isclose(start_onsets, onset, atol=tolerance_s))
        if matches.shape != (1,):
            raise AssertionError("Each artifact marker must match exactly one trial start")
        flags[int(matches[0])] = True
    return flags


def _run_ids(run_onsets: np.ndarray, start_onsets: np.ndarray) -> np.ndarray:
    raw_ids = np.searchsorted(run_onsets, start_onsets, side="right") - 1
    if np.any(raw_ids < 0):
        raise AssertionError("A trial starts before every run marker")
    used = sorted(np.unique(raw_ids).tolist())
    counts = Counter(raw_ids.tolist())
    if len(used) != 6 or any(counts[index] != 48 for index in used):
        raise AssertionError(f"Expected six MI runs of 48 trials, got {counts}")
    lookup = {raw_id: index + 1 for index, raw_id in enumerate(used)}
    return np.asarray([lookup[int(value)] for value in raw_ids], dtype=np.uint8)


def euclidean_align(
    samples: np.ndarray,
    *,
    eigenvalue_floor: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply standard Euclidean Alignment to a session of EEG trials."""

    values = np.asarray(samples)
    if values.ndim != 3 or values.shape[1] != NUM_CHANNELS:
        raise ValueError(f"Expected [trials, 22, time], got {values.shape}")
    if values.shape[0] < 1 or values.shape[2] < NUM_CHANNELS:
        raise ValueError("Euclidean Alignment requires non-empty, full-rank-capable trials")
    if not np.isfinite(values).all():
        raise ValueError("Euclidean Alignment received non-finite samples")
    if not 0.0 < float(eigenvalue_floor) < 1.0:
        raise ValueError("eigenvalue_floor must lie in (0, 1)")

    work = np.asarray(values, dtype=np.float64)
    reference = np.einsum("nct,ndt->cd", work, work, optimize=True) / work.shape[0]
    reference = 0.5 * (reference + reference.T)
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    largest = float(eigenvalues[-1])
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("Euclidean Alignment reference is not positive definite")
    floor = largest * float(eigenvalue_floor)
    inverse_sqrt = (eigenvectors * np.reciprocal(np.sqrt(np.maximum(eigenvalues, floor)))) @ eigenvectors.T
    aligned = np.einsum("cd,ndt->nct", inverse_sqrt, work, optimize=True)
    aligned = np.asarray(aligned, dtype=np.float32)
    operator = np.asarray(inverse_sqrt, dtype=np.float32)
    if not np.isfinite(aligned).all() or not np.isfinite(operator).all():
        raise AssertionError("Euclidean Alignment produced non-finite values")
    return aligned, operator


def _validate_session(session: REVESession) -> None:
    if session.subject not in range(1, 10) or session.session not in SESSIONS:
        raise AssertionError("Invalid subject/session metadata")
    if session.samples.shape != (TRIALS_PER_SESSION, NUM_CHANNELS, NUM_SAMPLES):
        raise AssertionError(f"Invalid sample shape: {session.samples.shape}")
    if session.samples.dtype != np.float32 or not np.isfinite(session.samples).all():
        raise AssertionError("Session samples must be finite float32")
    if session.labels.shape != (TRIALS_PER_SESSION,) or session.labels.dtype != np.int64:
        raise AssertionError("Session labels must be int64[288]")
    if Counter(session.labels.tolist()) != Counter({0: 72, 1: 72, 2: 72, 3: 72}):
        raise AssertionError("Each session must contain 72 trials per class")
    if session.artifact_flags.shape != (TRIALS_PER_SESSION,):
        raise AssertionError("Artifact metadata must have length 288")
    if session.run_ids.shape != (TRIALS_PER_SESSION,) or set(session.run_ids.tolist()) != set(range(1, 7)):
        raise AssertionError("Run metadata must describe six runs")
    if session.alignment_operator.shape != (NUM_CHANNELS, NUM_CHANNELS):
        raise AssertionError("Invalid EA operator shape")


def load_reve_session(
    gdf_path: str | Path,
    label_path: str | Path,
    *,
    filter_order: int = 5,
    eigenvalue_floor: float = 1e-12,
) -> REVESession:
    """Load and preprocess one official GDF/MAT session for the REVE protocol."""

    gdf, mat = Path(gdf_path), Path(label_path)
    if not gdf.is_file():
        raise FileNotFoundError(f"GDF file not found: {gdf}")
    if not mat.is_file():
        raise FileNotFoundError(f"MAT file not found: {mat}")
    stem = gdf.stem.upper()
    if len(stem) != 4 or stem[0] != "A" or not stem[1:3].isdigit() or stem[3] not in SESSIONS:
        raise ValueError(f"Expected a name such as A01T.gdf, got {gdf.name}")
    if mat.stem.upper() != stem:
        raise ValueError(f"GDF/MAT stem mismatch: {gdf.name} versus {mat.name}")
    subject, session_name = normalize_subject(stem[1:3]), stem[3]
    labels = _load_labels(mat)

    try:
        import mne
        from scipy.signal import butter, resample, sosfiltfilt
    except ImportError as error:  # pragma: no cover - focused dependency error
        raise ImportError("REVE preprocessing requires mne and scipy.") from error

    raw = mne.io.read_raw_gdf(str(gdf), preload=False, verbose="ERROR")
    try:
        source_fs = float(raw.info["sfreq"])
        if not np.isclose(source_fs, SOURCE_FS):
            raise AssertionError(f"Expected 250 Hz in {gdf}, got {source_fs}")
        if len(raw.ch_names) != 25 or tuple(raw.ch_names[-3:]) != (
            "EOG-left",
            "EOG-central",
            "EOG-right",
        ):
            raise AssertionError(f"Expected the official 22 EEG + 3 EOG layout in {gdf}")

        descriptions = np.asarray(raw.annotations.description, dtype=str)
        onsets = np.asarray(raw.annotations.onset, dtype=np.float64)
        start_onsets, _ = _selected_events(descriptions, onsets, (TRIAL_START_CODE,))
        cue_codes = TRAIN_CUE_CODES if session_name == "T" else (EVALUATION_CUE_CODE,)
        cue_onsets, cue_descriptions = _selected_events(descriptions, onsets, cue_codes)
        run_onsets, _ = _selected_events(descriptions, onsets, (RUN_START_CODE,))
        artifact_onsets, _ = _selected_events(descriptions, onsets, (ARTIFACT_CODE,))
        if start_onsets.shape != (TRIALS_PER_SESSION,) or cue_onsets.shape != (TRIALS_PER_SESSION,):
            raise AssertionError(
                f"Expected 288 starts/cues in {gdf}, got {len(start_onsets)}/{len(cue_onsets)}"
            )
        if not np.allclose(cue_onsets - start_onsets, TRIAL_WINDOW_S[0], atol=0.5 / SOURCE_FS, rtol=0.0):
            raise AssertionError(f"Cue/start alignment is not exactly 2 s in {gdf}")
        if session_name == "T":
            annotation_labels = np.asarray([int(code) - 769 for code in cue_descriptions], dtype=np.int64)
            if not np.array_equal(annotation_labels, labels):
                mismatch = np.flatnonzero(annotation_labels != labels)
                raise AssertionError(f"T cue/MAT mismatch in {gdf}: {mismatch[:10].tolist()}")
        elif not np.all(cue_descriptions == EVALUATION_CUE_CODE):
            raise AssertionError(f"Evaluation cues must all be 783 in {gdf}")

        # Filtering is applied to the continuous EEG before epoching, matching
        # the conventional downstream pipeline and avoiding per-epoch edge
        # transients.  Data are explicitly represented in microvolts.
        continuous = raw.get_data(picks=np.arange(NUM_CHANNELS), units="uV")
        sos = butter(
            int(filter_order),
            BANDPASS_HZ,
            btype="bandpass",
            fs=SOURCE_FS,
            output="sos",
        )
        continuous = sosfiltfilt(sos, continuous, axis=-1)
        cue_samples = np.rint(cue_onsets * SOURCE_FS).astype(np.int64)
        source_length = int(round((TRIAL_WINDOW_S[1] - TRIAL_WINDOW_S[0]) * SOURCE_FS))
        epochs = np.empty((TRIALS_PER_SESSION, NUM_CHANNELS, source_length), dtype=np.float64)
        for trial_index, cue_sample in enumerate(cue_samples):
            stop = int(cue_sample) + source_length
            if stop > continuous.shape[-1]:
                raise AssertionError(f"Trial {trial_index} exceeds {gdf}")
            epochs[trial_index] = continuous[:, int(cue_sample) : stop]
        del continuous
        samples = resample(epochs, NUM_SAMPLES, axis=-1)
        del epochs
        samples, operator = euclidean_align(samples, eigenvalue_floor=eigenvalue_floor)
        result = REVESession(
            subject=subject,
            session=session_name,
            samples=samples,
            labels=labels.astype(np.int64, copy=False),
            artifact_flags=_artifact_flags(start_onsets, artifact_onsets),
            run_ids=_run_ids(run_onsets, start_onsets),
            alignment_operator=operator,
        )
    finally:
        close = getattr(raw, "close", None)
        if callable(close):
            close()
    _validate_session(result)
    return result


def _cache_file(root: Path, split: str, field: str) -> Path:
    return root / f"{split}_{field}.npy"


def build_reve_cache(
    gdf_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    *,
    subjects: Iterable[int | str] = range(1, 10),
    filter_order: int = 5,
    eigenvalue_floor: float = 1e-12,
) -> Path:
    """Build a memory-mapped cache from official files."""

    gdf_root, label_root, output = Path(gdf_dir), Path(label_dir), Path(output_dir)
    selected = tuple(sorted({normalize_subject(subject) for subject in subjects}))
    if not selected:
        raise ValueError("At least one subject is required")
    if output.exists():
        raise FileExistsError(f"Cache directory already exists: {output}")
    output.mkdir(parents=True)

    selected_by_split = {
        split: tuple(subject for subject in split_subjects if subject in selected)
        for split, split_subjects in REVE_SPLIT_SUBJECTS.items()
    }
    counts = {
        split: len(split_subjects) * len(SESSIONS) * TRIALS_PER_SESSION
        for split, split_subjects in selected_by_split.items()
    }
    arrays: dict[str, dict[str, np.memmap]] = {}
    for split, count in counts.items():
        arrays[split] = {
            "samples": np.lib.format.open_memmap(
                _cache_file(output, split, "samples"),
                mode="w+",
                dtype=np.float32,
                shape=(count, NUM_CHANNELS, NUM_SAMPLES),
            ),
            "labels": np.lib.format.open_memmap(
                _cache_file(output, split, "labels"), mode="w+", dtype=np.int64, shape=(count,)
            ),
            "subjects": np.lib.format.open_memmap(
                _cache_file(output, split, "subjects"), mode="w+", dtype=np.uint8, shape=(count,)
            ),
            "sessions": np.lib.format.open_memmap(
                _cache_file(output, split, "sessions"), mode="w+", dtype=np.uint8, shape=(count,)
            ),
            "trials": np.lib.format.open_memmap(
                _cache_file(output, split, "trials"), mode="w+", dtype=np.uint16, shape=(count,)
            ),
            "runs": np.lib.format.open_memmap(
                _cache_file(output, split, "runs"), mode="w+", dtype=np.uint8, shape=(count,)
            ),
            "artifacts": np.lib.format.open_memmap(
                _cache_file(output, split, "artifacts"), mode="w+", dtype=np.bool_, shape=(count,)
            ),
        }

    offsets = {split: 0 for split in REVE_SPLIT_SUBJECTS}
    operators: list[np.ndarray] = []
    operator_subjects: list[int] = []
    operator_sessions: list[int] = []
    try:
        for subject in selected:
            split = split_for_subject(subject)
            for session_name in SESSIONS:
                stem = f"A{subject:02d}{session_name}"
                block = load_reve_session(
                    gdf_root / f"{stem}.gdf",
                    label_root / f"{stem}.mat",
                    filter_order=filter_order,
                    eigenvalue_floor=eigenvalue_floor,
                )
                start = offsets[split]
                stop = start + TRIALS_PER_SESSION
                target = arrays[split]
                target["samples"][start:stop] = block.samples
                target["labels"][start:stop] = block.labels
                target["subjects"][start:stop] = subject
                target["sessions"][start:stop] = SESSION_TO_ID[session_name]
                target["trials"][start:stop] = np.arange(TRIALS_PER_SESSION, dtype=np.uint16)
                target["runs"][start:stop] = block.run_ids
                target["artifacts"][start:stop] = block.artifact_flags
                offsets[split] = stop
                operators.append(block.alignment_operator)
                operator_subjects.append(subject)
                operator_sessions.append(SESSION_TO_ID[session_name])
        for split_arrays in arrays.values():
            for array in split_arrays.values():
                array.flush()
        np.save(output / "alignment_operators.npy", np.stack(operators).astype(np.float32))
        np.save(output / "alignment_subjects.npy", np.asarray(operator_subjects, dtype=np.uint8))
        np.save(output / "alignment_sessions.npy", np.asarray(operator_sessions, dtype=np.uint8))
        manifest = {
            "format": CACHE_FORMAT,
            "protocol": "REVE BCIC-IV-2a subject-independent split",
            "subjects": list(selected),
            "split_subjects": {key: list(value) for key, value in selected_by_split.items()},
            "counts": counts,
            "complete_protocol": selected == tuple(range(1, 10)),
            "total_trials": sum(counts.values()),
            "shape_per_trial": [NUM_CHANNELS, NUM_SAMPLES],
            "dtype": "float32",
            "source_fs": SOURCE_FS,
            "target_fs": TARGET_FS,
            "trial_window_s_from_trial_start": list(TRIAL_WINDOW_S),
            "bandpass_hz": list(BANDPASS_HZ),
            "filter": {"family": "butterworth", "order": int(filter_order), "phase": "zero"},
            "resampling": "scipy.signal.resample FFT 250_to_200",
            "alignment": "Euclidean Alignment independently per subject/session",
            "alignment_eigenvalue_floor_relative": float(eigenvalue_floor),
            "units_before_alignment": "microvolts",
            "channels": list(CHANNEL_NAMES),
            "labels": {"0": "left_hand", "1": "right_hand", "2": "feet", "3": "tongue"},
            "artifact_policy": "retained_and_flagged",
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception:
        # Leave the incomplete directory intact for diagnosis; the absence of
        # manifest.json prevents it from being mistaken for a valid cache.
        raise
    validate_reve_cache(output, require_full=selected == tuple(range(1, 10)))
    return output


def _read_manifest(root: Path) -> dict:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"REVE cache manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != CACHE_FORMAT:
        raise ValueError(f"Not a {CACHE_FORMAT} cache: {root}")
    return manifest


def validate_reve_cache(cache_dir: str | Path, *, require_full: bool = True) -> dict:
    """Validate cache structure, split isolation, shapes and class balance."""

    root = Path(cache_dir)
    manifest = _read_manifest(root)
    if require_full and not manifest.get("complete_protocol", False):
        raise AssertionError("A full REVE cache with all nine subjects is required")
    observed_total = 0
    for split, allowed_subjects in REVE_SPLIT_SUBJECTS.items():
        count = int(manifest["counts"][split])
        samples = np.load(_cache_file(root, split, "samples"), mmap_mode="r")
        labels = np.load(_cache_file(root, split, "labels"), mmap_mode="r")
        subjects = np.load(_cache_file(root, split, "subjects"), mmap_mode="r")
        sessions = np.load(_cache_file(root, split, "sessions"), mmap_mode="r")
        trials = np.load(_cache_file(root, split, "trials"), mmap_mode="r")
        runs = np.load(_cache_file(root, split, "runs"), mmap_mode="r")
        artifacts = np.load(_cache_file(root, split, "artifacts"), mmap_mode="r")
        if samples.shape != (count, NUM_CHANNELS, NUM_SAMPLES) or samples.dtype != np.float32:
            raise AssertionError(f"Invalid {split} sample array: {samples.shape}/{samples.dtype}")
        for name, array in {
            "labels": labels,
            "subjects": subjects,
            "sessions": sessions,
            "trials": trials,
            "runs": runs,
            "artifacts": artifacts,
        }.items():
            if array.shape != (count,):
                raise AssertionError(f"Invalid {split} {name} shape: {array.shape}")
        if not set(subjects.tolist()).issubset(set(allowed_subjects)):
            raise AssertionError(f"Subject leakage in {split}")
        if not set(sessions.tolist()).issubset(set(SESSION_TO_ID.values())):
            raise AssertionError(f"Invalid session ids in {split}")
        for subject in sorted(set(subjects.tolist())):
            for session_id in SESSION_TO_ID.values():
                mask = (subjects == subject) & (sessions == session_id)
                if int(mask.sum()) != TRIALS_PER_SESSION:
                    raise AssertionError(f"A{subject:02d}/{ID_TO_SESSION[session_id]} has {int(mask.sum())} trials")
                if Counter(labels[mask].tolist()) != Counter({0: 72, 1: 72, 2: 72, 3: 72}):
                    raise AssertionError(f"Unbalanced labels for A{subject:02d}/{ID_TO_SESSION[session_id]}")
                if set(trials[mask].tolist()) != set(range(TRIALS_PER_SESSION)):
                    raise AssertionError("Trial ids must be 0..287 per session")
                if Counter(runs[mask].tolist()) != Counter({run: 48 for run in range(1, 7)}):
                    raise AssertionError("Run ids must contain six groups of 48")
        if count:
            step = 64
            for start in range(0, count, step):
                if not np.isfinite(samples[start : start + step]).all():
                    raise AssertionError(f"Non-finite values in {split} samples")
        observed_total += count
    if observed_total != int(manifest["total_trials"]):
        raise AssertionError("Manifest total does not equal split totals")
    if require_full:
        if observed_total != EXPECTED_TOTAL_TRIALS:
            raise AssertionError(f"Expected {EXPECTED_TOTAL_TRIALS} trials, got {observed_total}")
        if {key: int(manifest["counts"][key]) for key in EXPECTED_SPLIT_COUNTS} != EXPECTED_SPLIT_COUNTS:
            raise AssertionError("Full cache split counts differ from REVE")
    return manifest
