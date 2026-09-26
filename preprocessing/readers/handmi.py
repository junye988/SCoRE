"""Signal preparation for the released anonymous HandMI recordings."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import scipy
from scipy.signal import butter, resample_poly, sosfiltfilt

from utils.io import read_json, sha256_file


STAGES = ("baseline_onset", "cue_onset", "mi_onset", "rest_onset")
CSV_FIELDS = ["subject_id", "block", "trial_id", "label", "baseline_onset_s",
              "cue_onset_s", "mi_onset_s", "rest_onset_s", "trial_end_s", "mi_duration_s"]
SUBJECT_NAMES = [f"S{i:03d}" for i in range(1, 13)]
NUMERIC_TAL = re.compile(rb"([+-]\d+(?:\.\d+)?)(?:\x15[^\x14]*)?\x14\s*(\d+)\x14")


def read_edf(raw: bytes) -> tuple[np.ndarray, dict, list[dict]]:
    """Read calibrated EEG and numeric events without exporting identity fields."""
    if len(raw) < 256:
        raise ValueError("Truncated EDF header")
    h = raw[:256]
    header_bytes, records, channels = int(h[184:192]), int(h[236:244]), int(h[252:256])
    record_seconds = float(h[244:252])
    if (h[192:236].strip() != b"EDF+C" or records <= 0 or not 1 <= channels <= 256
            or not np.isfinite(record_seconds) or record_seconds <= 0
            or header_bytes != 256 + 256 * channels):
        raise ValueError("Expected fixed-length continuous EDF+C with a valid header")
    sh, fields, cursor = raw[256:header_bytes], {}, 0
    for key, width in (("labels", 16), ("transducer", 80), ("units", 8),
                       ("physical_min", 8), ("physical_max", 8), ("digital_min", 8),
                       ("digital_max", 8), ("prefilter", 80), ("samples", 8), ("reserved", 32)):
        fields[key] = [sh[cursor + i * width:cursor + (i + 1) * width].decode("ascii").strip()
                       for i in range(channels)]
        cursor += width * channels
    counts = np.asarray(fields["samples"], dtype=np.int64)
    if np.any(counts <= 0) or len(raw) != header_bytes + records * int(counts.sum()) * 2:
        raise ValueError("EDF data length differs from its header")
    record_data = np.frombuffer(raw, dtype="<i2", offset=header_bytes).reshape(records, int(counts.sum()))
    offsets = np.r_[0, np.cumsum(counts)]
    eeg_indices = [i for i, label in enumerate(fields["labels"]) if label.startswith("EEG ")]
    annotation_indices = [i for i, label in enumerate(fields["labels"]) if label == "EDF Annotations"]
    if len(eeg_indices) != 32 or len(annotation_indices) != 1:
        raise ValueError("Expected 32 EEG channels and one annotation channel")
    if {int(counts[i]) / record_seconds for i in eeg_indices} != {500.0}:
        raise ValueError("Expected 500 Hz EEG")
    eeg, names = [], []
    for i in eeg_indices:
        if fields["units"][i] != "uV":
            raise ValueError("Expected microvolt EEG physical units")
        pmin, pmax = float(fields["physical_min"][i]), float(fields["physical_max"][i])
        dmin, dmax = float(fields["digital_min"][i]), float(fields["digital_max"][i])
        if not np.isfinite([pmin, pmax, dmin, dmax]).all() or dmax <= dmin or pmax <= pmin:
            raise ValueError("Invalid EDF physical scaling")
        digital = record_data[:, offsets[i]:offsets[i + 1]].reshape(-1).astype(np.float64)
        eeg.append((digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin)
        names.append(fields["labels"][i].removeprefix("EEG ").removesuffix("-CPz"))
    a = annotation_indices[0]
    annotation_data = record_data[:, offsets[a]:offsets[a + 1]].copy()
    for row, expected in ((0, 0.0), (-1, (records - 1) * record_seconds)):
        first_tal = annotation_data[row].tobytes().split(b"\x14", 1)[0].split(b"\x15", 1)[0]
        if abs(float(first_tal) - expected) > record_seconds + 1e-6:
            raise ValueError("EDF timekeeping does not match continuous sample coordinates")
    events = [{"time": float(m[1]), "code": int(m[2])}
              for m in NUMERIC_TAL.finditer(annotation_data.tobytes())]
    events.sort(key=lambda e: e["time"])
    info = {"sample_rate_hz": 500, "duration_seconds": records * record_seconds,
            "eeg_channels": names, "physical_unit": "uV", "signal_count": channels,
            "edf_numeric_event_count": len(events),
            "edf_numeric_code_counts": dict(Counter(str(e["code"]) for e in events))}
    return np.stack(eeg), info, events


def group_event_bursts(events: list[dict], tolerance: float = 0.020) -> list[dict]:
    """Merge transitions within 20 ms of a burst's first edge."""
    grouped = []
    for event in events:
        if not grouped or event["time"] - grouped[-1]["time"] > tolerance:
            grouped.append({"time": event["time"], "codes": [event["code"]], "edge_times": [event["time"]]})
        else:
            grouped[-1]["codes"].append(event["code"])
            grouped[-1]["edge_times"].append(event["time"])
    return grouped


def synchronize_events(csv_times: np.ndarray, edf_times: np.ndarray) -> dict:
    """Fit EDF_seconds = slope * CSV_seconds + offset, without labels/codes."""
    x, t = np.asarray(csv_times, float), np.asarray(edf_times, float)
    if (x.ndim != 1 or t.ndim != 1 or len(x) < 20 or len(t) < 20
            or not np.isfinite(x).all() or not np.isfinite(t).all()
            or np.any(np.diff(x) <= 0) or np.any(np.diff(t) <= 0)):
        raise ValueError("Synchronization requires finite, increasing session event sequences")
    deltas = (t[:, None] - x[None, :]).ravel()
    bins = np.rint(deltas / 0.020).astype(np.int64)
    keys, votes = np.unique(bins, return_counts=True)
    best = int(np.argmax(votes))
    offset = float(np.median(deltas[bins == keys[best]]))
    alternative = votes[np.abs(keys - keys[best]) > 50]
    alternative_votes = int(alternative.max()) if len(alternative) else 0
    if votes[best] < 0.50 * min(len(x), len(t)) or votes[best] < 2 * alternative_votes:
        raise ValueError("Session clock offset is ambiguous")
    slope = 1.0
    for _ in range(5):
        nearest = np.argmin(np.abs((slope * x + offset)[:, None] - t), axis=1)
        residual = t[nearest] - (slope * x + offset)
        fit = np.abs(residual) < 0.030
        if fit.sum() < 0.70 * len(x):
            raise ValueError("Insufficient event timing correspondence")
        slope, offset = (float(v) for v in np.polyfit(x[fit], t[nearest[fit]], 1))
    nearest = np.argmin(np.abs((slope * x + offset)[:, None] - t), axis=1)
    residual = t[nearest] - (slope * x + offset)
    matched = np.abs(residual) <= 0.020
    unique = np.unique(nearest[matched])
    if (abs(slope - 1) > 1e-4 or matched.sum() < 0.70 * len(x)
            or len(unique) != matched.sum() or len(unique) < 0.98 * len(t)
            or np.ptp(x[matched]) < 0.95 * np.ptp(x)):
        raise ValueError("Clock mapping failed drift, coverage or uniqueness checks")
    return {"slope": slope, "offset_seconds": offset, "nearest": nearest,
            "matched": matched, "residual_seconds": residual,
            "drift_ppm": (slope - 1) * 1e6,
            "matched_csv_events": int(matched.sum()), "csv_events": len(x), "edf_grouped_events": len(t),
            "csv_match_fraction": float(matched.mean()), "edf_match_fraction": len(unique) / len(t),
            "offset_peak_votes": int(votes[best]), "alternative_offset_votes": alternative_votes,
            "max_residual_ms": float(np.max(np.abs(residual[matched])) * 1000),
            "rms_residual_ms": float(np.sqrt(np.mean(residual[matched] ** 2)) * 1000)}


def parse_csv_events(raw: bytes, subject: str) -> tuple[list[dict], np.ndarray]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames != CSV_FIELDS:
        raise ValueError("CSV columns must match the documented anonymous event schema")
    rows = list(reader)
    if not rows:
        raise ValueError("Empty event CSV")
    times, ids = [], []
    previous_end = -np.inf
    for row in rows:
        if row["subject_id"] != subject or int(row["label"]) not in (0, 1):
            raise ValueError("CSV subject or label is inconsistent")
        trial_id = int(row["trial_id"])
        if not 1 <= trial_id < 1000 or int(row["block"]) < 1:
            raise ValueError("Expected positive block and trial_id between 1 and 999")
        ids.append(trial_id)
        phase_times = [float(row[stage + "_s"]) for stage in STAGES] + [float(row["trial_end_s"])]
        duration = float(row["mi_duration_s"])
        if (not np.isfinite(phase_times + [duration]).all() or abs(duration - 4.0) > 1e-6
                or np.any(np.diff(phase_times) <= 0) or phase_times[0] < previous_end):
            raise ValueError("Invalid, overlapping or non-finite CSV trial timing")
        previous_end = phase_times[-1]
        times.extend(phase_times[:4])
    if len(set(ids)) != len(ids) or any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError("CSV trial_id must be unique and increasing")
    if abs(times[0]) > 1e-6:
        raise ValueError("CSV time zero must be the first trial's baseline onset")
    return rows, np.asarray(times)


def mirror_permutation(channel_names: list[str]) -> np.ndarray:
    pairs = (("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("FC5", "FC6"),
             ("FC1", "FC2"), ("M1", "M2"), ("T7", "T8"), ("C3", "C4"),
             ("CP5", "CP6"), ("CP1", "CP2"), ("P7", "P8"), ("P3", "P4"), ("O1", "O2"))
    lookup = {n.casefold(): i for i, n in enumerate(channel_names)}
    if len(lookup) != len(channel_names):
        raise ValueError("Duplicate EEG channel names")
    permutation = np.arange(len(channel_names), dtype=np.int64)
    for left, right in pairs:
        if left.casefold() not in lookup or right.casefold() not in lookup:
            raise ValueError("Missing required left/right EEG channel pair")
        i, j = lookup[left.casefold()], lookup[right.casefold()]
        permutation[i], permutation[j] = j, i
    if not np.array_equal(permutation[permutation], np.arange(len(permutation))):
        raise ValueError("Mirror permutation is not an involution")
    return permutation


def estimate_ea(trials: np.ndarray, ridge_relative: float = 1e-4,
                eigenfloor_relative: float = 1e-6) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit one unlabeled covariance reference per entire retained MI session."""
    if trials.ndim != 3 or len(trials) == 0 or not np.isfinite(trials).all():
        raise ValueError("EA requires non-empty finite trial arrays")
    if not 0 < ridge_relative <= 1 or not 0 < eigenfloor_relative <= 1:
        raise ValueError("EA stabilization parameters must lie in (0,1]")
    centered = trials.astype(np.float64) - trials.mean(axis=-1, keepdims=True)
    reference = np.einsum("nct,ndt->cd", centered, centered) / (trials.shape[0] * trials.shape[-1])
    scale = float(np.trace(reference) / reference.shape[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid session covariance")
    regularized = reference + ridge_relative * scale * np.eye(reference.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    eigenvalues = np.maximum(eigenvalues, eigenfloor_relative * scale)
    A = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    A_inv = (eigenvectors * eigenvalues ** 0.5) @ eigenvectors.T
    info = {"fit_trials": len(trials), "fit_uses_labels": False, "ridge_relative": ridge_relative,
            "ridge_absolute": ridge_relative * scale, "eigenfloor_relative": eigenfloor_relative,
            "regularized_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "A_A_inv_max_error": float(np.max(np.abs(A @ A_inv - np.eye(len(A)))))}
    return A, A_inv, info


def trial_windows(trials: np.ndarray, fs: int = 200) -> np.ndarray:
    if fs != 200 or trials.ndim != 3 or trials.shape[-1] != 4 * fs:
        raise ValueError("Windowing requires complete four-second trials at 200 Hz")
    return np.stack([trial[:, start:start + 2 * fs] for trial in trials
                     for start in (0, fs, 2 * fs)]).astype(np.float32)


def find_recordings(data_dir: Path) -> list[dict]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Dataset requires manifest.json")
    if not isinstance(read_json(manifest_path), dict):
        raise ValueError("manifest.json must contain a JSON object")
    recordings = []
    for subject in SUBJECT_NAMES:
        edf, events = data_dir / "edf" / f"{subject}.edf", data_dir / "events" / f"{subject}.csv"
        if not edf.is_file() or not events.is_file():
            raise FileNotFoundError(f"Missing EDF/CSV pair for {subject}")
        recordings.append({"subject": subject, "edf": edf, "csv": events})
    return recordings


def build_cache(data_dir: Path, output: Path, overwrite: bool = False) -> dict:
    outputs = [output / "mi_windows.npz", output / "metadata.json"]
    if any(p.exists() for p in outputs) and not overwrite:
        raise FileExistsError("Output already exists; choose another --output-dir or explicitly pass --overwrite")
    recordings = find_recordings(data_dir)
    collected = {k: [] for k in ("raw_windows", "aligned_windows", "y", "subject_index", "trial_id",
                                "window_start_seconds", "window_offset_seconds")}
    matrices, inverses, reports, channel_names = [], [], [], None
    for subject_index, source in enumerate(recordings):
        edf_raw, csv_raw = source["edf"].read_bytes(), source["csv"].read_bytes()
        signal, header, events = read_edf(edf_raw)
        edf_sha, csv_sha = hashlib.sha256(edf_raw).hexdigest(), hashlib.sha256(csv_raw).hexdigest()
        del edf_raw
        if channel_names is None:
            channel_names = header["eeg_channels"]
        if header["eeg_channels"] != channel_names:
            raise ValueError("EEG channel order differs across subjects")
        rows, csv_times = parse_csv_events(csv_raw, source["subject"])
        grouped = group_event_bursts(events)
        sync = synchronize_events(csv_times, np.asarray([e["time"] for e in grouped]))
        sos = butter(4, (4, 40), btype="bandpass", fs=500, output="sos")
        filtered = resample_poly(sosfiltfilt(sos, signal, axis=-1), 2, 5, axis=-1)
        del signal
        trials, labels, ids, onsets, trial_reports = [], [], [], [], []
        for row_index, row in enumerate(rows):
            mappings = {}
            for stage_index, stage in enumerate(STAGES):
                k = 4 * row_index + stage_index
                matched = bool(sync["matched"][k])
                observed = grouped[int(sync["nearest"][k])] if matched else None
                mappings[stage] = {"csv_relative_seconds": float(csv_times[k]),
                                   "expected_edf_seconds": float(sync["slope"] * csv_times[k] + sync["offset_seconds"]),
                                   "observed_edf_event": observed,
                                   "residual_ms": float(sync["residual_seconds"][k] * 1000) if matched else None}
            mapped_onset = mappings["mi_onset"]["expected_edf_seconds"]
            mapped_rest = mappings["rest_onset"]["expected_edf_seconds"]
            if not 4.0 <= mapped_rest - mapped_onset <= 4.1:
                raise ValueError("CSV MI-to-rest interval disagrees with the four-second paradigm")
            start = int(round(mapped_onset * 200))
            reason = "Four-second MI segment is outside recorded EEG" if start < 0 or start + 800 > filtered.shape[1] else None
            global_id = subject_index * 1000 + int(row["trial_id"])
            trial_reports.append({"csv_trial": int(row["trial_id"]), "global_trial_id": global_id,
                                  "label": int(row["label"]), "included": reason is None,
                                  "exclusion_reason": reason, "event_mapping": mappings,
                                  "mi_onset_corroborated": mappings["mi_onset"]["observed_edf_event"] is not None,
                                  "rest_onset_corroborated": mappings["rest_onset"]["observed_edf_event"] is not None,
                                  "resampled_start_sample": start})
            if reason is None:
                trials.append(filtered[:, start:start + 800])
                labels.append(int(row["label"]))
                ids.append(global_id)
                onsets.append(start / 200)
        if len(trials) < 30:
            raise ValueError(f"Too few complete MI trials for {source['subject']}; inspect timing and recording length")
        trials = np.stack(trials)
        A, A_inv, ea_info = estimate_ea(trials)
        raw_windows = trial_windows(trials)
        aligned_windows = np.einsum("cd,ndt->nct", A, raw_windows, optimize=True).astype(np.float32)
        n = len(raw_windows)
        values = {"raw_windows": raw_windows, "aligned_windows": aligned_windows,
                  "y": np.repeat(np.asarray(labels, dtype=np.int64), 3),
                  "subject_index": np.full(n, subject_index, dtype=np.int64),
                  "trial_id": np.repeat(np.asarray(ids, dtype=np.int64), 3),
                  "window_start_seconds": np.repeat(onsets, 3) + np.tile([0., 1., 2.], len(trials)),
                  "window_offset_seconds": np.tile([0., 1., 2.], len(trials))}
        for key, value in values.items():
            collected[key].append(value)
        matrices.append(A)
        inverses.append(A_inv)
        sync_report = {k: v for k, v in sync.items() if k not in ("nearest", "matched", "residual_seconds")}
        reports.append({"subject": source["subject"], "subject_index": subject_index,
                        "edf_file": f"edf/{source['subject']}.edf", "csv_file": f"events/{source['subject']}.csv",
                        "edf_sha256": edf_sha, "csv_sha256": csv_sha, "header": header,
                        "csv_trials": len(rows), "retained_trials": len(trials), "excluded_trials": len(rows) - len(trials),
                        "windows": n, "label_counts_trials": dict(Counter(str(v) for v in labels)),
                        "synchronization": sync_report, "euclidean_alignment": ea_info, "trials": trial_reports})
        print(json.dumps({"subject": source["subject"], "trials": len(trials), "windows": n,
                          "max_sync_residual_ms": sync["max_residual_ms"]}, ensure_ascii=True), flush=True)
    arrays = {key: np.concatenate(values) for key, values in collected.items()}
    arrays.update(A=np.stack(matrices), A_inv=np.stack(inverses), subject_names=np.asarray(SUBJECT_NAMES),
                  channel_names=np.asarray(channel_names), mirror_permutation=mirror_permutation(channel_names))
    if not np.isfinite(arrays["raw_windows"]).all() or not np.isfinite(arrays["aligned_windows"]).all():
        raise ValueError("Non-finite EEG windows")
    metadata = {"schema_version": 1, "cache_file": "mi_windows.npz", "subjects": SUBJECT_NAMES,
                "manifest_sha256": sha256_file(data_dir / "manifest.json"),
                "preprocessing_code_sha256": sha256_file(Path(__file__)),
                "software": {"numpy": np.__version__, "scipy": scipy.__version__},
                "protocol": {
                    "task": "MI", "labels": {"0": "left hand", "1": "right hand"},
                    "subject_numbering": "Anonymous identifiers; numerical order does not encode acquisition chronology",
                    "channels": "All 32 EEG channels including M1/M2; CPz reference preserved; BIP and annotations are not model channels",
                    "filter": "Continuous fourth-order Butterworth 4-40 Hz; scipy sosfiltfilt zero phase",
                    "resampling": "500 to 200 Hz; scipy resample_poly up=2 down=5 default anti-alias FIR",
                    "raw_windows_units": "microvolts after filtering and resampling, before EA",
                    "aligned_windows_units": "dimensionless; aligned = A @ raw",
                    "clock_sync": "Label-independent affine fit: EDF_seconds = slope * CSV_relative_seconds + offset",
                    "csv_time_origin": "First baseline onset in that subject's paradigm log",
                    "edf_marker_warning": "Numeric EDF codes differ from paradigm labels; class labels are read exclusively from CSV",
                    "burst_merge_seconds": .020, "clock_match_tolerance_seconds": .020,
                    "trial_selection": "Affine-map every CSV MI onset and retain complete recorded four-second segments",
                    "missing_marker_policy": "Individual absent EDF MI/rest triggers are documented, not excluded after session clock validation",
                    "trial_seconds": 4, "window_seconds": 2, "stride_seconds": 1, "windows_per_trial": 3,
                    "window_start_seconds": "Seconds from EDF sample zero after nearest-200Hz-sample onset rounding",
                    "global_trial_id": "subject_index * 1000 + CSV trial_id",
                    "ea": "Per-subject mean covariance of centered complete MI trials, before overlapping windows; no labels used",
                    "ea_scope": "Whole-session unlabeled statistics, including any subject later selected for evaluation; this is transductive preprocessing",
                    "ea_ridge_relative": 1e-4, "ea_eigenfloor_relative": 1e-6,
                    "partitioning": "Unpartitioned subject windows",
                    "window_grouping": "Each window is contained in one trial; subject_index and trial_id identify its group",
                    "quality_scope": "Timing/completeness checks; no amplitude/artifact-based trial exclusions or manual bad-channel correction"},
                "total_csv_trials": sum(r["csv_trials"] for r in reports),
                "retained_trials": sum(r["retained_trials"] for r in reports), "windows": len(arrays["y"]),
                "array_shapes": {k: list(v.shape) for k, v in arrays.items()}, "subject_reports": reports}
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also catches an output appearing while preprocessing ran.
    mode = "wb" if overwrite else "xb"
    with outputs[0].open(mode) as stream:
        np.savez_compressed(stream, **arrays)
    with outputs[1].open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=True, indent=2)
    return metadata
