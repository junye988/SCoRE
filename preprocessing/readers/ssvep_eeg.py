"""Signal preparation for the released anonymous SSVEP-EEG recordings."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import platform
import re
import warnings

import mne
from mne.bem import fit_sphere_to_headshape
import numpy as np
import scipy
from scipy.signal import butter, resample_poly, sosfiltfilt

from utils.io import read_json, sha256_file


DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent
SUBJECT_NAMES = [f"S{i:03d}" for i in range(1, 13)]
FREQUENCIES = np.asarray([5.0, 7.5, 12.0, 15.0], dtype=np.float64)
STAGES = ("baseline_onset", "cue_onset", "gaze_onset", "rest_onset")
EVENT_COLUMNS = (
    "subject_id", "block", "trial_id", "label", "frequency_hz",
    "baseline_onset_s", "cue_onset_s", "gaze_onset_s", "rest_onset_s",
    "trial_end_s", "baseline_duration_s", "cue_duration_s",
    "gaze_duration_s", "rest_duration_s",
)
OUTPUT_RATE = 200
WINDOW_SAMPLES = 400
MAX_REST_JOIN_GAP_SECONDS = .100
REST_LABEL = 4
NUMERIC_TAL = re.compile(rb"([+-]\d+(?:\.\d+)?)(?:\x15[^\x14]*)?\x14\s*(\d+)\x14")


def read_edf(raw: bytes) -> tuple[np.ndarray, dict, list[dict]]:
    """Read EDF+C EEG in physical microvolts and numeric annotation events."""
    h = raw[:256]
    header_bytes, records, channels = int(h[184:192]), int(h[236:244]), int(h[252:256])
    record_seconds = float(h[244:252])
    if h[192:236].strip() != b"EDF+C" or records <= 0:
        raise ValueError("Only fixed-length continuous EDF+C is supported")
    sh, fields, cursor = raw[256:header_bytes], {}, 0
    for key, width in (("labels", 16), ("transducer", 80), ("units", 8),
                       ("physical_min", 8), ("physical_max", 8), ("digital_min", 8),
                       ("digital_max", 8), ("prefilter", 80), ("samples", 8), ("reserved", 32)):
        fields[key] = [sh[cursor + i * width:cursor + (i + 1) * width].decode("ascii").strip()
                       for i in range(channels)]
        cursor += width * channels
    counts = np.asarray(fields["samples"], dtype=np.int64)
    if len(raw) != header_bytes + records * int(counts.sum()) * 2:
        raise ValueError("EDF data length differs from its header")
    record_data = np.frombuffer(raw, dtype="<i2", offset=header_bytes).reshape(records, int(counts.sum()))
    offsets = np.r_[0, np.cumsum(counts)]
    eeg_indices = [i for i, label in enumerate(fields["labels"]) if label.startswith("EEG ")]
    annotation_indices = [i for i, label in enumerate(fields["labels"]) if label == "EDF Annotations"]
    if len(eeg_indices) != 32 or len(annotation_indices) != 1:
        raise ValueError("Expected 32 EEG channels and one annotation channel")
    rates = {int(counts[i]) / record_seconds for i in eeg_indices}
    if rates != {500.0}:
        raise ValueError(f"Expected 500 Hz EEG, got {rates}")
    eeg = []
    names = []
    for i in eeg_indices:
        if fields["units"][i] != "uV":
            raise ValueError("Unexpected EEG physical unit")
        pmin, pmax = float(fields["physical_min"][i]), float(fields["physical_max"][i])
        dmin, dmax = float(fields["digital_min"][i]), float(fields["digital_max"][i])
        if dmax <= dmin or pmax <= pmin:
            raise ValueError("Invalid EDF physical scaling")
        digital = record_data[:, offsets[i]:offsets[i + 1]].reshape(-1).astype(np.float64)
        eeg.append((digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin)
        names.append(fields["labels"][i].removeprefix("EEG ").removesuffix("-CPz"))
    a = annotation_indices[0]
    annotation_data = record_data[:, offsets[a]:offsets[a + 1]].copy()
    # Continuous sample coordinates must agree with the EDF timekeeping TALs.
    for row, expected in ((0, 0.0), (-1, (records - 1) * record_seconds)):
        first_tal = annotation_data[row].tobytes().split(b"\x14", 1)[0].split(b"\x15", 1)[0]
        if abs(float(first_tal) - expected) > record_seconds + 1e-6:
            raise ValueError("EDF continuous timekeeping does not match sample indices")
    events = [{"time": float(m[1]), "code": int(m[2])}
              for m in NUMERIC_TAL.finditer(annotation_data.tobytes())]
    events.sort(key=lambda e: e["time"])
    info = {"sample_rate_hz": 500, "duration_seconds": records * record_seconds,
            "eeg_channels": names, "physical_unit": "uV", "signal_count": channels,
            "start_date": h[168:176].decode().strip(), "start_time": h[176:184].decode().strip(),
            "edf_numeric_event_count": len(events),
            "edf_numeric_code_counts": dict(Counter(str(e["code"]) for e in events))}
    return np.stack(eeg), info, events


def group_event_bursts(events: list[dict], tolerance: float = 0.020) -> list[dict]:
    """Treat transitions within 20 ms of the first edge as one observed trigger."""
    grouped = []
    for event in events:
        if not grouped or event["time"] - grouped[-1]["time"] > tolerance:
            grouped.append({"time": event["time"], "codes": [event["code"]], "edge_times": [event["time"]]})
        else:
            grouped[-1]["codes"].append(event["code"])
            grouped[-1]["edge_times"].append(event["time"])
    return grouped


def mirror_permutation(channel_names: list[str]) -> np.ndarray:
    pairs = (("Fp1", "Fp2"), ("F7", "F8"), ("F3", "F4"), ("FC5", "FC6"),
             ("FC1", "FC2"), ("M1", "M2"), ("T7", "T8"), ("C3", "C4"),
             ("CP5", "CP6"), ("CP1", "CP2"), ("P7", "P8"), ("P3", "P4"), ("O1", "O2"))
    lookup = {n.casefold(): i for i, n in enumerate(channel_names)}
    permutation = np.arange(len(channel_names), dtype=np.int64)
    for left, right in pairs:
        i, j = lookup[left.casefold()], lookup[right.casefold()]
        permutation[i], permutation[j] = j, i
    if not np.array_equal(permutation[permutation], np.arange(len(permutation))):
        raise ValueError("Mirror permutation is not an involution")
    return permutation


def parse_ssvep_csv(raw: bytes, expected_subject: str | None = None) -> tuple[list[dict], np.ndarray]:
    """Validate 64 trials and return timing relative to the first baseline."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ()) != EVENT_COLUMNS:
        raise ValueError("Event CSV columns or column order do not match the packaged schema")
    rows = list(reader)
    if len(rows) != 64:
        raise ValueError("SSVEP requires 64 numbered trials per participant")
    if any(None in row or any(value is None or not value.strip() for value in row.values()) for row in rows):
        raise ValueError("Event CSV contains missing fields or extra values")
    subjects = {row["subject_id"] for row in rows}
    if len(subjects) != 1 or not re.fullmatch(r"S\d{3}", next(iter(subjects))):
        raise ValueError("Event CSV must contain one anonymous subject identifier")
    if expected_subject is not None and subjects != {expected_subject}:
        raise ValueError("Event CSV subject does not match its filename and manifest")
    numeric_rows, times = [], []
    for row in rows:
        parsed = {"subject_id": row["subject_id"]}
        parsed.update({key: int(row[key]) for key in ("block", "trial_id", "label")})
        parsed.update({key: float(row[key]) for key in EVENT_COLUMNS[4:]})
        if not np.isfinite([parsed[key] for key in EVENT_COLUMNS[4:]]).all():
            raise ValueError("Event CSV timing and frequency values must be finite")
        label, frequency = parsed["label"], parsed["frequency_hz"]
        if label not in range(4) or frequency != FREQUENCIES[label]:
            raise ValueError("Event CSV label and frequency disagree")
        boundaries = [parsed[stage + "_s"] for stage in STAGES] + [parsed["trial_end_s"]]
        durations = [parsed[stage + "_duration_s"] for stage in ("baseline", "cue", "gaze", "rest")]
        if np.any(np.diff(boundaries) <= 0):
            raise ValueError("Event CSV trial boundaries must be strictly increasing")
        if durations != [2.0, 1.0, 4.0, 1.0]:
            raise ValueError("Event CSV nominal phase durations must be 2, 1, 4 and 1 seconds")
        times.extend(boundaries[:-1])
        numeric_rows.append(parsed)
    if [row["trial_id"] for row in numeric_rows] != list(range(1, 65)):
        raise ValueError("Event CSV must contain ordered unique trial identifiers 1 through 64")
    if [row["block"] for row in numeric_rows] != [1] * 32 + [2] * 32:
        raise ValueError("Expected two ordered blocks of 32 trials")
    for block in (1, 2):
        counts = Counter(row["frequency_hz"] for row in numeric_rows if row["block"] == block)
        if counts != Counter({frequency: 8 for frequency in FREQUENCIES}):
            raise ValueError("Expected eight trials per frequency in each block")
    if abs(times[0]) > 1e-9 or not np.all(np.diff(times) > 0):
        raise ValueError("Event CSV times must increase from the first baseline at zero")
    if any(left["trial_end_s"] > right["baseline_onset_s"]
           for left, right in zip(numeric_rows, numeric_rows[1:])):
        raise ValueError("Event CSV trials overlap")
    return numeric_rows, np.asarray(times, dtype=np.float64)


def find_recordings(data_dir: Path) -> list[dict]:
    """Read the local manifest; only the advertised anonymous paths are accepted."""
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "manifest.json")
    if (manifest.get("dataset") != "SSVEP_dataset"
            or manifest.get("subject_count") != len(SUBJECT_NAMES)
            or manifest.get("trial_count") != 64 * len(SUBJECT_NAMES)
            or manifest.get("sampling_rate_hz") != 500
            or manifest.get("eeg_channels") != 32):
        raise ValueError("Manifest does not describe the expected SSVEP dataset")
    records = manifest.get("files", [])
    if (not isinstance(records, list) or len(records) != len(SUBJECT_NAMES)
            or sorted(record.get("subject_id", "") for record in records) != SUBJECT_NAMES):
        raise ValueError("Manifest must list exactly S001 through S012 once each")
    recordings = []
    for record in sorted(records, key=lambda item: item["subject_id"]):
        subject = record["subject_id"]
        if (record.get("edf") != f"edf/{subject}.edf"
                or record.get("events") != f"events/{subject}.csv"
                or record.get("trial_count") != 64):
            raise ValueError("Manifest recording paths or trial counts are invalid")
        for key in ("edf_sha256", "events_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(record.get(key, ""))):
                raise ValueError("Manifest requires valid SHA-256 checksums for both recording files")
        for key in ("edf", "events"):
            path = data_dir / record[key]
            if not path.is_file() or not path.resolve().is_relative_to(data_dir.resolve()):
                raise ValueError(f"Missing or out-of-directory recording: {record[key]}")
        recordings.append({**record, "subject": subject,
                           "edf_path": data_dir / record["edf"],
                           "events_path": data_dir / record["events"]})
    return recordings


def _fit_clock_candidate(x: np.ndarray, t: np.ndarray, offset: float) -> dict | None:
    slope = 1.0
    for _ in range(6):
        nearest = np.argmin(np.abs((slope * x + offset)[:, None] - t), axis=1)
        residual = t[nearest] - slope * x - offset
        fit = np.abs(residual) < 0.030
        if fit.sum() < 20:
            return None
        slope, offset = (float(v) for v in np.polyfit(x[fit], t[nearest[fit]], 1))
    nearest = np.argmin(np.abs((slope * x + offset)[:, None] - t), axis=1)
    residual = t[nearest] - slope * x - offset
    matched = np.abs(residual) <= 0.020
    unique = np.unique(nearest[matched])
    if abs(slope - 1) > 1e-4 or len(unique) != int(matched.sum()) or not matched.any():
        return None
    return {"slope": slope, "offset_seconds": offset, "nearest": nearest,
            "matched": matched, "residual_seconds": residual,
            "matched_csv_events": int(matched.sum()), "edf_grouped_events": len(t),
            "csv_match_fraction": float(matched.mean()), "edf_match_fraction": len(unique) / len(t),
            "max_residual_ms": float(np.max(np.abs(residual[matched])) * 1000),
            "rms_residual_ms": float(np.sqrt(np.mean(residual[matched] ** 2)) * 1000),
            "drift_ppm": (slope - 1) * 1e6,
            "matched_csv_span_fraction": float(np.ptp(x[matched]) / np.ptp(x))}


def synchronize_ssvep_events(csv_times: np.ndarray, edf_times: np.ndarray) -> dict:
    """Resolve periodic-trial offset aliases with whole-session timing evidence."""
    x, t = np.asarray(csv_times, float), np.asarray(edf_times, float)
    if (x.ndim != 1 or t.ndim != 1 or len(x) < 20 or len(t) < 20
            or not np.isfinite(x).all() or not np.isfinite(t).all()
            or np.any(np.diff(x) <= 0) or np.any(np.diff(t) <= 0)):
        raise ValueError("Synchronization requires finite increasing event sequences")
    xd, td = np.diff(x), np.diff(t)
    xi, ti = int(np.argmax(xd)), int(np.argmax(td))
    if (xd[xi] < max(20.0, 2 * np.partition(xd, -2)[-2])
            or td[ti] < max(20.0, 2 * np.partition(td, -2)[-2])):
        raise ValueError("No unique session pause to distinguish periodic trial aliases")
    deltas = (t[:, None] - x[None, :]).ravel()
    bins = np.rint(deltas / 0.020).astype(np.int64)
    keys, votes = np.unique(bins, return_counts=True)
    candidate_indices = np.argsort(votes)[-128:][::-1]
    candidates = []
    for index in candidate_indices:
        offset = float(np.median(deltas[bins == keys[index]]))
        candidate = _fit_clock_candidate(x, t, offset)
        if candidate is None:
            continue
        candidate["initial_offset_votes"] = int(votes[index])
        # Adjacent histogram bins can converge to the same clock.
        endpoints = candidate["slope"] * x[[0, -1]] + candidate["offset_seconds"]
        equivalent = next((i for i, c in enumerate(candidates)
                           if np.max(np.abs(endpoints - (c["slope"] * x[[0, -1]] + c["offset_seconds"]))) < .020), None)
        if equivalent is not None:
            old = candidates[equivalent]
            if (-candidate["matched_csv_events"], candidate["rms_residual_ms"]) < (-old["matched_csv_events"], old["rms_residual_ms"]):
                candidates[equivalent] = candidate
        else:
            candidates.append(candidate)
    candidates.sort(key=lambda c: (-c["matched_csv_events"], c["rms_residual_ms"]))
    passing = []
    for candidate in candidates:
        pause_ok = bool(candidate["matched"][xi:xi + 2].all()
                        and np.array_equal(candidate["nearest"][xi:xi + 2], [ti, ti + 1]))
        candidate["pause_endpoints_match"] = pause_ok
        if (candidate["csv_match_fraction"] >= .70 and candidate["edf_match_fraction"] >= .98
                and candidate["matched_csv_span_fraction"] >= .95
                and candidate["rms_residual_ms"] <= 3.0 and pause_ok):
            passing.append(candidate)
    if len(passing) != 1:
        raise ValueError(f"Expected one verified SSVEP session clock, found {len(passing)}")
    result = passing[0].copy()
    result["candidate_offset_bins_evaluated"] = len(candidate_indices)
    result["distinct_clock_candidates"] = len(candidates)
    result["verified_clock_count"] = 1
    result["csv_events"] = len(x)
    result["pause_anchor"] = {"csv_before_event_index": xi, "csv_after_event_index": xi + 1,
                              "edf_before_event_index": ti, "edf_after_event_index": ti + 1,
                              "csv_gap_seconds": float(xd[xi]), "edf_gap_seconds": float(td[ti]),
                              "endpoint_residual_ms": (result["residual_seconds"][xi:xi + 2] * 1000).tolist()}
    alternatives = [c for c in candidates if c is not passing[0]][:5]
    result["alternative_clock_candidates"] = [
        {k: v for k, v in c.items() if k not in ("nearest", "matched", "residual_seconds")}
        for c in alternatives]
    return result


def _validate_signal(signal_uV, channel_names, fs):
    signal = np.asarray(signal_uV)
    names = tuple(str(name) for name in channel_names)
    if (signal.ndim != 2 or signal.shape[0] != len(names) or signal.shape[1] == 0
            or signal.dtype.kind != "f" or not np.isfinite(fs) or fs <= 0):
        raise ValueError("Expected floating EEG [channels,samples], matching names and positive finite fs")
    if len(names) < 4 or len(set(n.casefold() for n in names)) != len(names):
        raise ValueError("At least four uniquely named EEG channels are required")
    return signal, names


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open [start,end) intervals for a one-dimensional boolean mask."""
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _intervals(mask: np.ndarray, fs: float) -> list[dict]:
    return [{"start_sample": start, "end_sample_exclusive": end,
             "start_seconds": start / fs, "end_seconds_exclusive": end / fs}
            for start, end in _runs(mask)]


def detect_bad_samples(signal_uV, fs=500, *, flat_min_seconds=.5,
                       flat_tolerance_uV=1e-6, rail_values_uV=(-32767., 32767.),
                       rail_tolerance_uV=1e-3):
    """Detect non-finite values, sustained flatlines and known physical rail codes."""
    x = np.asarray(signal_uV)
    if x.ndim != 2 or x.dtype.kind != "f" or x.shape[1] == 0:
        raise ValueError("Expected a nonempty floating EEG array [channels,samples]")
    parameters = [fs, flat_min_seconds, flat_tolerance_uV, rail_tolerance_uV]
    if (not np.isfinite(parameters).all() or fs <= 0 or flat_min_seconds <= 0
            or flat_tolerance_uV < 0 or rail_tolerance_uV < 0
            or not np.isfinite(rail_values_uV).all()):
        raise ValueError("Defect detection parameters must be finite and nonnegative")
    finite = np.isfinite(x)
    nonfinite = ~finite
    rails = np.zeros_like(finite)
    for value in rail_values_uV:
        rails |= finite & np.isclose(x, value, atol=rail_tolerance_uV, rtol=0)
    flat = np.zeros_like(finite)
    minimum_samples = max(2, int(np.ceil(flat_min_seconds * fs)))
    with np.errstate(invalid="ignore"):
        for channel in range(len(x)):
            if finite[channel].all() and np.ptp(x[channel]) <= flat_tolerance_uV:
                flat[channel] = True
                continue
            constant_edges = (finite[channel, :-1] & finite[channel, 1:]
                              & (np.abs(np.diff(x[channel].astype(np.float64))) <= flat_tolerance_uV))
            for start, end in _runs(constant_edges):
                if end - start + 1 >= minimum_samples:
                    flat[channel, start:end + 1] = True
    reasons = {"nonfinite": nonfinite, "known_adc_rail": rails, "flatline": flat}
    return nonfinite | rails | flat, reasons


@lru_cache(maxsize=16)
def _template_geometry(names: tuple[str, ...], fs: float):
    montage = mne.channels.make_standard_montage("standard_1020")
    available = {name.casefold(): name for name in montage.ch_names}
    canonical = []
    for name in names:
        base = name.removeprefix("EEG ").removesuffix("-CPz")
        if base.casefold() not in available:
            raise ValueError(f"Channel has no standard_1020 position: {base}")
        canonical.append(available[base.casefold()])
    if len(set(canonical)) != len(canonical):
        raise ValueError("Channel aliases map to duplicate electrode positions")
    info = mne.create_info(canonical, sfreq=fs, ch_types="eeg", verbose="ERROR")
    info.set_montage(montage, on_missing="raise", verbose="ERROR")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        radius, origin, _ = fit_sphere_to_headshape(info, dig_kinds="eeg", units="m", verbose="ERROR")
    return info, np.asarray(origin), float(radius), tuple(str(w.message) for w in caught)


@lru_cache(maxsize=256)
def _spline_weights(names: tuple[str, ...], fs: float, bad_indices: tuple[int, ...]):
    """Obtain linear interpolation weights through the public MNE Raw API."""
    info, origin, _, geometry_warnings = _template_geometry(names, fs)
    bad = np.asarray(bad_indices, dtype=int)
    good = np.asarray([i for i in range(len(names)) if i not in bad_indices], dtype=int)
    probe = mne.io.RawArray(np.eye(len(names), dtype=np.float64), info.copy(), verbose="ERROR")
    probe.info["bads"] = [probe.ch_names[i] for i in bad]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe.interpolate_bads(reset_bads=False, origin=origin, mode="accurate",
                               method={"eeg": "spline"}, verbose="ERROR")
    weights = probe.get_data()[np.ix_(bad, good)]
    if not np.isfinite(weights).all():
        raise RuntimeError("MNE produced non-finite spline weights")
    # The spline's constant-field constraint should reproduce a constant scalp potential.
    if np.max(np.abs(weights.sum(axis=1) - 1)) > 1e-4:
        raise RuntimeError("Spline weights failed the constant-field constraint")
    return good, weights, geometry_warnings + tuple(str(w.message) for w in caught)


def repair_continuous(signal_uV, channel_names, fs=500, *, min_donors=4,
                      flat_min_seconds=.5, flat_tolerance_uV=1e-6,
                      rail_values_uV=(-32767., 32767.), rail_tolerance_uV=1e-3,
                      chunk_samples=10000):
    """Return ``(repaired_uV, detected_bad_mask, audit)`` for continuous EEG."""
    x, names = _validate_signal(signal_uV, channel_names, fs)
    if not isinstance(min_donors, int) or not 4 <= min_donors <= len(names):
        raise ValueError("min_donors must be an integer between four and the channel count")
    if not isinstance(chunk_samples, int) or chunk_samples <= 0:
        raise ValueError("chunk_samples must be a positive integer")
    bad_mask, reasons = detect_bad_samples(
        x, fs, flat_min_seconds=flat_min_seconds, flat_tolerance_uV=flat_tolerance_uV,
        rail_values_uV=rail_values_uV, rail_tolerance_uV=rail_tolerance_uV)
    repaired = x.copy()
    repaired_mask = np.zeros_like(bad_mask)
    baselines = np.zeros(len(names), dtype=np.float64)
    baseline_counts = (~bad_mask).sum(axis=1)
    for channel in range(len(names)):
        if baseline_counts[channel]:
            baselines[channel] = np.median(x[channel, ~bad_mask[channel]].astype(np.float64))
    info, origin, radius, geometry_warnings = _template_geometry(names, float(fs))
    patterns, donor_minima = [], np.full(len(names), len(names) + 1, dtype=int)
    warning_messages = set(geometry_warnings)
    # Group by the instantaneous bad-channel set, not by trial/class or time-window labels.
    packed = np.packbits(bad_mask, axis=0, bitorder="little").T
    unique, inverse = np.unique(packed, axis=0, return_inverse=True)
    for index, packed_pattern in enumerate(unique):
        flags = np.unpackbits(packed_pattern, bitorder="little")[:len(names)].astype(bool)
        bad = np.flatnonzero(flags)
        if len(bad) == 0:
            continue
        times = np.flatnonzero(inverse == index)
        donor_count = len(names) - len(bad)
        donor_minima[bad] = np.minimum(donor_minima[bad], donor_count)
        entry = {"bad_channels": [names[i] for i in bad], "donor_count": donor_count,
                 "time_samples": len(times), "channel_samples": len(times) * len(bad),
                 "repaired": donor_count >= min_donors}
        if donor_count >= min_donors:
            good, weights, matrix_warnings = _spline_weights(names, float(fs), tuple(bad.tolist()))
            warning_messages.update(matrix_warnings)
            entry.update(donor_channels=[names[i] for i in good],
                         max_absolute_weight=float(np.max(np.abs(weights))),
                         max_weight_row_l1_norm=float(np.max(np.sum(np.abs(weights), axis=1))))
            for start in range(0, len(times), chunk_samples):
                chunk = times[start:start + chunk_samples]
                donors = x[np.ix_(good, chunk)].astype(np.float64) - baselines[good, None]
                estimate = weights @ donors + baselines[bad, None]
                if not np.isfinite(estimate).all():
                    raise RuntimeError("Non-finite spatial estimate despite finite donors")
                repaired[np.ix_(bad, chunk)] = estimate
                repaired_mask[np.ix_(bad, chunk)] = True
        else:
            entry["unrepaired_reason"] = "Fewer than the declared minimum simultaneous good donors"
        patterns.append(entry)
    if not np.array_equal(repaired[~bad_mask], x[~bad_mask]):
        raise AssertionError("Repair modified an observed good sample")
    unresolved = bad_mask & ~repaired_mask
    channels = []
    for i, name in enumerate(names):
        channels.append({"channel": name, "detected_bad_samples": int(bad_mask[i].sum()),
                         "repaired_samples": int(repaired_mask[i].sum()), "unrepaired_samples": int(unresolved[i].sum()),
                         "bad_fraction": float(bad_mask[i].mean()), "repair_fraction": float(repaired_mask[i].mean()),
                         "minimum_donor_count": int(donor_minima[i]) if bad_mask[i].any() else None,
                         "dc_baseline_uV": float(baselines[i]), "baseline_good_sample_count": int(baseline_counts[i]),
                         "baseline_source": "median detected-good samples" if baseline_counts[i] else "zero; no valid target baseline",
                         "reason_samples": {key: int(value[i].sum()) for key, value in reasons.items()},
                         "reason_intervals": {key: _intervals(value[i], fs) for key, value in reasons.items()},
                         "bad_intervals": _intervals(bad_mask[i], fs),
                         "repaired_intervals": _intervals(repaired_mask[i], fs),
                         "unrepaired_intervals": _intervals(unresolved[i], fs)})
    bad_sample_counts = bad_mask.sum(axis=0)
    audit = {"method": "MNE public Raw.interpolate_bads(method={'eeg':'spline'}) spatial interpolation",
             "reference": "https://mne.tools/stable/auto_examples/preprocessing/interpolate_bad_channels.html",
             "mne_version": mne.__version__, "montage": "standard_1020",
             "positions": "Template electrode coordinates only; no individual digitization is available",
             "sphere_origin_head_m": origin.tolist(), "sphere_radius_m": radius,
             "sphere_fit_basis": "All supplied template EEG electrode positions, shared across bad-channel patterns",
             "sampling_rate_hz": float(fs), "shape": list(x.shape), "output_dtype": str(repaired.dtype),
             "detection": {"flat_min_seconds": flat_min_seconds, "flat_tolerance_uV": flat_tolerance_uV,
                           "rail_values_uV": list(rail_values_uV), "rail_tolerance_uV": rail_tolerance_uV,
                           "nonfinite": True, "amplitude_rejection": False, "padding_samples": 0},
             "dc_policy": "Subtract each donor's full-recording detected-good median inside the interpolation; add target good-data median, or zero for an entirely missing target. Good samples remain unchanged.",
             "donor_policy": "Other detected-good channels at the exact same sample; repaired values are never used as donors",
             "minimum_required_donors": min_donors,
             "minimum_donor_count": int(len(names) - bad_sample_counts[bad_sample_counts > 0].max()) if bad_mask.any() else None,
             "maximum_simultaneously_bad_channels": int(bad_sample_counts.max()),
             "detected_bad_samples": int(bad_mask.sum()), "repaired_samples": int(repaired_mask.sum()),
             "unrepaired_samples": int(unresolved.sum()), "repair_fraction": float(repaired_mask.mean()),
             "observed_good_samples_unchanged": True, "input_modified": False,
             "uses_class_labels": False, "mixes_participants": False,
             "limitation": "Spatial estimates cannot recover unobserved true brain activity; broad regions with many missing channels have limited spatial support.",
             "warnings": sorted(warning_messages), "interpolation_patterns": patterns, "channels": channels}
    return repaired, bad_mask, audit


def make_segments(rows: list[dict], slope: float, offset: float,
                  subject_index: int, recording_samples: int) -> tuple[list[dict], dict]:
    """Create disjoint half-open segments, then inward-round their 200-Hz bounds."""
    if (not rows or not np.isfinite([slope, offset]).all() or slope <= 0
            or not isinstance(subject_index, int) or subject_index < 0
            or not isinstance(recording_samples, int) or recording_samples <= 0):
        raise ValueError("Invalid segment-building inputs")
    mapped = []
    for row in rows:
        stage_times = [float(row[stage + "_s"]) for stage in STAGES]
        stage_times.append(float(row["trial_end_s"]))
        times = slope * np.asarray(stage_times) + offset
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("CSV trial stages must be finite and strictly ordered")
        frequency = float(row["frequency_hz"])
        if frequency not in FREQUENCIES:
            raise ValueError("Unexpected active frequency")
        mapped.append({"trial": int(row["trial_id"]), "block": int(row["block"]),
                       "frequency": frequency, "times": times})
    if len(set(r["trial"] for r in mapped)) != len(mapped):
        raise ValueError("Duplicate CSV trial identifiers")
    segments, joins, block_pauses = [], [], []

    def append(kind, start, end, parents, block, frequency=0.0):
        start_sample = int(np.ceil(start * OUTPUT_RATE))
        end_sample = int(np.floor(end * OUTPUT_RATE))
        if start_sample < 0 or end_sample > recording_samples:
            raise ValueError("A declared class segment is outside the recorded EEG")
        if end_sample <= start_sample:
            raise ValueError("No samples inside a declared segment")
        if segments and start < segments[-1]["end_seconds"] - 1e-9:
            raise ValueError("Class segments overlap")
        n = end_sample - start_sample
        included = n >= WINDOW_SAMPLES
        count = (4 if kind == "active" else 2 if kind == "rest_merged" else 1) if included else 0
        starts = (np.rint(np.linspace(start_sample, end_sample - WINDOW_SAMPLES, count)).astype(np.int64)
                  if count > 1 else np.asarray([start_sample], dtype=np.int64) if count else np.asarray([], dtype=np.int64))
        if len(np.unique(starts)) != len(starts):
            raise ValueError("Segment is too short for its declared distinct window starts")
        if np.any(starts < start_sample) or np.any(starts + WINDOW_SAMPLES > end_sample):
            raise AssertionError("A window extends outside its class segment")
        segment_id = subject_index * 1000 + len(segments) + 1
        parent_ids = [subject_index * 1000 + p for p in parents]
        segments.append({"segment_id": segment_id, "segment_kind": kind,
                         "parent_trial_id": parent_ids[0], "parent_trial_ids": parent_ids,
                         "csv_parent_trials": parents, "block": block,
                         "label": int(np.flatnonzero(FREQUENCIES == frequency)[0]) if frequency else REST_LABEL,
                         "frequency_hz": float(frequency), "start_seconds": float(start),
                         "end_seconds": float(end), "duration_seconds": float(end - start),
                         "resampled_start_sample": start_sample,
                         "resampled_end_sample_exclusive": end_sample,
                         "resampled_samples": n, "included_in_windows_and_ea": included,
                         "exclusion_reason": None if included else "Shorter than a complete two-second window after inward boundary rounding",
                         "window_start_samples": starts.tolist(),
                         "window_start_seconds": (starts / OUTPUT_RATE).tolist(),
                         "window_offset_seconds": ((starts - start_sample) / OUTPUT_RATE).tolist()})

    for i, trial in enumerate(mapped):
        baseline, cue, gaze, rest, end = trial["times"]
        previous = mapped[i - 1] if i else None
        if previous is None or previous["block"] != trial["block"]:
            if previous is not None:
                block_pauses.append({"previous_block": previous["block"], "next_block": trial["block"],
                                     "start_seconds": float(previous["times"][-1]),
                                     "end_seconds": float(baseline),
                                     "duration_seconds": float(baseline - previous["times"][-1])})
                if baseline < previous["times"][-1]:
                    raise ValueError("Successive blocks overlap")
            append("rest_baseline", baseline, cue, [trial["trial"]], trial["block"])
        elif not joins[-1]["merged"]:
            append("rest_baseline", baseline, cue, [trial["trial"]], trial["block"])
        append("active", cue, rest, [trial["trial"]], trial["block"], trial["frequency"])
        following = mapped[i + 1] if i + 1 < len(mapped) else None
        if following is not None and following["block"] == trial["block"]:
            gap = float(following["times"][0] - end)
            if gap < 0:
                raise ValueError("Within-block CSV trials overlap")
            merge = gap <= MAX_REST_JOIN_GAP_SECONDS
            joins.append({"previous_csv_trial": trial["trial"], "next_csv_trial": following["trial"],
                          "block": trial["block"], "gap_seconds": gap, "merged": merge,
                          "gap_start_seconds": float(end), "gap_end_seconds": float(following["times"][0])})
            if merge:
                append("rest_merged", rest, following["times"][1],
                       [trial["trial"], following["trial"]], trial["block"])
            else:
                append("rest_postgaze", rest, end, [trial["trial"]], trial["block"])
        else:
            append("rest_postgaze", rest, end, [trial["trial"]], trial["block"])
    if any(a["resampled_end_sample_exclusive"] > b["resampled_start_sample"]
           for a, b in zip(segments, segments[1:])):
        raise AssertionError("Rounded segments overlap")
    gaps = [item["gap_seconds"] for item in joins]
    qa = {"rest_join_max_gap_seconds": MAX_REST_JOIN_GAP_SECONDS,
          "within_block_gaps": joins, "within_block_gap_range_seconds": [min(gaps), max(gaps)] if gaps else None,
          "rest_joins": sum(j["merged"] for j in joins),
          "rejected_rest_joins": sum(not j["merged"] for j in joins),
          "block_pauses_excluded": block_pauses,
          "segment_overlaps": 0, "out_of_segment_windows": 0,
          "windows_crossing_blocks_or_active_rest_boundaries": 0,
          "total_segments": len(segments),
          "eligible_segments": sum(s["included_in_windows_and_ea"] for s in segments),
          "short_segments_excluded_from_windows_and_ea": sum(not s["included_in_windows_and_ea"] for s in segments)}
    return segments, qa


def estimate_segment_ea(signal: np.ndarray, segments: list[dict],
                        ridge_relative: float = 1e-4,
                        eigenfloor_relative: float = 1e-6) -> tuple[np.ndarray, np.ndarray, dict]:
    """Time-weighted covariance of unique eligible samples centered per segment."""
    x = np.asarray(signal)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("EA requires finite [channels,samples] signal")
    if not np.isfinite([ridge_relative, eigenfloor_relative]).all() or min(ridge_relative, eigenfloor_relative) <= 0:
        raise ValueError("EA regularization must be positive and finite")
    reference = np.zeros((len(x), len(x)), dtype=np.float64)
    count, ids, previous_end = 0, [], -1
    for segment in segments:
        if not segment["included_in_windows_and_ea"]:
            continue
        start, end = segment["resampled_start_sample"], segment["resampled_end_sample_exclusive"]
        if start < previous_end or start < 0 or end > x.shape[1] or end - start < WINDOW_SAMPLES:
            raise ValueError("EA segments must be disjoint, ordered, in range and at least two seconds")
        centered = x[:, start:end].astype(np.float64)
        centered -= centered.mean(axis=-1, keepdims=True)
        reference += centered @ centered.T
        count += end - start
        ids.append(segment["segment_id"])
        previous_end = end
    if not count:
        raise ValueError("No eligible samples for EA")
    reference /= count
    scale = float(np.trace(reference) / len(x))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid session covariance")
    regularized = reference + ridge_relative * scale * np.eye(len(x))
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    eigenvalues = np.maximum(eigenvalues, eigenfloor_relative * scale)
    A = (eigenvectors * eigenvalues ** -.5) @ eigenvectors.T
    A_inv = (eigenvectors * eigenvalues ** .5) @ eigenvectors.T
    info = {"fit_segments": len(ids), "fit_segment_ids": ids, "fit_unique_samples": count,
            "fit_unique_seconds": count / OUTPUT_RATE, "fit_uses_labels": False,
            "reference_trace_per_channel": scale, "ridge_relative": ridge_relative,
            "ridge_absolute": ridge_relative * scale, "eigenfloor_relative": eigenfloor_relative,
            "regularized_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            "A_A_inv_max_error": float(np.max(np.abs(A @ A_inv - np.eye(len(x)))))}
    return A, A_inv, info


def window_repair_fraction(bad_mask: np.ndarray, start_sample: int) -> float:
    """Fraction of original channel-samples whose time cells overlap this window."""
    if bad_mask.ndim != 2 or bad_mask.dtype != np.bool_ or start_sample < 0:
        raise ValueError("Expected a boolean 500-Hz defect mask and nonnegative start")
    lo = start_sample * 5 // 2
    hi = ((start_sample + WINDOW_SAMPLES) * 5 + 1) // 2
    if hi > bad_mask.shape[1]:
        raise ValueError("Window source-mask coverage exceeds the original recording")
    return float(bad_mask[:, lo:hi].mean())


def _clock_report(sync: dict, csv_times: np.ndarray, grouped: list[dict]) -> dict:
    report = {k: v for k, v in sync.items() if k not in ("nearest", "matched", "residual_seconds")}
    report["event_mapping"] = []
    for i, timestamp in enumerate(csv_times):
        matched = bool(sync["matched"][i])
        report["event_mapping"].append({
            "csv_trial": i // 4 + 1, "stage": STAGES[i % 4],
            "csv_relative_seconds": float(timestamp),
            "mapped_edf_seconds": float(sync["slope"] * timestamp + sync["offset_seconds"]),
            "observed_edf_event_index": int(sync["nearest"][i]) if matched else None,
            "observed_edf_event": grouped[int(sync["nearest"][i])] if matched else None,
            "residual_ms": float(sync["residual_seconds"][i] * 1000) if matched else None})
    return report


def build_cache(data_dir: Path, output: Path, overwrite: bool = False) -> dict:
    data_dir, output = Path(data_dir), Path(output)
    if not overwrite and any((output / name).exists() for name in ("ssvep_windows.npz", "metadata.json")):
        raise FileExistsError("Five-class cache exists; explicitly pass --overwrite to rebuild")
    recordings = find_recordings(data_dir)
    collected = {key: [] for key in (
        "raw_windows", "aligned_windows", "y", "subject_index", "trial_id", "parent_trial_id",
        "segment_kind", "window_start_seconds", "window_offset_seconds", "block", "frequency_hz", "repair_fraction")}
    matrices, inverses, reports, channel_names = [], [], [], None
    for subject_index, source in enumerate(recordings):
        print(json.dumps({"subject": source["subject"], "stage": "read_synchronize_repair"}), flush=True)
        edf_raw, csv_raw = source["edf_path"].read_bytes(), source["events_path"].read_bytes()
        source_sha = {"edf_sha256": hashlib.sha256(edf_raw).hexdigest(),
                      "events_sha256": hashlib.sha256(csv_raw).hexdigest()}
        if any(source_sha[key] != source[key] for key in source_sha):
            raise ValueError(f"Recording checksum differs from manifest: {source['subject']}")
        signal, header, events = read_edf(edf_raw)
        del edf_raw
        header.pop("start_date", None)
        header.pop("start_time", None)
        if header["sample_rate_hz"] != 500 or signal.shape[0] != 32:
            raise ValueError("Expected 32 EEG channels sampled at 500 Hz")
        if channel_names is None:
            channel_names = header["eeg_channels"]
        if header["eeg_channels"] != channel_names:
            raise ValueError("EEG channel order differs across participants")
        rows, csv_times = parse_ssvep_csv(csv_raw, source["subject"])
        grouped = group_event_bursts(events)
        sync = synchronize_ssvep_events(csv_times, np.asarray([e["time"] for e in grouped]))
        repaired, bad_mask, repair_audit = repair_continuous(signal, channel_names, fs=500)
        if repair_audit["unrepaired_samples"] or not np.isfinite(repaired).all():
            raise ValueError("Cannot filter a session with unresolved defective samples")
        del signal
        sos = butter(4, (3, 45), btype="bandpass", fs=500, output="sos")
        filtered = resample_poly(sosfiltfilt(sos, repaired, axis=-1), 2, 5, axis=-1)
        del repaired
        segments, qa = make_segments(rows, sync["slope"], sync["offset_seconds"], subject_index, filtered.shape[1])
        A, A_inv, ea_info = estimate_segment_ea(filtered, segments)
        entries = [(segment, start) for segment in segments for start in segment["window_start_samples"]]
        raw_windows = np.stack([filtered[:, start:start + WINDOW_SAMPLES] for _, start in entries]).astype(np.float32)
        del filtered
        aligned = np.einsum("cd,ndt->nct", A, raw_windows, optimize=True).astype(np.float32)
        n = len(entries)
        def integers(key):
            return np.asarray([segment[key] for segment, _ in entries], dtype=np.int64)
        values = {"raw_windows": raw_windows, "aligned_windows": aligned,
                  "y": integers("label"), "subject_index": np.full(n, subject_index, dtype=np.int64),
                  "trial_id": integers("segment_id"), "parent_trial_id": integers("parent_trial_id"),
                  "segment_kind": np.asarray([segment["segment_kind"] for segment, _ in entries]),
                  "block": integers("block"),
                  "frequency_hz": np.asarray([segment["frequency_hz"] for segment, _ in entries], dtype=np.float64),
                  "window_start_seconds": np.asarray([start / OUTPUT_RATE for _, start in entries], dtype=np.float64),
                  "window_offset_seconds": np.asarray([(start - segment["resampled_start_sample"]) / OUTPUT_RATE for segment, start in entries], dtype=np.float64),
                  "repair_fraction": np.asarray([window_repair_fraction(bad_mask, start) for _, start in entries], dtype=np.float64)}
        del bad_mask
        if not np.isfinite(raw_windows).all() or not np.isfinite(aligned).all():
            raise ValueError("Non-finite five-class windows")
        for key, value in values.items():
            collected[key].append(value)
        matrices.append(A)
        inverses.append(A_inv)
        report = {"subject": source["subject"], "subject_index": subject_index,
                  **source_sha, "header": header,
                  "csv_trials": len(rows), "windows": n, "label_counts_windows": dict(Counter(str(v) for v in values["y"])),
                  "segment_kind_counts": dict(Counter(s["segment_kind"] for s in segments)),
                  "segment_kind_window_counts": dict(Counter(values["segment_kind"].tolist())),
                  "repair": repair_audit, "boundary_qa": qa, "euclidean_alignment": ea_info,
                  "synchronization": _clock_report(sync, csv_times, grouped), "segments": segments}
        reports.append(report)
        print(json.dumps({"subject": source["subject"], "stage": "complete", "windows": n,
                          "label_counts": report["label_counts_windows"],
                          "repair_fraction": repair_audit["repair_fraction"],
                          "unrepaired_samples": repair_audit["unrepaired_samples"],
                          "rest_joins": qa["rest_joins"], "short_segments": qa["short_segments_excluded_from_windows_and_ea"]}), flush=True)
    arrays = {key: np.concatenate(values) for key, values in collected.items()}
    arrays.update(A=np.stack(matrices), A_inv=np.stack(inverses), subject_names=np.asarray(SUBJECT_NAMES),
                  channel_names=np.asarray(channel_names), mirror_permutation=mirror_permutation(channel_names))
    metadata = {
        "schema_version": 1, "cache_file": "ssvep_windows.npz", "subjects": SUBJECT_NAMES,
        "preprocessing_code_sha256": sha256_file(Path(__file__)),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "mne": mne.__version__},
        "protocol": {
            "task": "Five-class SSVEP cue-plus-gaze versus protocol-defined rest",
            "labels": {"0": "5 Hz", "1": "7.5 Hz", "2": "12 Hz", "3": "15 Hz", "4": "rest"},
            "frequency_hz": "0 for rest; otherwise the CSV target frequency",
            "channels": "All 32 EEG channels including M1/M2; original CPz reference; no BIP inputs",
            "processing_order": ["source EEG defect detection and same-time spatial repair", "continuous 3-45 Hz bandpass",
                                 "500 to 200 Hz resampling", "actual-boundary segmentation", "session EA reference from unique eligible segment samples", "two-second windows and application of EA"],
            "repair": "repair_continuous: non-finite samples, >=0.5 s flatlines and known ADC rails; standard_1020 template spherical splines with >=4 original good simultaneous donors; good samples unchanged before filtering",
            "repair_scope": "Per-session label-free spatial estimation, including good-sample median DC baselines; not recovery of true missing activity",
            "filter": "Continuous Butterworth bandpass 3-45 Hz, scipy butter N=4, zero-phase sosfiltfilt",
            "resampling": "500 to 200 Hz with scipy resample_poly up=2, down=5, default anti-alias FIR; no time warping",
            "clock": "CSV relative event-log times mapped to EDF-relative seconds by the separately verified timing-only affine event clock and unique inter-block pause",
            "active": "CSV cue onset through CSV rest onset (cue plus gaze), assigned CSV target frequency",
            "rest": "CSV baseline through cue and CSV rest onset through trial end; same-block neighboring rest/baseline intervals joined only for a nonnegative gap <=0.100 s",
            "rest_join_gap_policy": "The short logged trial-end-to-next-baseline gap is assigned to the merged surrounding rest; every such gap is reported; block pauses never included",
            "boundary_policy": "200-Hz start=ceil(mapped start*200), end=floor(mapped end*200), half-open [start,end); all windows remain strictly within these bounds",
            "window_samples": WINDOW_SAMPLES, "window_seconds": 2.0, "sampling_rate_hz": OUTPUT_RATE,
            "window_starts": "Active: round(linspace(start,end-400,4)); merged rest: round(linspace(start,end-400,2)); standalone baseline/postrest: one start if >=400 samples. No padding, stretching or nominal-duration extension.",
            "window_stride": "Boundary-safe approximately one second, not a fixed stride; exact starts stored in each window and segment",
            "short_segment_policy": "Segments shorter than 400 inward-rounded samples excluded from both window generation and EA; retained in metadata",
            "ea": "Center each eligible segment across time separately; reference=sum(segment X_centered @ X_centered.T)/sum(segment sample counts). Eligible segments are disjoint, hence no repeated samples or overlapping-window weighting. Relative ridge=1e-4; relative eigenfloor=1e-6.",
            "ea_scope": "Whole-session unlabeled transductive preprocessing, including participants later held out; all eligible active/rest samples, with no label-dependent weighting or class balancing",
            "raw_windows_units": "Microvolts after repair/filter/resampling, before EA",
            "aligned_windows_units": "Dimensionless; aligned_windows=A @ raw_windows",
            "trial_id": "Globally unique continuous segment ID, subject_index*1000+one-based chronological segment ordinal; NOT the original trial ID",
            "parent_trial_id": "subject_index*1000+CSV trial_id; merged rest uses preceding trial here and both parents in segment metadata",
            "segment_kind": "active, rest_merged, or rest_baseline for included segments; excluded terminal short rest_postgaze is in metadata",
            "window_start_seconds": "Absolute EDF-relative start of a 200-Hz window",
            "window_offset_seconds": "Offset from the inward-rounded continuous segment start, not from its original parent trial",
            "repair_fraction": "Mean original 500-Hz bad mask over all 32 channels and source sample cells overlapping the window: [floor(start200*2.5),ceil((start200+400)*2.5)). All such samples were repaired; does not measure temporal propagation through filtering.",
            "partitioning": "No train/validation/test split is assigned",
            "source_numbering": "Anonymous subject identifiers S001-S012 in ascending order"},
        "total_csv_trials": sum(r["csv_trials"] for r in reports),
        "windows": len(arrays["y"]), "label_counts_windows": dict(Counter(str(v) for v in arrays["y"])),
        "segments": sum(r["boundary_qa"]["total_segments"] for r in reports),
        "eligible_segments": sum(r["boundary_qa"]["eligible_segments"] for r in reports),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()}, "subject_reports": reports}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "ssvep_windows.npz").open("wb" if overwrite else "xb") as stream:
        np.savez_compressed(stream, **arrays)
    metadata["cache_sha256"] = sha256_file(output / "ssvep_windows.npz")
    with (output / "metadata.json").open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=True, indent=2)
    return metadata
