"""AesthEEG recording filters, event alignment and stimulus epochs."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd


IMAGE_START_MARKER = "20"
IMAGE_START_MARKER_CANDIDATES = ("20", "4")
DEFAULT_CHANNEL_SUFFIX = "-CPz"


@dataclass
class SubjectSummary:
    subject: str
    split: str
    directory: str
    status: str
    reason: str
    n_trials: int = 0
    n_events: int = 0
    n_epochs_saved: int = 0
    n_ugly: int = 0
    n_beautiful: int = 0
    n_neutral_dropped: int = 0
    n_bad_epochs_dropped: int = 0
    event_marker_used: str = ""
    label_source: str = ""
    output_file: str = ""








def find_subject_dir(subjects_dir: Path, subject: str) -> Path:
    matches = sorted(subjects_dir.glob(f"{subject}_*"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one directory for subject {subject}, found {len(matches)}.")
    return matches[0]


def segment_number(path: Path) -> int:
    match = re.search(r"_Segment_(\d+)\.edf$", path.name)
    return int(match.group(1)) if match else 0


def clean_channel_name(name: str) -> str:
    cleaned = name.strip()
    if cleaned.startswith("EEG "):
        cleaned = cleaned[4:]
    if cleaned.endswith(DEFAULT_CHANNEL_SUFFIX):
        cleaned = cleaned[: -len(DEFAULT_CHANNEL_SUFFIX)]
    return cleaned


def read_trials(subject_dir: Path, subject: str) -> pd.DataFrame:
    csv_path = subject_dir / f"AestheticEEG_{subject}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    trials = pd.read_csv(csv_path)
    trials["subject"] = subject
    trials["trial_index"] = np.arange(len(trials), dtype=int)
    scores = pd.to_numeric(trials["Score"], errors="coerce")
    labels = pd.Series(np.nan, index=trials.index, dtype="float")
    labels.loc[scores <= 3] = 0
    labels.loc[scores >= 5] = 1
    trials["score_numeric"] = scores
    trials["label"] = labels
    return trials


def labels_from_scores(scores: pd.Series | list[int | float | None]) -> pd.Series:
    numeric = pd.to_numeric(pd.Series(scores), errors="coerce")
    labels = pd.Series(np.nan, index=numeric.index, dtype="float")
    labels.loc[numeric <= 3] = 0
    labels.loc[numeric >= 5] = 1
    return labels


def read_raw_subject(subject_dir: Path) -> mne.io.BaseRaw:
    edf_paths = sorted(subject_dir.glob("*.edf"), key=segment_number)
    if not edf_paths:
        raise FileNotFoundError(f"No EDF files found in {subject_dir}")

    raws = [mne.io.read_raw_edf(path, preload=True, verbose="ERROR") for path in edf_paths]
    raw = raws[0] if len(raws) == 1 else mne.concatenate_raws(raws, verbose="ERROR")

    rename = {name: clean_channel_name(name) for name in raw.ch_names}
    raw.rename_channels(rename)
    raw.set_channel_types({name: "eeg" for name in raw.ch_names}, verbose="ERROR")
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="warn", verbose="ERROR")
    return raw


def preprocess_raw(raw: mne.io.BaseRaw, args: argparse.Namespace) -> mne.io.BaseRaw:
    raw.pick("eeg", exclude=[])
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    if args.notch_freq:
        raw.notch_filter(freqs=[args.notch_freq], verbose="ERROR")
    raw.filter(l_freq=args.l_freq, h_freq=args.h_freq, verbose="ERROR")
    if args.resample_sfreq:
        raw.resample(args.resample_sfreq, npad="auto", verbose="ERROR")
    return raw


def image_start_events(raw: mne.io.BaseRaw, expected_count: int) -> tuple[np.ndarray, np.ndarray, str]:
    descriptions = np.array([str(desc).strip() for desc in raw.annotations.description])
    counts = {candidate: int((descriptions == candidate).sum()) for candidate in IMAGE_START_MARKER_CANDIDATES}
    exact = [candidate for candidate, count in counts.items() if count == expected_count]
    marker = exact[0] if exact else min(IMAGE_START_MARKER_CANDIDATES, key=lambda candidate: abs(counts[candidate] - expected_count))
    mask = descriptions == marker
    onsets = raw.annotations.onset[mask]
    samples = raw.time_as_index(onsets, use_rounding=True)
    events = np.column_stack(
        [
            samples.astype(int),
            np.zeros(len(samples), dtype=int),
            np.full(len(samples), int(marker), dtype=int),
        ]
    )
    return events, onsets, marker


def response_scores_for_events(raw: mne.io.BaseRaw, image_onsets: np.ndarray) -> list[int | None]:
    descriptions = np.array([str(desc).strip() for desc in raw.annotations.description])
    response_codes = [str(code) for code in range(61, 68)]
    response_mask = np.isin(descriptions, response_codes)
    response_onsets = np.asarray(raw.annotations.onset)[response_mask]
    response_descriptions = descriptions[response_mask]

    scores: list[int | None] = []
    for idx, onset in enumerate(image_onsets):
        next_onset = image_onsets[idx + 1] if idx + 1 < len(image_onsets) else onset + 30.0
        response_idx = np.where((response_onsets > onset) & (response_onsets < next_onset))[0]
        if len(response_idx) == 0:
            scores.append(None)
        else:
            scores.append(int(response_descriptions[response_idx[0]]) - 60)
    return scores


def align_event_scores_to_trials(event_scores: list[int | None], trial_scores: list[int]) -> list[int | None]:
    """Map event scores to CSV trial indices while allowing dropped/extra markers."""
    n_events = len(event_scores)
    n_trials = len(trial_scores)
    dp = np.zeros((n_events + 1, n_trials + 1), dtype=np.int16)

    for i in range(n_events - 1, -1, -1):
        for j in range(n_trials - 1, -1, -1):
            if event_scores[i] is not None and event_scores[i] == trial_scores[j]:
                dp[i, j] = dp[i + 1, j + 1] + 1
            else:
                dp[i, j] = max(dp[i + 1, j], dp[i, j + 1])

    mapping: list[int | None] = [None] * n_events
    i = j = 0
    while i < n_events and j < n_trials:
        if event_scores[i] is not None and event_scores[i] == trial_scores[j] and dp[i, j] == dp[i + 1, j + 1] + 1:
            mapping[i] = j
            i += 1
            j += 1
        elif dp[i + 1, j] >= dp[i, j + 1]:
            i += 1
        else:
            j += 1
    return mapping


def event_level_trials(trials: pd.DataFrame, event_scores: list[int | None], label_source: str) -> pd.DataFrame:
    if label_source == "csv":
        out = trials.reset_index(drop=True).copy()
        out["event_index"] = np.arange(len(out), dtype=int)
        out["event_score"] = out["score_numeric"]
        out["label"] = labels_from_scores(out["score_numeric"]).to_numpy()
        return out

    trial_scores = pd.to_numeric(trials["Score"], errors="coerce").fillna(-999).astype(int).tolist()
    mapping = align_event_scores_to_trials(event_scores, trial_scores)
    rows: list[pd.Series] = []
    for event_idx, trial_idx in enumerate(mapping):
        if trial_idx is None:
            row = pd.Series({column: pd.NA for column in trials.columns})
            row["subject"] = trials["subject"].iloc[0]
        else:
            row = trials.iloc[trial_idx].copy()
        row["event_index"] = event_idx
        row["event_score"] = event_scores[event_idx]
        rows.append(row)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out["label"] = labels_from_scores(out["event_score"]).to_numpy()
    return out


def save_subject(
    subject: str,
    split_name: str,
    subject_dir: Path,
    args: argparse.Namespace,
) -> tuple[SubjectSummary, pd.DataFrame | None]:
    output_path = args.out_dir / "subjects" / f"sub-{subject}.npz"
    if output_path.exists() and not args.overwrite:
        return (
            SubjectSummary(
                subject=subject,
                split=split_name,
                directory=subject_dir.name,
                status="skipped",
                reason="output exists; pass --overwrite to regenerate",
                output_file=str(output_path),
            ),
            None,
        )

    trials = read_trials(subject_dir, subject)
    keep_mask = trials["label"].notna().to_numpy()
    n_neutral = int((~keep_mask).sum())

    raw = preprocess_raw(read_raw_subject(subject_dir), args)
    events, event_onsets, event_marker_used = image_start_events(raw, expected_count=len(trials))
    event_trial_diff = abs(len(events) - len(trials))
    if event_trial_diff > args.max_event_trial_diff:
        return (
            SubjectSummary(
                subject=subject,
                split=split_name,
                directory=subject_dir.name,
                status="failed",
                reason=f"image_start events ({len(events)}) != CSV trials ({len(trials)})",
                n_trials=len(trials),
                n_events=len(events),
                n_neutral_dropped=n_neutral,
                event_marker_used=event_marker_used,
            ),
            None,
        )

    label_source = "csv" if len(events) == len(trials) else "edf_response"
    event_scores = response_scores_for_events(raw, event_onsets) if label_source == "edf_response" else []
    aligned_trials = event_level_trials(trials, event_scores, label_source)
    keep_mask = aligned_trials["label"].notna().to_numpy()
    n_neutral = int((~keep_mask).sum())
    kept_trials = aligned_trials.loc[keep_mask].reset_index(drop=True)
    kept_events = events[keep_mask]
    kept_onsets = event_onsets[keep_mask]

    reject = None
    if args.reject_eeg_uv is not None:
        reject = {"eeg": args.reject_eeg_uv * 1e-6}

    epochs = mne.Epochs(
        raw,
        kept_events,
        event_id={"image_start": int(event_marker_used)},
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=(args.baseline_start, args.baseline_end),
        picks="eeg",
        preload=True,
        reject=reject,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    kept_indices = epochs.selection
    saved_trials = kept_trials.iloc[kept_indices].reset_index(drop=True)
    saved_onsets = kept_onsets[kept_indices]
    labels = saved_trials["label"].astype(np.int64).to_numpy()

    data = epochs.get_data(copy=True).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=data,
        y=labels,
        subject=np.array(subject),
        split=np.array(split_name),
        channels=np.array(epochs.ch_names),
        times=epochs.times.astype(np.float32),
        sfreq=np.array(float(epochs.info["sfreq"]), dtype=np.float32),
        presentation_order=pd.to_numeric(saved_trials["PresentationOrder"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64),
        category=saved_trials["Category"].astype(str).to_numpy(),
        filename=saved_trials["Filename"].astype(str).to_numpy(),
        score=pd.to_numeric(saved_trials["score_numeric"], errors="coerce").to_numpy(dtype=np.float32),
        event_score=pd.to_numeric(saved_trials["event_score"], errors="coerce").to_numpy(dtype=np.float32),
        rt_ms=pd.to_numeric(saved_trials["RT"], errors="coerce").to_numpy(dtype=np.float32),
        event_onset_sec=saved_onsets.astype(np.float64),
        label_source=np.array(label_source),
    )

    epoch_manifest = saved_trials.reindex(
        columns=[
            "subject",
            "trial_index",
            "event_index",
            "PresentationOrder",
            "Block",
            "Trial",
            "Category",
            "Filename",
            "score_numeric",
            "event_score",
            "RT",
        ]
    ).copy()
    epoch_manifest.insert(1, "split", split_name)
    epoch_manifest.insert(2, "epoch_index", np.arange(len(epoch_manifest), dtype=int))
    epoch_manifest["label"] = labels
    epoch_manifest["label_name"] = np.where(labels == 1, "beautiful", "ugly")
    epoch_manifest["event_onset_sec"] = saved_onsets
    epoch_manifest["label_source"] = label_source
    epoch_manifest["npz_file"] = str(output_path)

    summary = SubjectSummary(
        subject=subject,
        split=split_name,
        directory=subject_dir.name,
        status="ok",
        reason="" if label_source == "csv" else f"event/trial count differed by {event_trial_diff}; labels used EDF response triggers",
        n_trials=len(trials),
        n_events=len(events),
        n_epochs_saved=len(labels),
        n_ugly=int((labels == 0).sum()),
        n_beautiful=int((labels == 1).sum()),
        n_neutral_dropped=n_neutral,
        n_bad_epochs_dropped=int(len(kept_trials) - len(labels)),
        event_marker_used=event_marker_used,
        label_source=label_source,
        output_file=str(output_path),
    )
    return summary, epoch_manifest
