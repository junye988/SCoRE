"""Download and safely install the prepared BCI-IV-2a demo dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile


SOURCE_CONFIG = Path(__file__).resolve().parent / "configs" / "demo_data.json"
SPLITS = ("train", "val", "test")


def cache_problem(root, splits=SPLITS):
    """Return a presence/manifest error, leaving array validation to the runner."""
    root = Path(root)
    try:
        with (root / "manifest.json").open(encoding="utf-8-sig") as stream:
            manifest = json.load(stream)
        if not isinstance(manifest, dict):
            return "manifest.json must contain an object"
        if manifest.get("format") != "score_eeg_v1" or manifest.get("status") != "complete":
            return "manifest.json does not describe a complete prepared cache"
        if manifest.get("dataset") not in ("BCI-IV-2a", "bci_iv_2a"):
            return "manifest.json does not describe BCI-IV-2a"
        for split in splits:
            required = [root / f"{split}_{suffix}.npy"
                        for suffix in ("samples", "labels", "group_indices")]
            pairs = root / f"{split}_alignment_pairs.npy"
            required.append(pairs if pairs.exists() else root / "alignment_pairs.npy")
            for path in required:
                if not path.is_file() or path.stat().st_size == 0:
                    return f"missing or empty {path.name}"
    except (OSError, UnicodeError, ValueError) as error:
        return f"cannot read prepared cache: {error}"
    return None


def _require_empty_destination(root):
    if root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        problem = cache_problem(root)
        raise FileExistsError(
            f"Cannot install BCI-IV-2a into {root}: {problem or 'destination is occupied'}. "
            "Choose an empty --data-dir or repair the existing prepared cache. "
            "Existing files have been left in place."
        )


def _read_source(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        source = json.load(stream)["bci_iv_2a"]
    if urlsplit(source["url"]).scheme != "https":
        raise ValueError("The demo dataset source must use HTTPS")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", source["sha256"]):
        raise ValueError("The demo dataset source must specify a SHA-256 digest")
    if type(source["size_bytes"]) is not int or source["size_bytes"] <= 0:
        raise ValueError("The demo dataset source must specify a positive size_bytes")
    return source


def _download(source, target):
    expected = source["size_bytes"]
    host = urlsplit(source["url"]).netloc
    print(f"Downloading prepared BCI-IV-2a ({expected / 2**20:.1f} MiB) from {host}", flush=True)
    request = Request(source["url"], headers={"User-Agent": "SCoRE-demo/1.0"})
    digest = hashlib.sha256()
    received = 0
    last_update = time.monotonic()
    with urlopen(request, timeout=60) as response, target.open("xb") as output:
        if urlsplit(response.geturl()).scheme != "https":
            raise ValueError("The demo dataset download redirected away from HTTPS")
        length = response.headers.get("Content-Length")
        if length is not None and int(length) != expected:
            raise ValueError(f"Download size differs from the configured {expected} bytes")
        while block := response.read(1 << 20):
            received += len(block)
            if received > expected:
                raise ValueError("Download exceeds the configured archive size")
            output.write(block)
            digest.update(block)
            now = time.monotonic()
            if now - last_update >= 5 or received == expected:
                print(f"  {received / 2**20:.1f} / {expected / 2**20:.1f} MiB "
                      f"({100 * received / expected:.1f}%)", flush=True)
                last_update = now
    if received != expected:
        raise ValueError(f"Incomplete download: received {received} of {expected} bytes")
    if digest.hexdigest() != source["sha256"].lower():
        raise ValueError("BCI-IV-2a archive SHA-256 mismatch; the download was not installed")
    print("Archive SHA-256 verified. Extracting prepared data...", flush=True)


def _extract_archive(archive, destination):
    """Validate every ZIP member before extracting regular files/directories."""
    with zipfile.ZipFile(archive) as bundle:
        members = []
        seen = set()
        for entry in bundle.infolist():
            name = entry.orig_filename
            path = PurePosixPath(name)
            mode = entry.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if (not name or "\\" in name or "\x00" in name or path.is_absolute()
                    or PureWindowsPath(name).drive or ".." in path.parts
                    or not path.parts or any(":" in part or part.endswith((".", " "))
                                             or re.fullmatch(r"(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", part)
                                             for part in path.parts)
                    or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                    or (kind == stat.S_IFDIR and not entry.is_dir())):
                raise ValueError(f"Unsafe ZIP member: {name!r}")
            key = str(path).casefold()
            if key in seen:
                raise ValueError(f"Duplicate ZIP member: {name!r}")
            seen.add(key)
            target = destination.joinpath(*path.parts)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"ZIP member escapes the extraction directory: {name!r}")
            members.append((entry, target))
        unpacked = sum(entry.file_size for entry, _ in members)
        if unpacked > shutil.disk_usage(destination).free:
            raise OSError(f"Not enough disk space to extract {unpacked / 2**20:.1f} MiB of data")
        for entry, target in members:
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(entry) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)


def ensure_bci_data(root, *, splits=SPLITS, source_config=SOURCE_CONFIG):
    """Reuse a prepared cache or atomically install a verified archive."""
    root = Path(root).absolute()
    problem = cache_problem(root, splits)
    if problem is None:
        print(f"Using prepared BCI-IV-2a data: {root}", flush=True)
        return root
    _require_empty_destination(root)
    source = _read_source(source_config)
    root.parent.mkdir(parents=True, exist_ok=True)
    if source["size_bytes"] > shutil.disk_usage(root.parent).free:
        raise OSError("Not enough disk space to download the BCI-IV-2a archive")
    with tempfile.TemporaryDirectory(prefix=f".{root.name}-download-", dir=root.parent) as work:
        staging = Path(work)
        archive = staging / "dataset.zip"
        _download(source, archive)
        extracted = staging / "extracted"
        extracted.mkdir()
        _extract_archive(archive, extracted)
        prepared = extracted
        if not (prepared / "manifest.json").is_file():
            children = list(extracted.iterdir())
            if len(children) == 1 and children[0].is_dir():
                prepared = children[0]
        problem = cache_problem(prepared)
        if problem is not None:
            raise ValueError(f"Downloaded archive is not a complete prepared BCI-IV-2a cache: {problem}")
        # Recheck after the download so a concurrent run cannot overwrite user files.
        _require_empty_destination(root)
        if root.exists():
            root.rmdir()
        prepared.rename(root)
    print(f"Prepared BCI-IV-2a data installed: {root}", flush=True)
    return root


def prepare_demo_data(args):
    """Resolve the demo data directory without changing the shared experiment CLI."""
    if args.show_config:
        return
    if args.dataset != "bci_iv_2a":
        if args.data_dir is None:
            raise ValueError(
                f"Automatic demo data download is available for BCI-IV-2a only. "
                f"For {args.dataset}, prepare the dataset with python -m preprocessing.{args.dataset} "
                "and supply --data-dir with the resulting directory."
            )
        return
    splits = ("test",) if args.mode == "evaluate" else SPLITS
    args.data_dir = ensure_bci_data(args.data_dir or Path("data") / "bci_iv_2a", splits=splits)
