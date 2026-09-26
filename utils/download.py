"""Verified HTTPS downloads and safe, atomic ZIP installation."""
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import stat
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import zipfile


def _validate_source(source):
    if urlsplit(source["url"]).scheme != "https":
        raise ValueError("The archive source must use HTTPS")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", source["sha256"]):
        raise ValueError("The archive source must specify a SHA-256 digest")
    if type(source["size_bytes"]) is not int or source["size_bytes"] <= 0:
        raise ValueError("The archive source must specify a positive size_bytes")


def _require_empty_destination(root):
    if root.is_symlink() or (root.exists() and (not root.is_dir() or any(root.iterdir()))):
        raise FileExistsError(f"Cannot install archive into occupied destination: {root}")


def _download(source, target):
    expected = source["size_bytes"]
    host = urlsplit(source["url"]).netloc
    print(f"Downloading archive ({expected / 2**20:.1f} MiB) from {host}", flush=True)
    request = Request(source["url"], headers={"User-Agent": "SCoRE/1.0"})
    digest = hashlib.sha256()
    received = 0
    last_update = time.monotonic()
    with urlopen(request, timeout=60) as response, target.open("xb") as output:
        if urlsplit(response.geturl()).scheme != "https":
            raise ValueError("The archive download redirected away from HTTPS")
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
        raise ValueError("Archive SHA-256 mismatch; the download was not installed")
    print("Archive SHA-256 verified. Extracting...", flush=True)


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


def install_archive(source, destination, validate):
    """Install a verified ZIP whose root passes ``validate(path)``.

    The callback returns ``None`` for valid contents or an explanatory string.
    A single enclosing directory is accepted. The destination must be absent or
    empty, and it is rechecked before the prepared directory is renamed into it.
    """
    root = Path(destination).absolute()
    _require_empty_destination(root)
    _validate_source(source)
    root.parent.mkdir(parents=True, exist_ok=True)
    if source["size_bytes"] > shutil.disk_usage(root.parent).free:
        raise OSError("Not enough disk space to download the archive")
    with tempfile.TemporaryDirectory(prefix=f".{root.name}-download-", dir=root.parent) as work:
        staging = Path(work)
        archive = staging / "archive.zip"
        _download(source, archive)
        extracted = staging / "extracted"
        extracted.mkdir()
        _extract_archive(archive, extracted)
        prepared = extracted
        problem = validate(prepared)
        if problem is not None:
            children = list(extracted.iterdir())
            if len(children) == 1 and children[0].is_dir():
                prepared = children[0]
                problem = validate(prepared)
        if problem is not None:
            raise ValueError(f"Downloaded archive failed validation: {problem}")
        # A concurrent process may have created destination files during download.
        _require_empty_destination(root)
        if root.exists():
            root.rmdir()
        prepared.rename(root)
    return root
