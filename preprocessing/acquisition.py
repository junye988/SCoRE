"""Locate prepared EEG caches and acquire the configured BCI-IV-2a release."""
from __future__ import annotations

from pathlib import Path

from utils.config import load_yaml
from utils.download import install_archive
from utils.io import read_json
from .cache import CACHE_FORMAT, SPLITS


SOURCE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "data_sources.yaml"


def cache_problem(root, splits=SPLITS):
    """Return a BCI cache presence/schema error; arrays are checked when read."""
    root = Path(root)
    try:
        manifest = read_json(root / "manifest.json")
        if not isinstance(manifest, dict):
            return "manifest.json must contain an object"
        if manifest.get("format") != CACHE_FORMAT or manifest.get("status") != "complete":
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


def ensure_bci_data(root, *, splits=SPLITS, source_config=SOURCE_CONFIG):
    """Reuse a prepared cache or atomically install the complete verified release."""
    root = Path(root).absolute()
    problem = cache_problem(root, splits)
    if problem is None:
        print(f"Using prepared BCI-IV-2a data: {root}", flush=True)
        return root
    _require_empty_destination(root)
    source = load_yaml(source_config)["bci_iv_2a"]
    install_archive(source, root, cache_problem)
    print(f"Prepared BCI-IV-2a data installed: {root}", flush=True)
    return root


def resolve_data_dir(dataset, data_dir=None, *, splits=SPLITS, source_config=SOURCE_CONFIG):
    """Resolve a prepared-data location using the available acquisition policy."""
    if dataset == "bci_iv_2a":
        return ensure_bci_data(data_dir or Path("data") / "bci_iv_2a", splits=splits,
                               source_config=source_config)
    if data_dir is None:
        raise ValueError(
            "Automatic demo data download is available for BCI-IV-2a only. "
            f"For {dataset}, prepare the dataset with python -m preprocessing.{dataset} "
            "and supply --data-dir with the resulting directory."
        )
    return Path(data_dir)
