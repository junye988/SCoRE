"""Prepare SSVEP-EEG from released anonymous EDF/CSV files or window arrays."""
import argparse
from pathlib import Path
import tempfile
from .self_collected import convert_windows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-format", choices=("raw", "windows"), default="raw")
    args = parser.parse_args(argv)
    if args.source_format == "windows":
        return convert_windows(args.source, args.output, "SSVEP-EEG")
    from ._ssvep_raw import build_cache
    with tempfile.TemporaryDirectory(prefix="score-ssvep-") as work:
        cache = Path(work) / "windows"
        build_cache(args.source, cache)
        return convert_windows(cache, args.output, "SSVEP-EEG")


if __name__ == "__main__":
    main()
