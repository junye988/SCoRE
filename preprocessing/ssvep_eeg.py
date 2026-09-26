"""Prepare SSVEP-EEG from released anonymous EDF/CSV files or window arrays."""
import argparse
from pathlib import Path
import tempfile
from .window_import import import_window_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-format", choices=("raw", "windows"), default="raw")
    args = parser.parse_args(argv)
    if args.source_format == "windows":
        return import_window_archive(args.source, args.output, "SSVEP-EEG")
    from .raw_readers.ssvep_eeg import build_window_archive
    with tempfile.TemporaryDirectory(prefix="score-ssvep-") as work:
        cache = Path(work) / "windows"
        build_window_archive(args.source, cache)
        return import_window_archive(cache, args.output, "SSVEP-EEG")


if __name__ == "__main__":
    main()
