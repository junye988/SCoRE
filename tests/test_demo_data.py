"""Bounded installer tests using in-memory downloads; no external network."""

import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from experiments import demo_data


def cache_files(splits=demo_data.SPLITS):
    files = {"manifest.json": json.dumps({"format": "score_eeg_v1", "status": "complete",
                                         "dataset": "BCI-IV-2a"}).encode()}
    for split in splits:
        for suffix in ("samples", "labels", "group_indices", "alignment_pairs"):
            files[f"{split}_{suffix}.npy"] = f"fixture {split} {suffix}".encode()
    return files


def make_archive(files, prefix=""):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(prefix + name, content)
    return buffer.getvalue()


class DownloadResponse(io.BytesIO):
    def __init__(self, payload, url="https://downloads.example.test/data.zip"):
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}
        self.url = url

    def geturl(self):
        return self.url


class DemoDataTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.cache = self.root / "bci_iv_2a"
        self.config = self.root / "source.json"

    def source(self, payload, **overrides):
        value = {"url": "https://downloads.example.test/data.zip?private_link=fixture",
                 "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        value.update(overrides)
        self.config.write_text(json.dumps({"bci_iv_2a": value}), encoding="utf-8")

    def write_cache(self, splits=demo_data.SPLITS):
        self.cache.mkdir()
        for name, payload in cache_files(splits).items():
            (self.cache / name).write_bytes(payload)

    def assert_no_staging(self):
        self.assertEqual(list(self.root.glob(".bci_iv_2a-download-*")), [])

    def test_existing_cache_needs_neither_source_config_nor_network(self):
        self.write_cache()
        with patch.object(demo_data, "urlopen", side_effect=AssertionError("network used")):
            result = demo_data.ensure_bci_data(self.cache, source_config=self.config)
        self.assertEqual(result, self.cache)

    def test_missing_cache_downloads_checks_hash_and_installs(self):
        files = cache_files()
        for prefix in ("", "bci_iv_2a/"):
            with self.subTest(prefix=prefix):
                target = self.root / ("flat" if not prefix else "nested")
                payload = make_archive(files, prefix)
                self.source(payload)
                with patch.object(demo_data, "urlopen", return_value=DownloadResponse(payload)) as download:
                    demo_data.ensure_bci_data(target, source_config=self.config)
                download.assert_called_once()
                self.assertEqual(download.call_args.args[0].full_url,
                                 "https://downloads.example.test/data.zip?private_link=fixture")
                self.assertEqual({path.name: path.read_bytes() for path in target.iterdir()}, files)
                self.assertFalse(list(self.root.glob(f".{target.name}-download-*")))

    def test_hash_mismatch_does_not_install_and_cleans_staging(self):
        payload = make_archive(cache_files())
        self.source(payload, sha256="0" * 64)
        with patch.object(demo_data, "urlopen", return_value=DownloadResponse(payload)):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                demo_data.ensure_bci_data(self.cache, source_config=self.config)
        self.assertFalse(self.cache.exists())
        self.assert_no_staging()

    def test_unsafe_zip_paths_and_symlinks_are_rejected(self):
        for name in ("../escaped.txt", "/absolute.txt", "C:/escaped.txt", "..\\escaped.txt", "CON.txt"):
            with self.subTest(name=name):
                files = cache_files()
                files[name] = b"unsafe"
                payload = make_archive(files)
                self.source(payload)
                with patch.object(demo_data, "urlopen", return_value=DownloadResponse(payload)):
                    with self.assertRaisesRegex(ValueError, "Unsafe ZIP member"):
                        demo_data.ensure_bci_data(self.cache, source_config=self.config)
                self.assertFalse(self.cache.exists())
                self.assertFalse((self.root / "escaped.txt").exists())
                self.assert_no_staging()
        payload = io.BytesIO(make_archive(cache_files()))
        with zipfile.ZipFile(payload, "a") as archive:
            link = zipfile.ZipInfo("link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(link, "../escaped.txt")
        self.source(payload.getvalue())
        with patch.object(demo_data, "urlopen", return_value=DownloadResponse(payload.getvalue())):
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP member"):
                demo_data.ensure_bci_data(self.cache, source_config=self.config)
        self.assertFalse(self.cache.exists())
        self.assert_no_staging()

    def test_partial_download_and_non_https_redirect_are_rejected(self):
        payload = make_archive(cache_files())
        self.source(payload)
        for response in (DownloadResponse(payload[:-1]), DownloadResponse(payload, "http://example.test/data.zip")):
            with self.subTest(url=response.geturl(), length=len(response.getvalue())):
                with patch.object(demo_data, "urlopen", return_value=response):
                    with self.assertRaises(ValueError):
                        demo_data.ensure_bci_data(self.cache, source_config=self.config)
                self.assertFalse(self.cache.exists())
                self.assert_no_staging()

    def test_invalid_nonempty_directory_is_preserved_without_network(self):
        self.cache.mkdir()
        original = self.cache / "notes.txt"
        original.write_text("keep this", encoding="utf-8")
        with patch.object(demo_data, "urlopen", side_effect=AssertionError("network used")):
            with self.assertRaisesRegex(FileExistsError, "Choose an empty --data-dir"):
                demo_data.ensure_bci_data(self.cache, source_config=self.config)
        self.assertEqual(original.read_text(encoding="utf-8"), "keep this")

    def test_evaluate_reuses_test_split_cache(self):
        self.write_cache(splits=("test",))
        args = SimpleNamespace(dataset="bci_iv_2a", data_dir=self.cache, mode="evaluate", show_config=False)
        with patch.object(demo_data, "urlopen", side_effect=AssertionError("network used")):
            demo_data.prepare_demo_data(args)
        self.assertEqual(args.data_dir, self.cache)

    def test_show_config_and_other_datasets_do_not_download(self):
        with patch.object(demo_data, "ensure_bci_data", side_effect=AssertionError("installer called")):
            demo_data.prepare_demo_data(SimpleNamespace(show_config=True))
            args = SimpleNamespace(dataset="hmc", data_dir=None, mode="train", show_config=False)
            with self.assertRaisesRegex(ValueError, "supply --data-dir"):
                demo_data.prepare_demo_data(args)
            args.data_dir = self.root / "hmc"
            demo_data.prepare_demo_data(args)
            self.assertEqual(args.data_dir, self.root / "hmc")


if __name__ == "__main__":
    unittest.main()
