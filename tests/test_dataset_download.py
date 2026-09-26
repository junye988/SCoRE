"""Bounded installer tests using in-memory downloads; no external network."""

import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import yaml

from preprocessing import dataset_download
from preprocessing.dataset_cache import PreparedDatasetSplit, SPLITS
from utils import download_archive


def cache_files(splits=SPLITS, *, aligned=True):
    manifest = {"format": "score_eeg_v1", "status": "complete", "dataset": "BCI-IV-2a",
                "samples_aligned": aligned, "channels": ["C3", "C4"], "num_channels": 2,
                "sample_rate": 200, "class_names": ["left hand", "right hand", "feet", "tongue"],
                "num_classes": 4, "num_samples": 3, "mirror_permutation": [1, 0]}
    files = {"manifest.json": json.dumps(manifest).encode()}
    arrays = {"samples": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
              "labels": np.array([0, 1], dtype=np.int64),
              "group_indices": np.array([0, 0], dtype=np.int64),
              "alignment_pairs": np.array([[[[2., 0.], [0., .5]], [[.5, 0.], [0., 2.]]]])}
    for split in splits:
        for suffix, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array)
            files[f"{split}_{suffix}.npy"] = buffer.getvalue()
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


class DatasetDownloadTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.cache = self.root / "bci_iv_2a"
        self.config = self.root / "source.yaml"

    def source(self, payload, **overrides):
        value = {"url": "https://downloads.example.test/data.zip?private_link=fixture",
                 "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        value.update(overrides)
        self.config.write_text(yaml.safe_dump({"bci_iv_2a": value}), encoding="utf-8")

    def write_cache(self, splits=SPLITS):
        self.cache.mkdir()
        for name, payload in cache_files(splits).items():
            (self.cache / name).write_bytes(payload)

    def assert_no_staging(self):
        self.assertEqual(list(self.root.glob(".bci_iv_2a-download-*")), [])

    def test_existing_cache_needs_neither_source_config_nor_network(self):
        self.write_cache()
        with patch.object(download_archive, "urlopen", side_effect=AssertionError("network used")):
            result = dataset_download.ensure_bci_iv_2a_data(self.cache, source_config=self.config)
        self.assertEqual(result, self.cache)

    def test_existing_cache_accepts_a_utf8_bom_manifest(self):
        self.write_cache()
        manifest = self.cache / "manifest.json"
        manifest.write_bytes(b"\xef\xbb\xbf" + manifest.read_bytes())
        with patch.object(download_archive, "urlopen", side_effect=AssertionError("network used")):
            self.assertEqual(dataset_download.ensure_bci_iv_2a_data(self.cache), self.cache)
        self.assertEqual(len(PreparedDatasetSplit(self.cache, "test")), 2)

    def test_missing_cache_downloads_checks_hash_and_installs(self):
        files = cache_files()
        for prefix in ("", "bci_iv_2a/"):
            with self.subTest(prefix=prefix):
                target = self.root / ("flat" if not prefix else "nested")
                payload = make_archive(files, prefix)
                self.source(payload)
                with patch.object(download_archive, "urlopen", return_value=DownloadResponse(payload)) as request:
                    dataset_download.ensure_bci_iv_2a_data(target, source_config=self.config)
                request.assert_called_once()
                self.assertEqual(request.call_args.args[0].full_url,
                                 "https://downloads.example.test/data.zip?private_link=fixture")
                self.assertEqual({path.name: path.read_bytes() for path in target.iterdir()}, files)
                self.assertFalse(list(self.root.glob(f".{target.name}-download-*")))

    def test_hash_mismatch_does_not_install_and_cleans_staging(self):
        payload = make_archive(cache_files())
        self.source(payload, sha256="0" * 64)
        with patch.object(download_archive, "urlopen", return_value=DownloadResponse(payload)):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                dataset_download.ensure_bci_iv_2a_data(self.cache, source_config=self.config)
        self.assertFalse(self.cache.exists())
        self.assert_no_staging()

    def test_unsafe_zip_paths_and_symlinks_are_rejected(self):
        for name in ("../escaped.txt", "/absolute.txt", "C:/escaped.txt", "..\\escaped.txt", "CON.txt"):
            with self.subTest(name=name):
                files = cache_files()
                files[name] = b"unsafe"
                payload = make_archive(files)
                self.source(payload)
                with patch.object(download_archive, "urlopen", return_value=DownloadResponse(payload)):
                    with self.assertRaisesRegex(ValueError, "Unsafe ZIP member"):
                        dataset_download.ensure_bci_iv_2a_data(self.cache, source_config=self.config)
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
        with patch.object(download_archive, "urlopen", return_value=DownloadResponse(payload.getvalue())):
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP member"):
                dataset_download.ensure_bci_iv_2a_data(self.cache, source_config=self.config)
        self.assertFalse(self.cache.exists())
        self.assert_no_staging()

    def test_partial_download_and_non_https_redirect_are_rejected(self):
        payload = make_archive(cache_files())
        self.source(payload)
        for response in (DownloadResponse(payload[:-1]), DownloadResponse(payload, "http://example.test/data.zip")):
            with self.subTest(url=response.geturl(), length=len(response.getvalue())):
                with patch.object(download_archive, "urlopen", return_value=response):
                    with self.assertRaises(ValueError):
                        dataset_download.ensure_bci_iv_2a_data(self.cache, source_config=self.config)
                self.assertFalse(self.cache.exists())
                self.assert_no_staging()

    def test_invalid_nonempty_directory_is_preserved_without_network(self):
        self.cache.mkdir()
        original = self.cache / "notes.txt"
        original.write_text("keep this", encoding="utf-8")
        with patch.object(download_archive, "urlopen", side_effect=AssertionError("network used")):
            with self.assertRaisesRegex(FileExistsError, "Choose an empty --data-dir"):
                dataset_download.ensure_bci_iv_2a_data(self.cache, source_config=self.config)
        self.assertEqual(original.read_text(encoding="utf-8"), "keep this")

    def test_evaluate_reuses_test_split_cache(self):
        self.write_cache(splits=("test",))
        with patch.object(download_archive, "urlopen", side_effect=AssertionError("network used")):
            result = dataset_download.resolve_dataset_directory("bci_iv_2a", self.cache, splits=("test",))
        self.assertEqual(result, self.cache)

    def test_other_datasets_require_explicit_data_directory(self):
        with patch.object(dataset_download, "ensure_bci_iv_2a_data", side_effect=AssertionError("installer called")):
            with self.assertRaisesRegex(ValueError, "supply --data-dir"):
                dataset_download.resolve_dataset_directory("hmc")
            self.assertEqual(dataset_download.resolve_dataset_directory("hmc", self.root / "hmc"), self.root / "hmc")

    def test_download_requires_complete_cache_even_when_only_test_is_requested(self):
        payload = make_archive(cache_files(splits=("test",)))
        self.source(payload)
        with patch.object(download_archive, "urlopen", return_value=DownloadResponse(payload)):
            with self.assertRaisesRegex(ValueError, "missing or empty train_samples"):
                dataset_download.ensure_bci_iv_2a_data(self.cache, splits=("test",), source_config=self.config)
        self.assertFalse(self.cache.exists())
        self.assert_no_staging()

    def test_rejected_cache_schema_does_not_install(self):
        files = cache_files()
        manifest = json.loads(files["manifest.json"])
        manifest["dataset"] = "HMC"
        files["manifest.json"] = json.dumps(manifest).encode()
        payload = make_archive(files)
        self.source(payload)
        with patch.object(download_archive, "urlopen", return_value=DownloadResponse(payload)):
            with self.assertRaisesRegex(ValueError, "does not describe BCI-IV-2a"):
                dataset_download.ensure_bci_iv_2a_data(self.cache, source_config=self.config)
        self.assertFalse(self.cache.exists())
        self.assert_no_staging()

    def test_destination_created_during_download_is_preserved(self):
        payload = make_archive(cache_files())
        self.source(payload)
        def response(*args, **kwargs):
            self.cache.mkdir()
            (self.cache / "notes.txt").write_text("keep this", encoding="utf-8")
            return DownloadResponse(payload)
        with patch.object(download_archive, "urlopen", side_effect=response):
            with self.assertRaises(FileExistsError):
                dataset_download.ensure_bci_iv_2a_data(self.cache, source_config=self.config)
        self.assertEqual((self.cache / "notes.txt").read_text(encoding="utf-8"), "keep this")
        self.assert_no_staging()

    def test_installed_cache_reads_alignment_and_detaches_returned_arrays(self):
        payload = make_archive(cache_files(aligned=False))
        self.source(payload)
        with patch.object(download_archive, "urlopen", return_value=DownloadResponse(payload)):
            dataset_download.ensure_bci_iv_2a_data(self.cache, source_config=self.config)
        prepared = PreparedDatasetSplit(self.cache, "train")
        self.assertEqual(len(prepared), 2)
        record = prepared.read(0)
        np.testing.assert_array_equal(record["sample"], [[0., 2., 4.], [1.5, 2., 2.5]])
        self.assertEqual(record["sample"].dtype, np.float64)
        self.assertEqual((record["label"], record["index"]), (0, 0))
        record["sample"][:] = -1
        record["alignment_pair"][:] = -1
        np.testing.assert_array_equal(prepared.samples[0], [[0., 1., 2.], [3., 4., 5.]])
        np.testing.assert_array_equal(prepared.read(0)["alignment_pair"][0], [[2., 0.], [0., .5]])

    def test_aligned_cache_retains_dtype_and_shared_pair_fallback(self):
        files = cache_files(splits=("test",))
        files["alignment_pairs.npy"] = files.pop("test_alignment_pairs.npy")
        self.cache.mkdir()
        for name, value in files.items():
            (self.cache / name).write_bytes(value)
        self.assertIsNone(dataset_download.check_bci_iv_2a_files(self.cache, splits=("test",)))
        record = PreparedDatasetSplit(self.cache, "test").read(1)
        self.assertEqual(record["sample"].dtype, np.float32)
        np.testing.assert_array_equal(record["sample"], [[6., 7., 8.], [9., 10., 11.]])


class ArchiveTests(unittest.TestCase):
    def test_installer_accepts_generic_contents_with_a_validation_callback(self):
        payload = make_archive({"hello.txt": b"hello"}, prefix="release/")
        source = {"url": "https://downloads.example.test/release.zip",
                  "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        def validate(path):
            return None if (path / "hello.txt").is_file() else "hello.txt is missing"
        with tempfile.TemporaryDirectory() as work:
            target = Path(work) / "installed"
            with patch.object(download_archive, "urlopen", return_value=DownloadResponse(payload)):
                self.assertEqual(download_archive.download_and_extract_archive(source, target, validate), target)
            self.assertEqual((target / "hello.txt").read_bytes(), b"hello")

    def test_invalid_source_metadata_is_rejected_before_network(self):
        for overrides in ({"url": "http://example.test/data.zip"}, {"sha256": "invalid"},
                          {"size_bytes": 0}, {"size_bytes": True}):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as work:
                source = {"url": "https://example.test/data.zip", "sha256": "0" * 64, "size_bytes": 10}
                source.update(overrides)
                with patch.object(download_archive, "urlopen", side_effect=AssertionError("network used")):
                    with self.assertRaises(ValueError):
                        download_archive.download_and_extract_archive(source, Path(work) / "installed", lambda path: None)


if __name__ == "__main__":
    unittest.main()
