"""Project dependency direction and portable YAML configuration contracts."""

import ast
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import yaml

from experiments.config import CONFIG_ROOT, DATASETS, load_config, member_config
from model import EnsembleConfig, ModelConfig
from utils.config import format_yaml, load_yaml, save_yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_DATASETS = {
    "aestheeg", "bci_iv_2a", "handmi", "hmc", "isruc", "mat",
    "mumtaz", "physionet_mi", "ssvep_eeg",
}


class ProjectBoundaryTests(unittest.TestCase):
    def test_project_imports_follow_dependency_direction(self):
        allowed = {
            "model": {"model"},
            "utils": {"utils"},
            "preprocessing": {"preprocessing", "utils"},
            "experiments": {"experiments", "preprocessing", "model", "utils"},
        }
        project_packages = set(allowed)
        for package, dependencies in allowed.items():
            sources = sorted((ROOT / package).rglob("*.py"))
            self.assertTrue(sources, package)
            for path in sources:
                relative = path.relative_to(ROOT)
                tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(relative))
                for node in ast.walk(tree):
                    targets = []
                    if isinstance(node, ast.Import):
                        targets = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        if node.level:
                            parent = list(relative.parent.parts)
                            self.assertLessEqual(node.level, len(parent),
                                                 f"{relative}:{node.lineno} escapes its package")
                            base = parent[:len(parent) - node.level + 1]
                            targets = [".".join(base + (node.module or "").split("."))]
                        else:
                            targets = [node.module or ""]
                    for target in targets:
                        dependency = target.split(".")[0]
                        if dependency in project_packages:
                            self.assertIn(dependency, dependencies,
                                          f"{relative}:{node.lineno} imports {target}")

    def test_experiment_configuration_has_one_yaml_location(self):
        self.assertEqual(set(DATASETS), EXPECTED_DATASETS)
        self.assertEqual(CONFIG_ROOT.resolve(), ROOT / "configs" / "experiments")
        self.assertEqual({path.stem for path in CONFIG_ROOT.glob("*.yaml")}, EXPECTED_DATASETS)
        self.assertFalse((ROOT / "experiments" / "configs").exists())
        self.assertEqual(list((ROOT / "configs").rglob("*.json")), [])
        self.assertEqual(list((ROOT / "experiments").rglob("*.json")), [])

    def test_all_nine_yaml_configs_and_ensemble_members_are_valid(self):
        dimensions = ("num_channels", "num_samples", "num_classes")
        for dataset in sorted(EXPECTED_DATASETS):
            with self.subTest(dataset=dataset):
                args = SimpleNamespace(dataset=dataset, config=None, num_workers=None,
                                       eval_batch_size=None)
                config = load_config(args)
                original = deepcopy(config)
                self.assertEqual(config["dataset"], dataset)
                for section in ("model", "training", "input", "evaluation"):
                    self.assertIsInstance(config[section], dict)
                model = ModelConfig.from_dict(config["model"])
                self.assertIsInstance(model.ensemble, EnsembleConfig)
                for name in dimensions:
                    self.assertIs(type(config["model"][name]), int)
                self.assertIsInstance(config["model"]["mirror_permutation"], list)
                self.assertTrue(all(type(index) is int for index in model.mirror_permutation))
                self.assertGreater(config["input"]["sampling_rate_hz"], 0)
                self.assertLessEqual(set(config["input"]),
                                     {"sampling_rate_hz", "channel_names", "require_identity_alignment"})
                self.assertIs(type(config["training"]["batch_size"]), int)
                for name in ("amp", "deterministic"):
                    self.assertIs(type(config["training"][name]), bool)
                for name, value in config["training"].items():
                    if name.endswith("_lr") or name.endswith("weight_decay"):
                        self.assertIn(type(value), (int, float), name)
                self.assertEqual(len(model.ensemble.members), model.ensemble.size)
                for member in model.ensemble.members:
                    self.assertIsInstance(member["name"], str)
                    self.assertIs(type(member["seed"]), int)
                    resolved = member_config(config, member)
                    self.assertIsInstance(resolved, ModelConfig)
                    self.assertFalse(resolved.ensemble.enabled)
                    self.assertEqual(resolved.ensemble.size, 1)
                    for name in (*dimensions, "mirror_permutation"):
                        self.assertEqual(getattr(resolved, name), getattr(model, name))
                self.assertEqual(config, original, "Resolving members must not mutate the experiment")

    def test_experiment_runtime_overrides_do_not_modify_yaml(self):
        args = SimpleNamespace(dataset="bci_iv_2a", config=None, num_workers=2,
                               eval_batch_size=7)
        path = CONFIG_ROOT / "bci_iv_2a.yaml"
        original = load_yaml(path)
        resolved = load_config(args)
        self.assertEqual(resolved["training"]["num_workers"], 2)
        self.assertEqual(resolved["training"]["eval_batch_size"], 7)
        self.assertEqual(resolved["model"], original["model"])
        self.assertEqual(load_yaml(path), original)
        args.config = path
        args.dataset = "hmc"
        with self.assertRaisesRegex(ValueError, "must agree"):
            load_config(args)


class YamlConfigurationTests(unittest.TestCase):
    def test_loader_requires_a_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            for document in ("", "null\n", "- item\n", "scalar\n", "42\n", "true\n"):
                with self.subTest(document=document):
                    path.write_text(document, encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "YAML mapping"):
                        load_yaml(path)
            path.write_text("{}\n", encoding="utf-8")
            self.assertEqual(load_yaml(path), {})

    def test_decimal_scientific_values_and_types_survive_roundtrip(self):
        document = (
            "training:\n"
            "  backbone_lr: 0.0003\n"
            "  reflection_lr: 3e-05\n"
            "  scale: 2.5e03\n"
            "  batch_size: 7\n"
            "  amp: false\n"
            "  patience: null\n"
            "  rates: [0.0, 3.0e-05]\n"
            "  numeric_label: '3.0e-05'\n"
            "label: 合成配置\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.yaml"
            path.write_text(document, encoding="utf-8")
            config = load_yaml(path)
            settings = config["training"]
            for name, expected in (("backbone_lr", 0.0003), ("reflection_lr", 0.00003),
                                   ("scale", 2500.0)):
                self.assertIs(type(settings[name]), float)
                self.assertEqual(settings[name], expected)
            self.assertIs(type(settings["batch_size"]), int)
            self.assertIs(settings["amp"], False)
            self.assertIsNone(settings["patience"])
            self.assertIsInstance(settings["numeric_label"], str)
            self.assertEqual(yaml.safe_load(format_yaml(config)), config)
            destination = Path(directory) / "nested" / "output.yaml"
            save_yaml(destination, config)
            restored = load_yaml(destination)
            self.assertEqual(restored, config)
            self.assertIs(type(restored["training"]["reflection_lr"]), float)
            self.assertIs(type(restored["training"]["batch_size"]), int)
            self.assertIs(restored["training"]["amp"], False)


if __name__ == "__main__":
    unittest.main()
