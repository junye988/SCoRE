"""Dataset-independent configuration and numerical model boundary checks."""

import ast
from dataclasses import MISSING, fields
from pathlib import Path
import unittest

import numpy as np
import torch

from model import EnsembleConfig, ModelConfig, SCoRE, WarmupModel, aggregate_logits


def synthetic_config(**overrides):
    values = dict(num_channels=3, num_samples=96, num_classes=3,
                  mirror_permutation=(1, 0, 2), temporal_filters=3,
                  temporal_kernel=5, pool_kernel=8, pool_stride=4,
                  energy_bins=3, embedding_width=8, dropout=0.0)
    return ModelConfig(**(values | overrides))


class ModelBoundaryTests(unittest.TestCase):
    def test_data_dependent_architecture_inputs_are_required(self):
        required = {"num_channels", "num_samples", "num_classes", "mirror_permutation"}
        for field in fields(ModelConfig):
            if field.name in required:
                self.assertIs(field.default, MISSING)
                self.assertIs(field.default_factory, MISSING)
        values = synthetic_config().to_dict()
        for name in required:
            with self.subTest(missing=name), self.assertRaises(TypeError):
                ModelConfig.from_dict({key: value for key, value in values.items() if key != name})

    def test_model_sources_only_depend_on_model_and_numerical_libraries(self):
        allowed = {"collections", "dataclasses", "math", "typing", "numpy", "torch"}
        forbidden_calls = {"open", "load", "save", "loadtxt", "savetxt", "load_state_dict_from_url",
                           "urlopen", "read_text", "write_text", "read_bytes", "write_bytes"}
        for path in sorted((Path(__file__).resolve().parents[1] / "model").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name.split(".")[0], allowed, str(path))
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    self.assertIn(node.module.split(".")[0], allowed, str(path))
                elif isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 1, str(path))
                elif isinstance(node, ast.Call):
                    name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                        node.func.id if isinstance(node.func, ast.Name) else None)
                    self.assertNotIn(name, forbidden_calls, str(path))

    def test_ensemble_configuration_and_aggregation_remain_in_model(self):
        ensemble = EnsembleConfig(
            enabled=True, size=2, aggregation="weighted_probabilities",
            weights=(1.0, 3.0), temperatures=(1.0, 2.0),
            members=({"name": "first", "seed": 7, "model_overrides": {"reflection_init": "physical"}},
                     {"name": "second", "seed": 8, "model_overrides": {"reflection_init": "haar"}}),
        )
        config = synthetic_config(ensemble=ensemble)
        restored = ModelConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)
        logits = torch.tensor([[[1.0, 2.0, 0.0]], [[3.0, 1.0, 0.0]]])
        expected = (0.25 * logits[0].double().softmax(-1)
                    + 0.75 * (logits[1].double() / 2).softmax(-1)).log()
        torch.testing.assert_close(aggregate_logits(logits, restored.ensemble), expected,
                                   rtol=0, atol=0)

    def test_explicit_synthetic_config_checkpoint_roundtrip(self):
        config = synthetic_config()
        generator = torch.Generator().manual_seed(19)
        sample = torch.randn(2, config.num_channels, config.num_samples, generator=generator)
        pair = torch.eye(config.num_channels, dtype=torch.float64).repeat(2, 2, 1, 1)
        for model_type in (SCoRE, WarmupModel):
            with self.subTest(model=model_type.__name__), torch.inference_mode():
                original = model_type(config).eval()
                restored = model_type(config.to_dict()).eval()
                restored.load_state_dict(original.state_dict(), strict=True)
                self.assertEqual(restored.classifier.out_features, config.num_classes)
                expected = original(sample, pair, return_diagnostics=True)
                actual = restored(sample, pair, return_diagnostics=True)
                self.assertEqual(actual["logits"].shape, (2, config.num_classes))
                self.assertEqual(actual.keys(), expected.keys())
                for name in expected:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_warmup_numpy_transport_preserves_product_order_and_cast(self):
        random = np.random.default_rng(17)
        matrices = np.eye(3)[None] + random.normal(size=(4, 3, 3)) * 0.1
        pairs = np.stack((matrices, np.linalg.inv(matrices)), axis=1)
        for precision in ("float32", "float64"):
            with self.subTest(initial_reflection_precision=precision):
                warmup = WarmupModel(synthetic_config(
                    reflection_init="cayley", reflection_init_strength=0.37,
                    initial_reflection_precision=precision,
                ))
                initial = warmup.initial_input_reflection.detach().cpu().double().numpy()
                expected = np.stack([a @ initial @ inverse for a, inverse in pairs]).astype(np.float32)
                actual = warmup.transport_reflections(pairs)
                self.assertEqual(actual.device.type, "cpu")
                self.assertEqual(actual.dtype, torch.float32)
                np.testing.assert_array_equal(actual.numpy(), expected)
                with self.assertRaisesRegex(ValueError, "alignment pairs"):
                    warmup.transport_reflections(pairs[:, 0])


if __name__ == "__main__":
    unittest.main()
