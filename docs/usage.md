# Usage reference

For source formats and dataset preparation, see the [preprocessing guide](../preprocessing/README.md).

## Environment

Install dependencies into an existing Python 3.11+ environment:

```bash
python -m pip install -r requirements.txt
```

On Linux and macOS, the installation script creates a virtual environment:

```bash
bash install.sh
source .venv/bin/activate
```

The script accepts `PYTHON`, `VENV_DIR`, and optional `TORCH_INDEX_URL` environment variables. Use `TORCH_INDEX_URL` to select the desired PyTorch distribution.

## BCI-IV-2a demo

```bash
python demo.py --device cuda
```

The demo acquires the complete prepared BCI-IV-2a cache when needed, verifies the archive checksum, and then runs the experiment. The default cache directory is `data/bci_iv_2a`; the default output directory is `runs/bci_iv_2a`.

```bash
python demo.py \
  --data-dir data/bci_iv_2a \
  --output-dir runs/bci_iv_2a_run2 \
  --device cuda
```

Use a new output directory for each training run. Training refuses to overwrite an existing member's `joint_best.pt` checkpoint.

## Experiment entry points

Available dataset identifiers are `bci_iv_2a`, `physionet_mi`, `isruc`, `hmc`, `mat`, `mumtaz`, `handmi`, `ssvep_eeg`, and `aestheeg`.

```bash
python -m experiments.run --dataset handmi \
  --data-dir data/handmi \
  --output-dir runs/handmi \
  --device cuda
```

Dataset entry points select the same defaults, for example:

```bash
python -m experiments.handmi --data-dir data/handmi --device cuda
```

The selected dataset defaults to `configs/experiments/<dataset>.yaml`. Pass `--config` to use a custom YAML file; its `dataset` value must match `--dataset`.

```bash
python -m experiments.run --dataset handmi \
  --config configs/experiments/handmi.yaml \
  --data-dir data/handmi --output-dir runs/handmi --device cuda

python -m experiments.run --dataset handmi --show-config
python -m experiments.run --help
```

## YAML configuration

Each supplied experiment YAML is complete and can be copied as the starting point for a custom experiment.

| Section | Contents |
|---|---|
| `dataset` | Dataset identifier |
| `model` | Architecture, input dimensions, reflection settings, and ensemble |
| `training` | Optimization, scheduling, batch sizes, seeds, and checkpoint selection |
| `input` | Expected sampling rate and channel order, with an identity-alignment requirement where applicable |
| `evaluation` | Reference balanced accuracy for reporting |

`model.ensemble` defines member seeds and model overrides, aggregation, and any probability weights or temperatures. One training invocation runs the complete configured ensemble. For example, the BCI-IV-2a configuration contains this member specification within `model.ensemble.members`:

```yaml
members:
  - name: physical
    seed: 307
    model_overrides:
      reflection_init: physical
      reflection_init_strength: 0.0
      reflection_init_seed: 401
```

Member names determine checkpoint subdirectories. Preserve member order when supplying checkpoints for evaluation. The ensemble `size` must equal the number of members.

Input dimensions, class count, sampling rate, channel correspondence, and configured channel order are checked against the prepared dataset before running.

## Execution resources

| Option | Purpose |
|---|---|
| `--device cuda` / `--device cpu` | Select the compute device |
| `--num-workers N` | Override the data loader worker count |
| `--cpu-threads N` | Set PyTorch CPU threads; default: 4 |
| `--eval-batch-size N` | Override the evaluation batch size |
| `--show-config` | Print the resolved configuration and exit |

## Checkpoint evaluation

To evaluate the checkpoints in an existing run directory:

```bash
python -m experiments.run --dataset bci_iv_2a \
  --data-dir data/bci_iv_2a --output-dir runs/bci_iv_2a \
  --mode evaluate --device cuda
```

To write evaluation outputs to another directory, provide one checkpoint for each configured member, in `model.ensemble.members` order:

```bash
python -m experiments.run --dataset bci_iv_2a \
  --config configs/experiments/bci_iv_2a.yaml \
  --data-dir data/bci_iv_2a --mode evaluate \
  --checkpoints runs/bci_iv_2a/physical/joint_best.pt \
                runs/bci_iv_2a/cayley060/joint_best.pt \
                runs/bci_iv_2a/haar/joint_best.pt \
  --output-dir runs/bci_iv_2a_evaluation --device cuda
```

Evaluation requires the prepared test split and the same model and ensemble configuration used during training.

## Run outputs

The output directory contains the resolved `config.yaml`, `results.json`, and `test_predictions.npz`. Training adds one subdirectory per ensemble member containing checkpoints and training histories.

`results.json` reports ensemble and member metrics, checkpoint and data-manifest hashes, sample counts, elapsed time, and dependency versions. Metrics include balanced accuracy, Cohen's κ, and weighted F1; binary tasks also report AUROC and area under the precision–recall curve. `test_predictions.npz` contains labels, combined logits, and individual member logits in matching sample order.

## Python model interface

The following example creates one model from an experiment's base model settings and runs two synthetic trials:

```python
import torch
from model import ModelConfig, SCoRE
from utils.config import load_yaml

experiment = load_yaml("configs/experiments/bci_iv_2a.yaml")
config = ModelConfig.from_dict(experiment["model"])
model = SCoRE(config).eval()
eeg = torch.randn(2, config.num_channels, config.num_samples)
alignment_pair = torch.eye(config.num_channels).expand(2, 2, -1, -1)
with torch.no_grad():
    logits = model(eeg, alignment_pair)  # [2, config.num_classes]
```

For real inputs, use the prepared EEG and its alignment pairs. `eeg` has shape `[batch, channels, time]`. `alignment_pair` has shape `[batch, 2, channels, channels]`, containing the alignment operator followed by its inverse. Match `ModelConfig` to the dataset's dimensions and channel correspondence, and place the model and inputs on the same device.

Use the experiment runner to apply member overrides and run the configured ensemble. `model/initialization.py` provides the spectral output-direction initializer; `model/ensemble.py` provides `SCoREEnsemble` and `aggregate_logits` for prediction aggregation. The `model` package is independent of dataset acquisition and experiment execution.
