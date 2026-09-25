# SCoRE

**Structured Contrast Reflection Equivariance for EEG**

SCoRE combines a shared spatiotemporal network with learned input and output reflections. The input action is transported into each recording's aligned coordinates. A Householder output reflection aligns the two routes before prediction fusion. Its direction is learned for multiclass tasks, while binary tasks use the fixed class-swap reflection. Training consists of supervised warm-up, output-direction initialization, and joint optimization.

## Installation

Use Python 3.11 or newer. A CUDA-enabled PyTorch installation is recommended for training.

```bash
git clone https://github.com/junye988/SCoRE.git
cd SCoRE
bash install.sh
source .venv/bin/activate
```

For an existing environment:

```bash
python -m pip install -r requirements.txt
```

`install.sh` accepts `PYTHON`, `VENV_DIR`, and an optional `TORCH_INDEX_URL` environment variable for the desired PyTorch distribution.

## BCI-IV-2a demo

Prepare the official GDF recordings and corresponding MAT labels:

```bash
python -m preprocessing.bci_iv_2a \
  --source /path/to/BCICIV_2a_gdf \
  --labels /path/to/true_labels \
  --output data/bci_iv_2a
```

Run the configured experiment:

```bash
python demo.py \
  --data-dir data/bci_iv_2a \
  --output-dir runs/bci_iv_2a \
  --device cuda
```

This command trains the configured ensemble members and evaluates their combined predictions. The model, optimization settings, and ensemble definition are in `experiments/configs/bci_iv_2a.json`.

```bash
python demo.py --show-config
python demo.py --help
```

## Dataset preprocessing

Each dataset has its own preprocessing module. Modules accept `--source` and `--output`; `--help` lists the supported source formats and additional arguments.

See [the preprocessing reference](preprocessing/README.md) for source layouts, data partitions, and alignment details.

| Dataset | Module | Input to the preparation module | Model input |
|---|---|---|---|
| BCI-IV-2a | `preprocessing.bci_iv_2a` | Official GDF recordings and MAT labels | 22 × 800, 200 Hz |
| PhysioNet-MI | `preprocessing.physionet_mi` | Official EDF recordings or REVE-format LMDB | 64 × 800, 200 Hz |
| ISRUC | `preprocessing.isruc` | Raw Group I recordings or sequence and label arrays | 6 × 3000, 100 Hz |
| HMC | `preprocessing.hmc` | Official EDF recordings and sleep-scoring text files | 4 × 3000, 100 Hz |
| MAT | `preprocessing.mat` | Raw EDF recordings or preprocessed 100-Hz LMDB | 19 × 500, 100 Hz |
| Mumtaz | `preprocessing.mumtaz` | Raw EDF recordings or released preprocessed LMDB | 19 × 1000, 200 Hz |
| HandMI | `preprocessing.handmi` | Anonymous EEG recordings and event files | 32 × 400, 200 Hz |
| SSVEP-EEG | `preprocessing.ssvep_eeg` | Anonymous EEG recordings and event files | 32 × 400, 200 Hz |
| AesthEEG | `preprocessing.aestheeg` | Anonymous recordings or subject epochs | 32 × 751, 250 Hz |

Download links for HandMI, SSVEP-EEG, and AesthEEG will be added upon data release. Their input package formats are described in the preprocessing reference.

For example:

```bash
python -m preprocessing.handmi --source /path/to/HandMI --output data/handmi
python -m preprocessing.ssvep_eeg --source /path/to/SSVEP-EEG --output data/ssvep_eeg
python -m preprocessing.aestheeg --source /path/to/AesthEEG --output data/aestheeg
python -m preprocessing.hmc --source /path/to/HMC/recordings --output data/hmc
python -m preprocessing.mumtaz --source /path/to/Mumtaz/edf --source-format raw --output data/mumtaz
```

Preprocessing preserves the dataset partitions and channel order. Alignment matrices are estimated independently for each subject, session, or recording group, using EEG signal statistics. The supplied HMC and MAT experiment configurations use their preprocessed inputs without an additional EA transform. The other configurations retain their dataset-specific alignment. The prepared dataset manifest records the sampling rate, classes, alignment, and input dimensions.

Prepared files use a common format:

```text
data/<dataset>/
  manifest.json
  train_samples.npy
  train_labels.npy
  train_group_indices.npy
  train_alignment_pairs.npy
  val_*.npy
  test_*.npy
```

Samples have shape `[trials, channels, time]`. Alignment pairs contain each group's forward and inverse matrices. The loader uses the manifest to distinguish aligned samples from samples requiring alignment.

## Experiments

The dataset identifiers are `bci_iv_2a`, `physionet_mi`, `isruc`, `hmc`, `mat`, `mumtaz`, `handmi`, `ssvep_eeg`, and `aestheeg`.

```bash
python -m experiments.handmi \
  --data-dir data/handmi \
  --output-dir runs/handmi \
  --device cuda

python demo.py --dataset isruc \
  --data-dir data/isruc \
  --output-dir runs/isruc \
  --device cuda
```

Every dataset module also accepts `--config` for a custom experiment JSON. `--num-workers`, `--cpu-threads`, and `--eval-batch-size` control execution resources.

The JSON configuration groups architecture and ensemble settings under `model`, optimization under `training`, and input settings under `data`. `model.ensemble` specifies member seeds, input-reflection initializations, aggregation, and any probability weights or temperatures. One invocation runs one complete configured ensemble.

Balanced accuracy is the primary metric. Evaluation also computes Cohen's κ and weighted F1, with AUROC and area under the precision–recall curve for binary tasks. Outputs include member checkpoints, training histories, sample-level predictions, and `results.json`.

## Checkpoint evaluation

Evaluate trained members in the order listed by `model.ensemble.members`:

```bash
python demo.py --dataset bci_iv_2a \
  --data-dir data/bci_iv_2a \
  --mode evaluate \
  --checkpoints runs/bci_iv_2a/physical/joint_best.pt \
                runs/bci_iv_2a/cayley060/joint_best.pt \
                runs/bci_iv_2a/haar/joint_best.pt \
  --output-dir runs/bci_iv_2a_evaluation
```

## Model interface

```python
from model import ModelConfig, SCoRE

model = SCoRE(ModelConfig())
logits = model(eeg, alignment_pair)
```

`eeg` has shape `[batch, channels, time]`; `alignment_pair` has shape `[batch, 2, channels, channels]`. The two matrices are the alignment operator and its inverse. `model/initialization.py` implements the spectral output-direction initializer. `model/ensemble.py` implements the configured prediction aggregation.

## Repository layout

```text
model/                 Architecture, initialization, and ensembles
preprocessing/         Dataset-specific preparation modules
experiments/           Dataset entry points and shared training code
experiments/configs/   Nine experiment configurations
demo.py                BCI-IV-2a demo and common CLI
requirements.txt       Python dependencies
install.sh             Environment installation
```

## License

The project code is distributed under the MIT License. Third-party preprocessing notices are included in `preprocessing/licenses/`. Datasets retain their respective licenses.
