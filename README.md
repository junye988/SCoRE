# SCoRE

**Structured Contrast Reflection Equivariance for EEG**

PyTorch implementation of SCoRE with reproducible experiments across nine EEG datasets.

## Installation

Use Python 3.11 or newer and a CUDA-enabled PyTorch build for GPU training.

```bash
git clone https://github.com/junye988/SCoRE.git
cd SCoRE
python -m pip install -r requirements.txt
```

Alternatively, run `bash install.sh` to create a virtual environment. See [usage](docs/usage.md) for installation options.

## Quick start

```bash
python demo.py --device cuda
```

The demo automatically downloads the complete [prepared BCI-IV-2a cache](https://figshare.com/s/cc1e803b0838940a522a) (340 MB), verifies its SHA-256 checksum, and extracts it to `data/bci_iv_2a`. Later runs reuse the cache.

Training and evaluation use [`configs/experiments/bci_iv_2a.yaml`](configs/experiments/bci_iv_2a.yaml), with checkpoints, predictions, and metrics written to `runs/bci_iv_2a`. Use `--device cpu` to run without CUDA.

```bash
python demo.py --show-config
python demo.py --help
```

## Train and evaluate

Prepare a dataset using the [preprocessing guide](preprocessing/README.md), then select its YAML experiment configuration:

```bash
python -m experiments.run --dataset handmi \
  --config configs/experiments/handmi.yaml \
  --data-dir data/handmi --output-dir runs/handmi --device cuda

python -m experiments.run --dataset handmi \
  --config configs/experiments/handmi.yaml --mode evaluate \
  --data-dir data/handmi --output-dir runs/handmi --device cuda
```

Evaluation loads checkpoints from the run directory. Balanced accuracy is the primary metric; detailed results are saved in `results.json`.

See [usage](docs/usage.md) for custom configurations, checkpoint selection, resource options, and the Python model interface.

## Data availability

| Dataset | Source |
|---|---|
| BCI-IV-2a | [BCI Competition IV](https://www.bbci.de/competition/iv/) |
| PhysioNet-MI | [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) |
| ISRUC | [ISRUC-Sleep](https://sleeptight.isr.uc.pt/?page_id=48) |
| HMC | [HMC Sleep Staging Database](https://physionet.org/content/hmc-sleep-staging/1.1/) |
| MAT | [EEG During Mental Arithmetic Tasks](https://physionet.org/content/eegmat/1.0.0/) |
| Mumtaz | [EEG Data New](https://figshare.com/articles/dataset/EEG_Data_New/4244171) |
| HandMI, SSVEP-EEG, AesthEEG | [Datasets on Figshare](https://figshare.com/s/07498f285053b3a8229b) |

HandMI, SSVEP-EEG, and AesthEEG accompany a manuscript currently under review and should not be redistributed until the manuscript is publicly available.

Source layouts and preparation commands are documented in the [preprocessing guide](preprocessing/README.md). The demo download URL and checksum are configured in [`configs/data_sources.yaml`](configs/data_sources.yaml).

## Repository layout

```text
demo.py                         Download data and run the BCI-IV-2a demo
configs/
  experiments/<dataset>.yaml    Complete experiment configurations
  data_sources.yaml             Download URLs and checksums
model/                          Model architecture, configuration, and initialization
preprocessing/
  <dataset>.py                  Dataset preparation commands
  raw_readers/                  Raw recording readers
  dataset_cache.py              Prepared-cache format and I/O
  dataset_download.py           Prepared-data checks and download
  recording_import.py           Recording and cached-array imports
  window_import.py              Window-archive imports
experiments/
  run.py                        Training and evaluation CLI
  experiment.py                 Experiment workflow
  data_loader.py                Runtime datasets and data loaders
  training.py                   Optimization and checkpoint selection
  prediction.py                 Batched prediction
utils/                          Shared file I/O, YAML, downloads, seeds, and metrics
docs/usage.md                   Command and model interface reference
tests/                          Verification suite
data/<dataset>/                 Prepared EEG caches (generated)
runs/<dataset>/                 Checkpoints and results (generated)
```

## License

Code is distributed under the [MIT License](LICENSE). Third-party preprocessing notices are included in [`preprocessing/licenses/`](preprocessing/licenses/). Datasets retain their respective licenses.
