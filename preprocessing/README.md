# Dataset preparation

Run these commands from the repository root. Input paths refer to locally downloaded datasets. Output directories must be new. Each command writes a portable cache for `experiments` and `demo.py`.

| Module | Input | Model input | Alignment in the released configuration |
|---|---|---|---|
| `bci_iv_2a` | Official GDF and MAT labels | 22 × 800, 200 Hz | Per subject/session |
| `physionet_mi` | Official EDF recordings or REVE-format LMDB | 64 × 800, 200 Hz | Per recording |
| `isruc` | Preprocessed `seq/` and `labels/` arrays at 200 Hz | 6 × 3000, 100 Hz | Per subject |
| `hmc` | Preprocessed `train/`, `eval/`, `test/` pickle epochs at 100 Hz | 4 × 3000, 100 Hz | Identity |
| `mat` | Raw EDF or preprocessed 100-Hz LMDB with subject IDs | 19 × 500, 100 Hz | Identity |
| `mumtaz` | Raw EDF or REVE preprocessed LMDB | 19 × 1000, 200 Hz | Per recording |
| `handmi` | Released anonymous EDF/CSV package | 32 × 400, 200 Hz | Per subject |
| `ssvep_eeg` | Released anonymous EDF/CSV package | 32 × 400, 200 Hz | Per subject |
| `aestheeg` | Released anonymous EDF/event CSV package | 32 × 751, 250 Hz | Per subject |

The model input dimensions include dataset loading transformations. ISRUC applies float32 conversion, division by 10 and average pooling from 200 to 100 Hz. Its bundled sequence manifest preserves the experiment's sample order across platforms. MAT retains the 19 scalp channels and omits the auxiliary A2–A1 channel. Mumtaz applies the release loader's division by 100 before float32 conversion. PhysioNet-MI applies division by 100 before its float64 alignment calculation. The 751 AesthEEG time points include both endpoints of the −0.5 to 2.5 second interval.

```bash
python -m preprocessing.bci_iv_2a --source /data/bci/gdf --labels /data/bci/labels --output /data/score/bci_iv_2a
python -m preprocessing.physionet_mi --source /data/physionet/edf --output /data/score/physionet_mi
python -m preprocessing.isruc --source /data/isruc --output /data/score/isruc
python -m preprocessing.hmc --source /data/hmc --output /data/score/hmc
python -m preprocessing.mat --source /data/mat/lmdb --output /data/score/mat
python -m preprocessing.mumtaz --source /data/mumtaz/edf --source-format raw --output /data/score/mumtaz
python -m preprocessing.handmi --source /data/HandMI --output /data/score/handmi
python -m preprocessing.ssvep_eeg --source /data/SSVEP-EEG --output /data/score/ssvep_eeg
python -m preprocessing.aestheeg --source /data/AesthEEG --output /data/score/aestheeg
```

PhysioNet-MI uses the official `S001` through `S109` EDF folders and the released REVE preprocessing implementation. The pipeline selects the 64 EEG channels, applies average referencing, a 0.3 Hz high-pass filter and a 60 Hz notch, resamples directly to 200 Hz, and extracts four-second imagery epochs. Use `--source-format released` to import an existing REVE-format LMDB.

MAT also accepts `--source-format raw` with the directory containing `Subject00_1.edf` through `Subject35_2.edf`. It applies the 0.5–45 Hz bandpass, resamples to 100 Hz, extracts disjoint 5-second windows and removes each channel's temporal mean within each window.

Mumtaz accepts `--source-format raw` with the original EC/EO EDF directory. It follows REVE's released sequence of 200 Hz resampling, 0.3–30 Hz bandpass filtering, 50 Hz notch filtering and disjoint 5-second epochs. The reader retains the sorted recording partitions and the loader's amplitude scaling. The optional `preprocessing/requirements-reve.txt` specifies the tested EDF preprocessing environment. Install it in a separate environment with `pip install -r requirements.txt -r preprocessing/requirements-reve.txt`.

## Partitions and alignment

BCI-IV-2a uses subjects 1–5 for training, 6–7 for validation and 8–9 for testing, including both sessions. PhysioNet-MI uses subjects 1–70, 71–89 and 90–109. ISRUC uses subjects 1–80, 81–90 and 91–100. HandMI and SSVEP-EEG use S001–S008, S009–S010 and S011–S012. AesthEEG uses the fixed 40/10/10 subject partition encoded in `aestheeg.py`. The HMC, MAT and Mumtaz readers retain their input partitions and sample order.

Alignment uses each group's complete unlabeled input within its partition. The resulting operators remain fixed. BCI-IV-2a, PhysioNet-MI, ISRUC, Mumtaz and AesthEEG use the mean epoch Gram matrix with a relative eigenvalue floor of 10⁻¹². HandMI estimates covariance on complete imagery trials before overlapping windows. SSVEP-EEG estimates covariance on disjoint stimulation/rest intervals before window extraction. Both use relative ridge 10⁻⁴ and eigenvalue floor 10⁻⁶. HMC and MAT use identity operators in their released model configurations. `--alignment ea` enables EA for these clinical input formats.

## Import existing experiment inputs

BCI-IV-2a, PhysioNet-MI, ISRUC, HMC, MAT, Mumtaz and AesthEEG accept `--source-format experiment-cache`. This preserves the existing sample values and order. Clinical caches store samples before EA. Pass `--ea-source` to reuse their saved alignment operators, or compute them from the preprocessed inputs using `--alignment ea`.

```bash
python -m preprocessing.isruc --source /data/isruc_reference_cache --source-format experiment-cache --ea-source /data/isruc_ea --alignment ea --output /data/score/isruc
python -m preprocessing.handmi --source /data/mi_windows.npz --source-format windows --output /data/score/handmi
python -m preprocessing.ssvep_eeg --source /data/ssvep_windows.npz --source-format windows --output /data/score/ssvep_eeg
python -m preprocessing.aestheeg --source /data/aesthetic_epochs --source-format subject-epochs --output /data/score/aestheeg
```

## Cache format

Each cache contains `manifest.json` and four arrays per split:

- `{split}_samples.npy`: EEG with shape `[N, C, T]`.
- `{split}_labels.npy`: integer class indices with shape `[N]`.
- `{split}_group_indices.npy`: split-local alignment group indices with shape `[N]`.
- `{split}_alignment_pairs.npy`: frozen `A` and `inverse(A)` with shape `[G, 2, C, C]`.

The manifest records channels, class indices, sample rate, dimensions, counts, dtypes and `samples_aligned`. The loader applies `A` exactly once when this flag is false. Saved aligned samples retain their original float32 or float64 precision. Model-side referencing and normalization are controlled by the model configuration.
