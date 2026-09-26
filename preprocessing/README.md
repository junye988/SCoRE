# Dataset preparation

Run these commands from the repository root. Input paths refer to locally downloaded datasets. Output directories must be new. Each command writes a portable cache for `experiments` and `demo.py`.

The public dataset modules provide the preparation CLI and partition conversion. Dataset-specific recording readers live in `readers/`; `alignment.py` contains the shared frozen alignment and electrode transforms, and `cache.py` reads and writes the portable cache. `acquisition.py` resolves prepared-data locations using the sources in `configs/data_sources.yaml` at the repository root.

## Data availability

Official download pages are available for [BCI-IV-2a](https://www.bbci.de/competition/iv/) (GDF recordings and evaluation labels), [PhysioNet-MI](https://physionet.org/content/eegmmidb/1.0.0/), [ISRUC](https://sleeptight.isr.uc.pt/?page_id=48) (raw Group I recordings), [HMC](https://physionet.org/content/hmc-sleep-staging/1.1/), [MAT](https://physionet.org/content/eegmat/1.0.0/) and [Mumtaz](https://figshare.com/articles/dataset/EEG_Data_New/4244171).

The newly collected HandMI, SSVEP-EEG and AesthEEG datasets are available on [Figshare](https://figshare.com/s/07498f285053b3a8229b). Their package layouts are documented below.

For a quick start, `python demo.py --device cuda` downloads the complete [prepared BCI-IV-2a archive from Figshare](https://figshare.com/s/cc1e803b0838940a522a) (340 MB) when `data/bci_iv_2a` is absent, verifies its checksum, and extracts it. Later runs reuse the cache. BCI-IV-2a provides a compact demo; prepare the other datasets separately, as their caches can be larger. The raw BCI-IV-2a preparation command below remains an alternative.

## Preparation modules

| Module | Input | Model input | Alignment in the released configuration |
|---|---|---|---|
| `bci_iv_2a` | Official GDF and MAT labels | 22 × 800, 200 Hz | Per subject/session |
| `physionet_mi` | Official EDF recordings or REVE-format LMDB | 64 × 800, 200 Hz | Per recording |
| `isruc` | Raw Group I recordings or preprocessed sequence arrays | 6 × 3000, 100 Hz | Per subject |
| `hmc` | Official EDF recordings and sleep-scoring text files | 4 × 3000, 100 Hz | Identity |
| `mat` | Raw EDF or preprocessed 100-Hz LMDB with subject IDs | 19 × 500, 100 Hz | Identity |
| `mumtaz` | Raw EDF or REVE preprocessed LMDB | 19 × 1000, 200 Hz | Per recording |
| `handmi` | Anonymous data package with EDF/CSV files | 32 × 400, 200 Hz | Per subject |
| `ssvep_eeg` | Anonymous data package with EDF/CSV files | 32 × 400, 200 Hz | Per subject |
| `aestheeg` | Anonymous data package with EDF/event CSV files | 32 × 751, 250 Hz | Per subject |

The model input dimensions include dataset loading transformations. ISRUC applies float32 conversion, division by 10 and average pooling from 200 to 100 Hz. Its bundled sequence manifest preserves the experiment's sample order across platforms. MAT retains the 19 scalp channels and omits the auxiliary A2–A1 channel. Mumtaz applies the release loader's division by 100 before float32 conversion. PhysioNet-MI applies division by 100 before its float64 alignment calculation. The 751 AesthEEG time points include both endpoints of the −0.5 to 2.5 second interval.

```bash
python -m preprocessing.bci_iv_2a --source /data/bci/gdf --labels /data/bci/labels --output /data/score/bci_iv_2a
python -m preprocessing.physionet_mi --source /data/physionet/edf --output /data/score/physionet_mi
python -m preprocessing.isruc --source /data/isruc/group1 --source-format raw --output /data/score/isruc
python -m preprocessing.hmc --source /data/hmc/recordings --output /data/score/hmc
python -m preprocessing.mat --source /data/mat/lmdb --output /data/score/mat
python -m preprocessing.mumtaz --source /data/mumtaz/edf --source-format raw --output /data/score/mumtaz
python -m preprocessing.handmi --source /data/HandMI --output /data/score/handmi
python -m preprocessing.ssvep_eeg --source /data/SSVEP-EEG --output /data/score/ssvep_eeg
python -m preprocessing.aestheeg --source /data/AesthEEG --output /data/score/aestheeg
```

PhysioNet-MI uses the official `S001` through `S109` EDF folders and the released REVE preprocessing implementation. The pipeline selects the 64 EEG channels, applies average referencing, a 0.3 Hz high-pass filter and a 60 Hz notch, resamples directly to 200 Hz, and extracts four-second imagery epochs. Use `--source-format released` to import an existing REVE-format LMDB.

HMC uses the official `recordings/` folder containing `SNXXX.edf` and `SNXXX_sleepscoring.txt`. It retains F4–M1, C4–M1, O2–M1 and C3–M2, applies a 0.3–45 Hz bandpass and a 50 Hz notch, resamples to 100 Hz, and extracts labeled 30-second epochs in microvolts. The bundled `metadata/hmc_recording_splits.json` fixes the 123/13/15 recording partition and epoch order, yielding 111,970/11,795/13,478 training/validation/test epochs. Existing 100-Hz pickle epochs in `train/`, `eval/` and `test/` can be imported with `--source-format released`.

ISRUC uses `1/1.rec` through `100/100.rec` and the corresponding first-scorer files `<subject>_1.txt`. The raw pipeline applies a 0.3–35 Hz bandpass and a 50 Hz notch, retains source channel indices 2–7, and forms complete sequences of twenty 30-second epochs. Float32 conversion, division by 10 and average pooling produce the 100 Hz model inputs. Existing `seq/` and `labels/` arrays at 200 Hz can be imported with `--source-format released`.

The ISRUC channel inventory records actual source labels separately from nominal model slots. The selected columns are `C3-A2, O1-A2, C4-A1, O2-A1, X1, X2` for subject 8 and `LOC, A2, C4, O2, C3, O1` for subject 40. These source mappings are included in each generated cache manifest.

MAT also accepts `--source-format raw` with the directory containing `Subject00_1.edf` through `Subject35_2.edf`. It applies the 0.5–45 Hz bandpass, resamples to 100 Hz, extracts disjoint 5-second windows and removes each channel's temporal mean within each window.

Mumtaz accepts `--source-format raw` with the original EC/EO EDF directory. It follows REVE's released sequence of 200 Hz resampling, 0.3–30 Hz bandpass filtering, 50 Hz notch filtering and disjoint 5-second epochs. The reader retains the sorted recording partitions and the loader's amplitude scaling. The optional `preprocessing/requirements-reve.txt` specifies the tested EDF preprocessing environment. Install it in a separate environment with `pip install -r requirements.txt -r preprocessing/requirements-reve.txt`.

## Anonymous data package formats

The HandMI, SSVEP-EEG and AesthEEG commands take the package root as `--source`. The raw recordings contain 32 channels sampled at 500 Hz. For the supplied experiment configurations, retain this channel order after removing the `EEG ` prefix and `-CPz` suffix:

```text
Fp1 Fpz Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 M1 T7 C3 Cz
C4 T8 M2 CP5 CP1 CP2 CP6 P7 P3 Pz P4 P8 POz O1 Oz O2
```

### HandMI

Provide all twelve anonymous participants `S001` through `S012` in this layout:

```text
HandMI/
  manifest.json
  edf/S001.edf
  events/S001.csv
  ...
  edf/S012.edf
  events/S012.csv
```

`manifest.json` must contain a JSON object. The HandMI reader discovers recordings from the filenames above and does not require specific manifest fields. A descriptive package manifest can contain `{"dataset": "HandMI", "subject_count": 12, "sampling_rate_hz": 500, "eeg_channels": 32}`.

Each EDF must be continuous EDF+C with 32 channels whose labels start with `EEG `, physical units `uV`, and one `EDF Annotations` channel containing the recorded numeric event markers. Keep these markers for synchronization with the CSV clock.

CSV files use UTF-8 and this exact header order:

```csv
subject_id,block,trial_id,label,baseline_onset_s,cue_onset_s,mi_onset_s,rest_onset_s,trial_end_s,mi_duration_s
```

`subject_id` matches the filename. `block` is a positive integer, `trial_id` is unique and increasing within 1–999, and `label` is 0 for left-hand or 1 for right-hand imagery. Times are finite seconds relative to the first baseline onset, which is zero. The five phase boundaries increase within each trial and trials do not overlap. Set `mi_duration_s` to 4 and retain the measured onset times. The reader aligns these times to EDF events before extracting imagery windows.

### SSVEP-EEG

Use the same `manifest.json`, `edf/S001.edf` and `events/S001.csv` layout for `S001` through `S012`. EDF format, channel labels, physical units and numeric annotations follow the HandMI specification above.

The manifest requires these top-level fields:

| Field | Value |
|---|---|
| `dataset` | `"SSVEP_dataset"` |
| `subject_count` | `12` |
| `trial_count` | `768` |
| `sampling_rate_hz` | `500` |
| `eeg_channels` | `32` |
| `files` | One object for each of `S001` through `S012`, with no duplicates |

Each `files` object has `subject_id`, `edf`, `events`, `trial_count`, `edf_sha256` and `events_sha256`. For example, the paths for `S001` are `edf/S001.edf` and `events/S001.csv`, and its `trial_count` is 64. Both checksum fields contain the corresponding file's SHA-256 digest as 64 lowercase hexadecimal characters. Checksums are computed on the final packaged file bytes.

CSV files use UTF-8 and this exact header order:

```csv
subject_id,block,trial_id,label,frequency_hz,baseline_onset_s,cue_onset_s,gaze_onset_s,rest_onset_s,trial_end_s,baseline_duration_s,cue_duration_s,gaze_duration_s,rest_duration_s
```

Each CSV contains 64 rows with `trial_id` 1–64. Rows 1–32 belong to block 1 and rows 33–64 to block 2. Each block has eight trials at each frequency. Labels 0, 1, 2 and 3 correspond to 5, 7.5, 12 and 15 Hz, respectively. Rest is extracted from the recorded baseline/rest intervals and receives label 4 during preprocessing; it is not an additional CSV trial.

The four duration fields are 2, 1, 4 and 1 seconds, respectively. Onset and end fields retain measured times in seconds from the first baseline at zero, with increasing phase boundaries and non-overlapping trials. Preserve the recorded pause between blocks and the matching EDF markers, which determine the mapping between the CSV and EEG clocks. Every CSV field must be present and nonempty.

### AesthEEG

Provide anonymous subject folders `001` through `060`, either directly under `--source` or under its `subjects/` directory. No root manifest is required.

```text
AesthEEG/
  subjects/
    001/
      recording_Segment_1.edf
      recording_Segment_2.edf
      AestheticEEG_001.csv
    ...
    060/
      recording.edf
      AestheticEEG_060.csv
```

One EDF per subject is sufficient. For split recordings, use the suffix `_Segment_<integer>.edf` so that segments are concatenated in numeric order. Each recording contains the 32 EEG channels above and image-onset annotations with code `20` or `4`. Response codes `61`–`67` encode ratings 1–7 and support trial alignment when the number of image markers differs from the CSV by up to two.

The CSV requires these case-sensitive columns; their order is unrestricted:

| Column | Content |
|---|---|
| `PresentationOrder` | Numeric stimulus presentation index |
| `Category` | `face` or `landscape`, case-insensitive |
| `Filename` | Stimulus image filename, retained as metadata |
| `Score` | Aesthetic rating on the 1–7 scale; missing values are permitted |
| `RT` | Numeric response time in milliseconds, or a missing value |

Rows follow image presentation order. `Block` and `Trial` are optional metadata columns. Scores 1–3 and 5–7 yield negative and positive judgments; score 4 and missing ratings are omitted. Image files are not read by the EEG preprocessing command.

For `--source-format subject-epochs`, provide `subjects/sub-001.npz` through `subjects/sub-060.npz`, or place these files directly in `--source`. Each archive requires unaligned `X` epochs with shape `[N,32,751]`, binary `y` with shape `[N]`, `category` with shape `[N]`, `channels` with shape `[32]`, and scalar `subject`, `split`, and `sfreq`. Use 0/1 for negative/positive judgments, `face`/`landscape` categories, a three-digit subject string such as `"001"`, and a split string of `"train"`, `"val"` or `"test"`. Set `sfreq` to 250 and retain the fixed subject partition in `aestheeg.py`. The importer constructs the four classes and applies subject-level alignment.

## Partitions and alignment

BCI-IV-2a uses subjects 1–5 for training, 6–7 for validation and 8–9 for testing, including both sessions. PhysioNet-MI uses subjects 1–70, 71–89 and 90–109. ISRUC uses subjects 1–80, 81–90 and 91–100. HandMI and SSVEP-EEG use S001–S008, S009–S010 and S011–S012. AesthEEG uses the fixed 40/10/10 subject partition encoded in `aestheeg.py`. HMC uses the bundled recording partition.

MAT's raw reader uses subject IDs 0–27, 28–31 and 32–35 for training, validation and testing. Mumtaz's raw reader excludes `TASK` recordings and sorts healthy and MDD EDF filenames separately. Training uses the first 40 healthy and 42 MDD recordings, validation the next 8 and 10, and testing the remainder. Released LMDB imports preserve the supplied partitions and sample order.

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

The manifest records channels, class indices, sample rate, dimensions, counts, dtypes and `samples_aligned`. The loader applies `A` exactly once when this flag is false. Saved aligned samples retain their original float32 or float64 precision. The model backbone applies common-average referencing and RMS normalization before convolution.
