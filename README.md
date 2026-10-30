# FedPRISM: Federated Intrusion Detection over Wireless Fading Uplinks

FedPRISM is a federated learning framework for network intrusion detection among wireless edge gateways. Each gateway
keeps its traffic records locally and trains a compact feature-attentive detector with sample-level differential
privacy (DP-SGD with a Rényi accountant). The privatised model updates are sent to the server over orthogonal
block-Rayleigh-fading subchannels with truncated transmission and zero-forcing equalisation, and the server combines
the received updates with a noise-robust integrity-weighted rule:

- cosine agreement of every update with the coordinate-wise median update, normalised by the median agreement, so
  that privacy and channel noise do not penalise honest gateways while sign-flipped or off-consensus updates get zero weight;
- norm bounding to twice the median update norm;
- server momentum.

The code also contains the baselines used for comparison (FedAvg, FedProx, DP-FedAvg, DP-FedProx, coordinate-wise
median, absolute cosine trust), centralised and local-only training, label-flipping and sign-flipping adversaries,
and an error-free-uplink mode.

## Repository layout

| File | Purpose |
|---|---|
| `data.py` | loads NSL-KDD and UNSW-NB15, encodes and scales the features, builds the stratified 70/10/20 split or the official split, and the Dirichlet non-IID partition |
| `fl.py` | detector, DP-SGD local training, fading-uplink model, aggregation rules, privacy accounting and evaluation; one experiment per call |
| `run_queue.py` | runs a list of experiments with bounded GPU concurrency and skips finished ones |
| `experiments/error_free_uplink.txt` | main, ablation, privacy, label-skew, poisoning and official-split runs over an error-free uplink |
| `experiments/fading_uplink.txt` | three-seed comparison, SNR sweep, ablations and poisoning over the Rayleigh-fading uplink |

## Setup

```bash
pip install -r requirements.txt
```

A CUDA GPU is recommended; per-example gradients are computed with `torch.func`.

## Data

Download the datasets from their official sources and place them as follows:

```
data/nslkdd/KDDTrain+.txt
data/nslkdd/KDDTest+.txt
data/unsw/UNSW_NB15_training-set.csv
data/unsw/UNSW_NB15_testing-set.csv
```

- NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset

Preprocessed tensors are cached in `cache/` on first use (`python data.py` builds all of them).

## Usage

Single run, FedPRISM on UNSW-NB15 over a 10 dB mean-SNR fading uplink:

```bash
python fl.py unsw prop_10db channel=rayleigh snr=10 save_probs=1
```

Every hyper-parameter can be overridden as `key=value` (see `DEFAULT` in `fl.py`), for example `sigma=3` (noise
multiplier), `alpha=0.1` (label skew), `mal=0.2` (fraction of compromised gateways), `agg=avg|median|trust|abstrust`,
`chw=1` (channel-quality weighting), `protocol=official`. Results are written to `runs/<dataset>_<tag>.json`.

Full experiment sets:

```bash
python run_queue.py experiments/fading_uplink.txt 4
python run_queue.py experiments/error_free_uplink.txt 4
```

Keep the concurrency at four or below on an 8 GB GPU.

## Results

Test accuracy (%) at 10 dB mean uplink SNR, ten non-IID gateways (Dirichlet α = 0.5), mean ± standard deviation over
three seeds. Private methods use a noise multiplier of 2.0 (worst-case ε = 10.3 on NSL-KDD and ε = 15.0 on
UNSW-NB15, δ = 1e-5).

| Method | NSL-KDD | UNSW-NB15 |
|---|---|---|
| Centralised (non-private) | 99.40 ± 0.04 | 81.93 ± 0.38 |
| Local only (non-private) | 94.49 ± 2.26 | 72.53 ± 1.02 |
| FedAvg | 99.41 ± 0.05 | 80.21 ± 1.29 |
| FedProx | 99.39 ± 0.04 | 80.65 ± 0.94 |
| FedPRISM without DP | 99.45 ± 0.04 | 81.65 ± 0.64 |
| DP-FedAvg | 98.22 ± 0.23 | 78.37 ± 0.83 |
| DP-FedProx | 98.29 ± 0.17 | 78.60 ± 0.78 |
| DP-Median | 98.05 ± 0.05 | 76.84 ± 0.85 |
| **FedPRISM** | 98.32 ± 0.09 | 79.35 ± 0.48 |

## Configuration defaults

10 gateways, 40 rounds, one local epoch, batch size 1024, Adam (learning rate 0.003), clipping norm 1.0, noise
multiplier 2.0, truncation threshold 0.05 on the fading power gain (4.9% outage per gateway and round), mean SNRs
spread over 10 dB around the system mean, server momentum 0.5.

## Citation

The accompanying article is under review. Citation details will be added on publication.
