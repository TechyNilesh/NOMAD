# Datasets

The 16 benchmark streams are the `.npz` files in this folder (keys `X`, `y`; `y = 1` marks an anomaly).
They come from ADBench (Han et al., NeurIPS 2022, https://github.com/Minqi824/ADBench) and ADRepository
(Pang, 2019, https://github.com/GuansongPang/ADRepository-Anomaly-detection-datasets). Please cite those
sources if you use the data.

| File | File | File | File |
|---|---|---|---|
| `1_ALOI` | `3_backdoor` | `5_campaign` | `8_celeba` |
| `9_census` | `10_cover` | `11_donors` | `13_fraud` |
| `16_http` | `32_shuttle` | `33_skin` | `34_smtp` |
| `48_chess` | `49_kddcup99_prob` | `50_bank` | `51_kddcup99_u2r` |

## INSECTS (controlled drift)

`benchmarks/run_insects_drift.py` expects `INSECTS_Abr.npz`, `INSECTS_Incr.npz`, `INSECTS_IncrGrd.npz` and
`INSECTS_IncrRecr.npz` here, in the same `X`/`y` format. They are not redistributed. Download the INSECTS
streams (Souza et al., DMKD 2020) from the USP DS repository (https://sites.google.com/view/uspdsrepository)
and save each one as an `.npz` with `X` (features) and `y` (1 for the minority/anomalous class).
