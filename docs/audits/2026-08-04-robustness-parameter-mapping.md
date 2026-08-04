# Robustness-v2 parameter and 3GPP applicability audit

## Scope

`configs/thesis/confirmatory_robustness_v2.toml` is the only parameter source for
the 22 registered robustness cases. Case IDs are identifiers, not parsers: the
runner reads the explicit numeric fields below and rejects missing fields.

| Factor | Locked cells | Resolved parameter |
|---|---|---|
| estimated large-scale CSI error | `csi0`, `csi05`, `csi10`, `csi20` | relative Gaussian standard deviation `0, 0.05, 0.10, 0.20` |
| residual SIC | `sic_residual0`, `02`, `05`, `10` | residual fraction `0, 0.02, 0.05, 0.10` |
| SIC threshold | `sic_threshold_m3`, `0`, `3`, `5` | `10^(dB/10)` = `0.501187..., 1, 1.995262..., 3.162277...` |
| per-UAV MEC CPU | homogeneous, heterogeneous | `(4,4,4)` and `(2,4,6)` GHz; both total 12 GHz |
| relative deadline | light, default, heavy | fixed `1.0, 0.8, 0.6` s |
| arrivals | periodic, jitter, burst | periods `(2,2,3)` with jitter `0`; the same periods with uniform `-1/0/+1`-slot renewal jitter; rotating-cluster Bernoulli rates `0.9/0.2` with `8--15`-slot switches |
| path loss | Al-Hourani, audited UMa-AV | `al_hourani`; `3gpp_uma_av_audited` |

The CPU comparison preserves aggregate capacity, so it isolates layout rather
than capacity. The periodic and jitter cases have the same mean renewal
intervals. With equal-sized clusters, the burst case has spatial-average
arrival probability `(0.9 + 0.2 + 0.2)/3 = 0.4333`, close to the periodic
mixture's long-run rate `(1/2 + 1/2 + 1/3)/3 = 0.4444`.

## 3GPP source and extracted clauses

Authoritative source: 3GPP TR 36.777 V15.0.0, official archive
`https://ftp.3gpp.org/Specs/archive/36_series/36.777/36777-f00.zip`.

- archive SHA-256: `c32c2f38aeda9ef0005407ebe15edfbb719cf6fdad0f45f3c81224c76fb11d4b`;
- extracted `36777-f00_1.doc` SHA-256: `a1f38ac390efd64c49af65893a1b16d4a552cea9bbb538a1ff40f0d252452fd4`;
- normative location used here: Annex B, Tables B-1 and B-2.

For UMa-AV at aerial height `h=100 m`, Table B-1 gives

```text
p1 = 4300 log10(h) - 3800
d1 = max(460 log10(h) - 700, 18)
P_LOS = 1                                      if d2D <= d1
        d1/d2D + exp(-d2D/p1)(1-d1/d2D)       otherwise.
```

Table B-2 gives, with carrier frequency in GHz and distance in metres,

```text
PL_LOS  = 28.0 + 22 log10(d3D) + 20 log10(fc)
PL_NLOS = -17.5 + (46 - 7 log10(h)) log10(d3D)
          + 20 log10(40 pi fc / 3).
```

The registered geometry uses `h=100 m`, `fc=2 GHz`, and horizontal distances
well below the table's `4 km` limit. Height 100 m is inside the LOS-probability
and NLOS applicability intervals shown by the table.

## Deliberate modelling boundary

TR 36.777 describes a terrestrial base-station--aerial-UE cellular link. This
project applies the large-scale propagation equations by link reciprocity to a
ground-terminal--UAV receiver link, with the UAV as the 100-m aerial endpoint.
It does not reproduce the TR's base-station antenna height, sector pattern,
downtilt, random LOS-state draw, log-normal shadowing, or fast fading.

To remain comparable with the deterministic Al-Hourani main model, the
robustness cell uses the probability-weighted loss in dB,
`P_LOS*PL_LOS + (1-P_LOS)*PL_NLOS`, and converts that loss to a power gain. It
must therefore be described as an **audited UMa-AV large-scale sensitivity
model**, not as a complete 3GPP UMa-AV system simulation. The main experiment
continues to use Al-Hourani Urban; this 3GPP cell is supporting evidence only.
