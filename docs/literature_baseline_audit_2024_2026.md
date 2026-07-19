# Literature-baseline audit (2024--2026 review window)

## Purpose

This ledger separates three claims that must not be conflated:

1. a definition verified in a cited primary paper;
2. an explicit adaptation required to run on the thesis-v2 model; and
3. behavior that cannot be verified without the original software.

The review window names the thesis literature-update period; older work may be
retained when it is the closest method-level comparator. No row in this audit
is a promise of venue acceptance or a claim that a baseline has already won or
lost an experiment.

## Closest-work implementation ledger

| Baseline | Why retained | Verified from the primary paper | Explicit thesis-v2 adaptation | Unavailable or intentionally omitted behavior | Allowed label |
|---|---|---|---|---|---|
| Guo-JAGP, ICC 2019, [DOI 10.1109/ICC.2019.8761779](https://doi.org/10.1109/ICC.2019.8761779) | It directly couples multi-receiver association, uplink NOMA grouping, QoS-minimal power recursion, and sequential exact-potential best response. | Eq. 5 supplies the required-power recursion; Eqs. 9--10 define own power plus within-group externality; Eq. 11 links unilateral utility differences to total-power differences; Algorithm 1 updates users sequentially until grouping stops changing. | BS/channel groups map to UAV/subcarrier blocks; the implementation imposes a two-user block cap, freezes a capacity-limited set of offload-required tasks, uses planning air-to-ground gains and a declared 0.5 bit/s/Hz target, and accepts only strict total-required-power decreases. | unavailable author code behavior, source-specific initialization/ties/tolerances, arbitrary group sizes, heterogeneous QoS targets, source terrestrial fading realization, and source per-user power execution are not asserted. | “Guo-JAGP adapted” or “closest-work adaptation”; never “reproduction,” “reimplementation,” or “original Guo code.” |

The full component mapping is
[`docs/audits/2026-07-19-adapted-baseline-mapping.md`](audits/2026-07-19-adapted-baseline-mapping.md).

## Fair-evaluation boundary

`guo_jagp_adapted` owns only task selection, association/grouping, the
QoS-required-power potential, and sequential strict best response. It returns
a raw `Schedule`. The shared experiment path then applies exactly the same
canonical planning radio physics, configured CPU policy, policy closure, and
frozen realized-channel replay used for the other end-to-end NOMA methods.

That boundary creates a deliberate two-part interpretation:

- association/grouping decisions retain the closest verified Guo structure;
- completion, deadline, CPU, cross-UAV exposure, residual-SIC, and realized
  replay results belong to thesis-v2 and must not be presented as Guo et al.'s
  reported quantities.

The adaptation's required powers are not passed into canonical replay, whose
uniform configured transmit power is fixed for method comparability. Thus this
baseline supports a fair schedule-level comparison under shared physics, but
does not support source-paper power-consumption replication claims.

## Evidence and limitations status

- Primary-paper definitions: verified against the paper text on 2026-07-20.
- DOI and venue identity: pinned.
- Hand recursion: executable test frozen at `(0.5, 1.0)`.
- Author implementation: not available in the audited repository or paper
  record; behavior beyond the published equations/pseudocode remains unknown.
- Experimental outcome: intentionally not recorded here. Formal experiments
  must consume the registered method and canonical result pipeline.
