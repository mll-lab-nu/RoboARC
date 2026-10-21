# Harbor on the local 1.2.0 image (pre-release, not shipped)

Image `ghcr.io/embodied-first-exam/roboarc:1.2.0` as built locally = `sha256:301adcb64b06` (renamed since to `roboarc:1.2.0-jianwen-pre`), tasks of RoboARC `d6f2f23` (`25e5a2a`'s render; seed copies by `scripts/task_seed.py materialize`), rendering off (`source/scripts/robotangle/harbor_check.py`), 2026-10-20/21. All hosts are AMD Ryzen 9 7950X (Zen 4).

## new_mll1: the nine new bases (4 of them), host B (RTX 4090, driver 580)

| task | oracle | nop | pass | oracle wall s | nop wall s | oracle end state |
|---|---|---|---|---|---|---|
| roboarc-bed-bolt-01-i01-privileged | 1 | 0 | yes | 1346 | 47 | fa027063506f37b9 (x2) |
| roboarc-bed-bolt-01-i01-standard | 1 | 0 | yes | 1523 | 70 | fa027063506f37b9 (x2) |
| roboarc-bed-bolt-02-i01-privileged | 1 | 0 | yes | 1559 | 46 | 9065337a7d7b18f5 (x2) |
| roboarc-bed-bolt-02-i01-standard | 1 | 0 | yes | 1509 | 72 | 9065337a7d7b18f5 (x2) |
| roboarc-caliper-sorting-01-i01-privileged | 1 | 0 | yes | 1929 | 50 | ec9ade23fc0a034c (x2) |
| roboarc-caliper-sorting-01-i01-standard | 1 | 0 | yes | 2052 | 74 | ec9ade23fc0a034c (x2) |
| roboarc-interlocked-claws-13-i01-privileged | 1 | 0 | yes | 853 | 48 | 66be99300a3ceebd (x2) |
| roboarc-interlocked-claws-13-i01-standard | 1 | 0 | yes | 875 | 72 | 66be99300a3ceebd (x2) |

## new_mll3: the nine new bases (5 of them), host C (RTX 4090, driver 580)

| task | oracle | nop | pass | oracle wall s | nop wall s | oracle end state |
|---|---|---|---|---|---|---|
| roboarc-battery-replacement-01-i01-privileged | 1 | 0 | yes | 535 | 46 | de1cb2ee33e369ed (x2) |
| roboarc-battery-replacement-01-i01-standard | 1 | 0 | yes | 584 | 69 | de1cb2ee33e369ed (x2) |
| roboarc-bed-bolt-03-i01-privileged | 1 | 0 | yes | 972 | 47 | 478283fbcfd6f603 (x2) |
| roboarc-bed-bolt-03-i01-standard | 1 | 0 | yes | 1256 | 74 | 478283fbcfd6f603 (x2) |
| roboarc-bed-bolt-04-i01-privileged | 1 | 0 | yes | 790 | 48 | 496e553fc76d49fb (x2) |
| roboarc-bed-bolt-04-i01-standard | 1 | 0 | yes | 1317 | 72 | 496e553fc76d49fb (x2) |
| roboarc-bed-bolt-05-i01-privileged | 1 | 0 | yes | 739 | 46 | 9ff4432d309ee102 (x2) |
| roboarc-bed-bolt-05-i01-standard | 1 | 0 | yes | 1050 | 67 | 9ff4432d309ee102 (x2) |
| roboarc-clock-assembling-01-i01-privileged | 1 | 0 | yes | 803 | 44 | 9683cab251ece2c0 (x2) |
| roboarc-clock-assembling-01-i01-standard | 1 | 0 | yes | 863 | 70 | 9683cab251ece2c0 (x2) |

## seed_s1_local: seed 1 copies, host A (RTX 4090, driver 570.153): blanket-fold-01 cannot run here (its GPU cloth needs CUDA 13); rerun on host B below

| task | oracle | nop | pass | oracle wall s | nop wall s | oracle end state |
|---|---|---|---|---|---|---|
| roboarc-blanket-fold-01-i01-privileged | 0 | 0 | NO | 89 | 50 |  |
| roboarc-blanket-fold-01-i01-standard | 0 | 0 | NO | 1934 | 95 |  |
| roboarc-hedgehog-in-cage-02-i01-privileged | 1 | 0 | yes | 295 | 47 | d4b3f5e930251e98 (x2) |
| roboarc-hedgehog-in-cage-02-i01-standard | 1 | 0 | yes | 296 | 75 | d4b3f5e930251e98 (x2) |
| roboarc-interlocked-claws-03-i01-privileged | 1 | 0 | yes | 149 | 49 | 6a19a1f3a3e87453 (x2) |
| roboarc-interlocked-claws-03-i01-standard | 1 | 0 | yes | 198 | 84 | 6a19a1f3a3e87453 (x2) |

## seed_s2_local: seed 2 copies, host A (claws-03, hedgehog-02)

| task | oracle | nop | pass | oracle wall s | nop wall s | oracle end state |
|---|---|---|---|---|---|---|
| roboarc-hedgehog-in-cage-02-i01-privileged | 1 | 0 | yes | 250 | 54 | 7517485f9d0eb269 (x2) |
| roboarc-hedgehog-in-cage-02-i01-standard | 1 | 0 | yes | 278 | 86 | 7517485f9d0eb269 (x2) |
| roboarc-interlocked-claws-03-i01-privileged | 1 | 0 | yes | 170 | 54 | 762e709f1a01c81d (x2) |
| roboarc-interlocked-claws-03-i01-standard | 1 | 0 | yes | 190 | 84 | 762e709f1a01c81d (x2) |

## blanket_s1_mll1: blanket-fold-01 seed 1 copies, host B

| task | oracle | nop | pass | oracle wall s | nop wall s | oracle end state |
|---|---|---|---|---|---|---|
| roboarc-blanket-fold-01-i01-privileged | 1 | 0 | yes | 990 | 46 | 8f137e81ca82036b (x2) |
| roboarc-blanket-fold-01-i01-standard | 1 | 0 | yes | 974 | 144 | 8f137e81ca82036b (x2) |

## blanket_s2_mll1: blanket-fold-01 seed 2 copies, host B

| task | oracle | nop | pass | oracle wall s | nop wall s | oracle end state |
|---|---|---|---|---|---|---|
| roboarc-blanket-fold-01-i01-privileged | 1 | 0 | yes | 1212 | 44 | ff4e2fb24ee17e7d (x2) |
| roboarc-blanket-fold-01-i01-standard | 1 | 0 | yes | 1216 | 174 | ff4e2fb24ee17e7d (x2) |

## rt04_s1_local: rope-threading-04 seed 1 copy (a reference without a birth stamp), privileged oracle only, host A

| task | oracle | nop | pass | oracle wall s | nop wall s | oracle end state |
|---|---|---|---|---|---|---|
| roboarc-rope-threading-04-i01-privileged | 1 |  | yes | 3712 | 0 | 25fbd242ba117e87 (x2) |

Totals: the new bases 36 / 36 (oracle 1 at the reference's end state, deterministic; nop 0); seed copies 24 / 24 (claws-03 and hedgehog-02 at seeds 1 and 2 on host A, 16; blanket-fold-01 at seeds 1 and 2 on host B, 8); rope-threading-04 seed 1 privileged oracle 1. The three blanket-fold rows marked NO on host A are the 570 driver ("Warp sees no CUDA device"), not the task.
