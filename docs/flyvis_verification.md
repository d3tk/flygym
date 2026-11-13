# FlyVis checkpoint verification

The pretrained FlyVis network `flow/0000/000` loads the released weights, and its flash responses match the published ON/OFF polarity for 30 of 32 known cell types.

This check uses FlyVis's own flash stimulus. It does not pass FlyGym eye images into the network, so it does not test whether FlyGym ommatidium order matches the FlyVis hex lattice.

## Setup

| | |
| --- | --- |
| Date | 2026-11-12 |
| Host | Raider `raider-viz09`, NVIDIA A40 |
| Job | Slurm `2623771` (`frontier`, `constraint=viz`, `gpu:a40:1`), completed in 35 s, exit 0 |
| Software | FlyVis 1.2.0, PyTorch 2.5.1+cu124, Python 3.12 |
| Checkpoint | `flow/0000/000` under `/p/work1/declank/flygym-flyvis-e2e-401/pretrained` |
| Record | `flyvis-flash-verification.json` in that work directory |

`NetworkView("flow/0000/000").init_network(checkpoint="best")` recovers the network. The release stores that checkpoint twice: `best_chkpt`, and `chkpts/chkpt_00000`. The numbered file is what FlyVis selects as `"best"`, because the validation loss is a single value and `chkpt_00000` is the only numbered checkpoint. The tensor payloads of the two files are the same.

## Weight check

Every recovered parameter matches both checkpoint files. The largest absolute difference is 0. The same parameters differ from a freshly constructed `Network()` by up to 0.439, so the loaded model is the trained checkpoint rather than a new initialization.

## Flash responses

`NetworkView.flash_responses()` ran FlyVis's default flash protocol:

- dynamic range `[0, 1]`
- `t_pre = 1 s`, `t_stim = 1 s`, `dt = 1/200`
- radii `-1` (full field) and `6`
- alternations `(0, 1, 0)`, gray, then the flash, then gray

Responses are the central cell of each type: shape `(1, 4, 600, 65)`, one network, four stimuli, 600 frames, 65 cell types. Activity was finite, from −3.741 to 6.558.

The flash response index (FRI) was computed at radius 6 over the interval from `-dt` to `t_stim`. Responses were shifted so the window was nonnegative, then

```text
FRI = (bright peak − dark peak) / (bright peak + dark peak)
```

A positive FRI is ON (bright-preferring). A negative FRI is OFF (dark-preferring). The expected sign is `flyvis.utils.groundtruth_utils.polarity`, where `1` is ON, `-1` is OFF, and `0` is unknown. Unknown types are reported in the JSON and left out of the correlation.

## Result

The FRI values correlate with the known polarity at **0.810**. The sign matches for **30 of 32** known cell types. L1 is OFF (−0.334), which is the polarity FlyVis uses as the basic check for this stimulus.

| Cell type | Expected | FRI | Match |
| --- | ---: | ---: | --- |
| R1 | +1 | +0.139 | yes |
| R2 | +1 | +0.158 | yes |
| R3 | +1 | +0.217 | yes |
| R4 | +1 | +0.128 | yes |
| R5 | +1 | +0.101 | yes |
| R6 | +1 | +0.091 | yes |
| R7 | +1 | +0.518 | yes |
| R8 | +1 | +0.137 | yes |
| L1 | −1 | −0.334 | yes |
| L2 | −1 | −0.332 | yes |
| L3 | −1 | −0.335 | yes |
| L4 | −1 | −0.009 | yes |
| L5 | +1 | +0.348 | yes |
| C3 | +1 | −0.172 | no |
| CT1(Lo1) | −1 | −0.311 | yes |
| CT1(M10) | +1 | +0.143 | yes |
| Mi1 | +1 | +0.320 | yes |
| Mi4 | +1 | +0.315 | yes |
| Mi9 | −1 | −0.337 | yes |
| T4a | +1 | +0.407 | yes |
| T4b | +1 | +0.431 | yes |
| T4c | +1 | +0.434 | yes |
| T4d | +1 | −0.312 | no |
| T5a | −1 | −0.237 | yes |
| T5b | −1 | −0.325 | yes |
| T5c | −1 | −0.351 | yes |
| T5d | −1 | −0.510 | yes |
| Tm1 | −1 | −0.133 | yes |
| Tm2 | −1 | −0.294 | yes |
| Tm3 | +1 | +0.198 | yes |
| Tm4 | −1 | −0.082 | yes |
| Tm9 | −1 | −0.326 | yes |

C3 and T4d are the two misses. Both are expected ON and responded OFF. L4 matches the OFF sign, but its index is close to zero (−0.009). A single ensemble member is not expected to reproduce every published sign. The run was accepted because the weights match the checkpoint, the responses are finite, L1 is OFF, and the correlation is positive.

## Outside this check

Moving-edge direction selectivity and the optic-flow task loss were not rerun. The checkpoint directory stores a scalar validation loss of 5.137; that value was not recomputed. An earlier GPU smoke test only showed that a moving bar changes network activity. It did not compare responses to known cell polarity.
