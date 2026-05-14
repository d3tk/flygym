# CPG v1 vs v2 kinematics comparison - sweep findings

This note summarizes a **v2 actuator-gain sweep** (fixed intrinsic frequency **12 Hz**, coupling strength **10**, duration **1.0 s**, seed **0**) using `scripts/compare_cpg_v1_v2_kinematics.py` and `scripts/sweep_cpg_v1_v2_compare.py`, plus follow-up debugging runs with side-by-side video and endpoint logging enabled.

## Setup

- **v1**: FlyGym 1.0 reference stack (gymnasium-based for MP4); MuJoCo defaults as documented in the compare script.
- **v2**: FlyGym 2; v2 **position actuator gain** (`--actuator-gain`) was swept; other “fair compare” knobs (adhesion, spawn height, warmup, camera style) follow the compare script defaults.

## Findings

### 1. CPG and open-loop commands match

Across all tested gains, **oscillator phase and magnitude** differences (v1 vs v2) were **0**. **Mean L2 joint speed** from the preprogrammed step mapping matched between versions. So for these parameters, the **CPG network and commanded kinematics pipeline** are aligned.

### 2. Constant `ja_v2_concat` vs `ja_v1` RMSE is not a physics regression

**RMSE(ja_v2_concat − ja_v1)** stayed **~1.07 rad** for all gains, with **left legs (lf, lm, lh) ≈ 0** and **right legs (rf, rm, rh) ≈ 1.51 rad**. That split is consistent with a **joint indexing / concatenation order** mismatch in the comparison path, not something actuator gain fixes.

### 3. MuJoCo: v1 forward motion is stable; v2 depends strongly on actuator gain

| v2 gain | v2 thorax Δx (mm) | v2 mean RMSE(cmd vs qpos) [rad] | v2 mean L2 joint vel [rad/s] | v2 mean contact fraction |
|--------:|------------------:|--------------------------------:|-----------------------------:|-------------------------:|
| 40 | −0.18 | 0.183 | 49.2 | 1.00 |
| 60 | 0.37 | 0.146 | 67.1 | 0.94 |
| 80 | 0.39 | 0.125 | 79.1 | 0.90 |
| 100 | 0.38 | 0.112 | 86.6 | 0.92 |
| 120 | 0.83 | 0.097 | 93.9 | 0.85 |
| 140 | **2.06** | **0.090** | **98.8** | 0.89 |

**v1** thorax **Δx ≈ 15.3 mm** was essentially unchanged across the sweep (v1 actuator gain is not what was swept). **v2** shows **low gain → little forward displacement and higher tracking error**; **higher gain → lower cmd–state RMSE and more forward motion** in this window.

### 4. Linear extrapolation of Δx vs gain is unreliable outside the sweep

The sweep script optionally fits **v2 thorax Δx vs actuator gain** and estimates a gain for a target Δx (e.g. **2.5 mm → ~193** from a linear fit over **g = 40 … 140**). A **validation run at ~193** gave **v2 Δx ≈ 9 mm**, so the relationship is **nonlinear**; **do not use that extrapolation** without denser sampling or a better model.

### 5. Practical recommendation from the sweep window

For **f = 12 Hz**, **coupling = 10**, and **1 s** under the compare script’s v2 defaults, **g = 140** was the **best in-sweep** compromise in that table: **lowest v2 tracking RMSE** and **largest v2 forward Δx** without relying on extrapolation.

### 6. Follow-up debug findings after contact/compliance instrumentation

The first sweep hid several lower-level mismatches. The compare script now logs endpoint positions, adhesion commands, tarsus5 contact forces, and v2 contact parameters.

Code fixes / instrumentation added:

- `ContactParams.get_solimp_tuple()` now emits the full five-value MuJoCo `solimp` tuple. Previously the width term was omitted, shifting midpoint/power into the wrong slots in the compiled model.
- `make_locomotion_fly()` now uses v1-like leg compliance by default: actuated leg joints use stiffness/damping **0.05 / 0.06**, while passive tarsi use **7.5 / 0.01**.
- `Simulation.get_ground_contact_info()` now reports a true 0/1 contact-active flag instead of the raw contact-sensor `found` value.
- `compare_cpg_v1_v2_kinematics.py` now defaults v2 ground collision bodies to `legs_only`, matching v1 `Fly(floor_collisions="legs")`, and can run with `--v1-like-contact-params`.
- `compare_cpg_v1_v2_kinematics.py` is now a staged debugger with `--stage openloop|fk|tracking|contact|full`. It writes NPZ traces plus JSON/CSV summaries for all stages, and MP4 for simulator rollout stages by default.

New default diagnostic run (`kp=45`, v2 contact bodies `legs_only`):

| metric | v1 | v2 |
|---|---:|---:|
| thorax Δx over 1 s (mm) | **15.277** | **-0.151** |
| mean RMSE(cmd vs joint state) (rad) | 0.054 | 0.087 |
| mean endpoint relative x/y/z span (mm) | 0.914 / 0.518 / 0.481 | 0.545 / 0.436 / 0.489 |
| mean endpoint z relative to thorax (mm) | -0.864 | -1.190 |
| v2 tarsus5 contact while adhesion on/off | n/a | 0.996 / 1.000 |

Interpretation:

- The CPG state and commanded joint trajectories still match, so the problem is downstream of command generation.
- v2 tracks joints much better after the compliance fix, but default `kp=45` still fails to generate thrust.
- v2 endpoint **x sweep is much smaller** than v1 at default gain, and v2 tarsus5 is effectively in contact even during commanded swing. Per-leg world-z traces show most v2 tarsus5 tips max out around **0.10 mm**, while v1 middle-leg tips reach about **0.55-0.61 mm** in the same rollout.
- Switching v2 to v1-like contact `solref` / `solimp` / friction did **not** materially close the gap (`Δx≈0.055 mm` at `kp=45`), so the next target should be endpoint kinematics / body height / model geometry rather than another contact-parameter sweep.
- At `kp=140`, v2 joint tracking improves (`RMSE≈0.052 rad`) and foot lift increases, but forward displacement is still only **≈1.30 mm** and endpoint spans become very large, so high gain is not a clean v1 reproduction.

Most useful artifacts from this pass:

- `debug_outputs/contact_compliance_debug/default_legs_only_endpoint_after_fixes_kinematics_compare.txt`
- `debug_outputs/contact_compliance_debug/default_legs_only_endpoint_after_fixes.npz`
- `debug_outputs/contact_compliance_debug/default_legs_only_v1like_contact_after_fixes_kinematics_compare.txt`
- `debug_outputs/contact_compliance_debug/default_legs_only_v1like_contact_after_fixes.npz`
- `debug_outputs/contact_compliance_debug/g140_legs_only_endpoint_after_fixes_kinematics_compare.txt`
- `debug_outputs/contact_compliance_debug/g140_legs_only_endpoint_after_fixes.npz`

### 7. Video / MP4 notes

- The compare script emits **`{stem}_side_by_side.mp4`**; avoid putting `_side_by_side` in `--stem` or the filename will duplicate that suffix.
- A **frame count mismatch** (e.g. v1 132 vs v2 124) was reported; the MP4 uses the **shorter** sequence.

### 8. V1-parity staged debugger

Use v1 defaults as the oracle: `dt=1e-4`, duration `1.0 s`, seed `0`, CPG frequency `12 Hz`, coupling `10`, v1 actuator gain `45`, v1 adhesion force `40`, and v1 leg floor collisions.

Stages:

- `openloop`: CPG phases/magnitudes, adhesion commands, joint commands, and v2 DOF reorder checks.
- `fk`: commanded joint poses only, no dynamic stepping; compares thorax-relative tarsus5 trajectories.
- `tracking`: contact disabled (`v2_contact_bodies=none`, v1 floor collisions disabled) to isolate actuator tracking.
- `contact`: fixed-pose v2 contact probe plus CPG contact rollout, including per-leg contact fractions, tarsus5 forces, and whole-leg contact sensor forces.
- `full`: complete walking rollout metrics and side-by-side MP4.

New outputs include per-leg endpoint world traces, thorax-relative endpoint traces, thorax position/orientation/velocity summaries, per-leg swing/stance contact fractions, v2 whole-leg contact force vectors, v2 tarsus5 force vectors, side-by-side MP4s for `tracking`, `contact`, and `full`, and pass/fail threshold summaries in `{stem}_summary.json` and `{stem}_summary.csv`.

Short smoke runs under `debug_outputs/cpg_v1_parity/` reproduced the main failure pattern:

- `tracking`: v1/v2 command tracking is close when contact is disabled.
- `contact`: v2 fixed default pose and CPG rollout both show mean ground-contact flag `1.0`, consistent with ground pinning before parameter tuning.

A direct adhesion diagnostic is decisive:

- Default v2 adhesion gain `40`: v2 thorax Δx `-0.151 mm`, mean contact `0.997`, tarsus5 swing contact `1.000`.
- v2 adhesion gain `0`: v2 thorax Δx `11.178 mm`, mean contact `0.680`, tarsus5 swing contact `0.252`.

This indicates the default v2 failure is primarily adhesion/contact plumbing, not CPG phase generation, FK geometry, or actuator tracking. The root cause was that v2 adhesion actuators used `ctrlrange=(1, 100)` while locomotion controllers send boolean `0/1` commands. This allowed "off" commands to be clamped to `1`, so swing legs could remain adhesive. v1 adhesion actuators allow zero control.

After changing v2 adhesion actuator control range to `0..100`, the default `kp=45`, adhesion gain `40` run improves substantially:

- v1 thorax Δx: `15.277 mm`.
- v2 thorax Δx: `12.780 mm`.
- v2 mean contact: `0.707`.
- v2 endpoint relative x/y/z span: `0.965 / 0.821 / 0.583 mm`.

Remaining mismatch after this fix is no longer "stuck to the ground"; it is gait/detail parity: v2 lateral drift is too high (`Δy≈4.50 mm` vs v1 `2.55 mm`) and endpoint y/z spans are larger than v1.

## Example artifacts (paths on one machine)

- Sweep table: `debug_outputs/sweep_gain/summary.csv`
- MP4 (extrapolated gain, overshoots Δx): `debug_outputs/sweep_gain/g193_f12_c10_target_dx_side_by_side.mp4`
- MP4 (sweep-high gain **140**): `debug_outputs/sweep_gain/g140_f12_c10_side_by_side_side_by_side.mp4` (awkward stem; prefer `--stem g140_f12_c10` for a single `_side_by_side` suffix)

## Reproduce

```bash
# Sweep (example)
uv run python scripts/sweep_cpg_v1_v2_compare.py \
  --output-dir debug_outputs/sweep_gain \
  --gains 40,60,80,100,120,140 \
  --target-dx-v2-mm 2.5

# Single compare with MP4 (requires cpg-v1-compare / gymnasium stack for v1 video)
uv run --extra cpg-v1-compare python scripts/compare_cpg_v1_v2_kinematics.py \
  --output-dir debug_outputs/sweep_gain \
  --stem g140_f12_c10 \
  --duration 1.0 --seed 0 \
  --intrinsic-frequency 12.0 --coupling-strength 10.0 \
  --actuator-gain 140.0

# Focused no-MP4 endpoint/contact diagnostic
uv run --extra cpg-v1-compare python scripts/compare_cpg_v1_v2_kinematics.py \
  --output-dir debug_outputs/contact_compliance_debug \
  --stem default_legs_only_endpoint_after_fixes \
  --duration 1.0 --seed 0 \
  --intrinsic-frequency 12.0 --coupling-strength 10.0 \
  --actuator-gain 45.0 --no-mp4

# V1-parity staged diagnostics
uv run --extra cpg-v1-compare python scripts/compare_cpg_v1_v2_kinematics.py \
  --stage openloop \
  --output-dir debug_outputs/cpg_v1_parity \
  --stem openloop_defaults \
  --duration 1.0 --seed 0 \
  --intrinsic-frequency 12.0 --coupling-strength 10.0 \
  --actuator-gain 45.0

uv run --extra cpg-v1-compare python scripts/compare_cpg_v1_v2_kinematics.py \
  --stage fk \
  --output-dir debug_outputs/cpg_v1_parity \
  --stem fk_defaults \
  --duration 1.0 --seed 0 \
  --intrinsic-frequency 12.0 --coupling-strength 10.0 \
  --actuator-gain 45.0

uv run --extra cpg-v1-compare python scripts/compare_cpg_v1_v2_kinematics.py \
  --stage tracking \
  --output-dir debug_outputs/cpg_v1_parity \
  --stem tracking_defaults \
  --duration 1.0 --seed 0 \
  --intrinsic-frequency 12.0 --coupling-strength 10.0 \
  --actuator-gain 45.0

uv run --extra cpg-v1-compare python scripts/compare_cpg_v1_v2_kinematics.py \
  --stage contact \
  --output-dir debug_outputs/cpg_v1_parity \
  --stem contact_defaults \
  --duration 1.0 --seed 0 \
  --intrinsic-frequency 12.0 --coupling-strength 10.0 \
  --actuator-gain 45.0

uv run --extra cpg-v1-compare python scripts/compare_cpg_v1_v2_kinematics.py \
  --stage contact \
  --output-dir debug_outputs/cpg_v1_parity \
  --stem contact_v2_adhesion0 \
  --duration 1.0 --seed 0 \
  --intrinsic-frequency 12.0 --coupling-strength 10.0 \
  --actuator-gain 45.0 --adhesion-gain 0.0

# Baseline MP4/NPZ comparisons requested for visual acceptance.
uv run --extra cpg-v1-compare python scripts/compare_cpg_v1_v2_kinematics.py \
  --stage full \
  --output-dir debug_outputs/cpg_v1_parity \
  --stem baseline_kp45 \
  --duration 1.0 --seed 0 \
  --intrinsic-frequency 12.0 --coupling-strength 10.0 \
  --actuator-gain 45.0

uv run --extra cpg-v1-compare python scripts/compare_cpg_v1_v2_kinematics.py \
  --stage full \
  --output-dir debug_outputs/cpg_v1_parity \
  --stem baseline_kp140 \
  --duration 1.0 --seed 0 \
  --intrinsic-frequency 12.0 --coupling-strength 10.0 \
  --actuator-gain 140.0
```
