# V1-Style Random Layouts for FlyGym v2

## Summary
- Add one shared v1-style layout sampler used by both CPU `OdorObstacleNavigationTask` and `WarpNavigationTaskRunner`.
- Keep current base defaults stable; make v1-like behavior opt-in through a training preset.
- Unblock Warp randomized layouts by updating per-env odor targets, obstacle mocap positions, and odor observations during reset.

## Public Interfaces
- Extend `NavigationTaskConfig` with backward-compatible layout controls: optional `obstacle_count_range`, `inactive_obstacle_position`, clearance distances, and `layout_sampling_attempts`.
- Add `active_obstacle_count` and `active_obstacle_positions` to CPU and Warp infos; keep `obstacle_positions` as all configured slots, including inactive parked slots.
- Add training CLI support:
  - `--layout-preset {default,v1}`
  - `--spawn-xy-jitter`, `--spawn-yaw-jitter`
  - optional `--obstacle-count-range MIN MAX`
- Add launch support: `RANDOMIZE_LAYOUT=1` passes `--randomize-layout --layout-preset v1`; spawn jitter env vars default to `0.0` for v1-style runs.

## Implementation Changes
- In `task.py`, introduce an internal `NavigationLayout` result and shared sampler:
  - Fixed layouts return base spawn, odor, obstacles, and active count = slot count.
  - Random layouts sample odor x/y, optional spawn jitter, and either all slots or an inclusive obstacle count range.
  - Count-range layouts park inactive slots at `(1000, 0)` and populate the first active slots.
  - Validate with v1 clearance rules; raise a clear `RuntimeError` after `layout_sampling_attempts`.
- CPU task uses the shared sampler, tracks active obstacle count, and preserves the existing `set_layout` mocap path.
- Warp runner removes the randomize guard, caches odor/obstacle mocap IDs, samples per reset env, updates host layout arrays, writes batched `mjw_data.mocap_pos`, then calls `mjw.forward`.
- Warp `_odor_features` computes inverse-square olfaction from `current_odor_source[env_idx]`, matching `OdorWorld.get_olfaction`, so reward target and olfaction stay aligned.
- `--layout-preset v1` uses v1’s 5 obstacle slots, `obstacle_count_range=(3,5)`, clearances `(10, 3, 4, 4)`, inactive parking `(1000,0)`, and zero spawn jitter unless overridden.

## Tests
- CPU: deterministic seeded layouts, 3-5 active obstacles, inactive parking, clearance constraints, and unchanged default randomize behavior without a count range.
- Shared odor helper: matches `ObstacleOdorWorld.get_olfaction` for fixed-source layouts.
- Warp: randomized runner constructs, seeded resets are deterministic, `reset_at` only changes selected rows, mocap positions match per-env layouts, and per-env odor observations use each env’s target.
- CLI/launch: v1 preset builds the expected task config and `RANDOMIZE_LAYOUT=1` passes the preset args.

## Verification
- `uv run pytest tests/examples/test_odor_obstacle_navigation.py tests/examples/test_odor_obstacle_navigation_training.py`
- With CUDA: `uv run --extra warp pytest tests/warp/test_odor_obstacle_navigation_warp.py`
- Smoke: `uv run --extra warp python scripts/flygym/train_odor_obstacle_navigation.py --total-timesteps 5000 --num-envs 4 --env-device cuda --env-runner warp --randomize-layout --layout-preset v1 --device cuda --no-wandb --run-dir .outputs/flygym/sb3/smoke_random_layout`

## Assumptions
- Preserve the current dirty working tree and edit atop existing local changes.
- Do not change global `NavigationTaskConfig` obstacle defaults; v1-style capacity comes from the preset/config.
- Warp rendering remains unsupported; this work targets training physics, rewards, olfaction observations, and info consistency.
