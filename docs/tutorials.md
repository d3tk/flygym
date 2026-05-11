# Tutorials

If you are new to FlyGym, we recommend that you get started with tutorials under the "Tutorials" section in the navigation bar.

## Ported FlyGym 1.x tutorials (4–14)

The original FlyGym 1.x / Gymnasium tutorials live in `.flygym-1.0-ref/notebooks`
for historical comparison. Tutorials **4–14** in this repository are **FlyGym
2.x** ports: they import `flygym` and `flygym_demo.examples` (controllers,
worlds, sensory helpers) and use the direct v2 simulation loop (`Simulation`,
`run_closed_loop`, …).

The legacy Gymnasium API remains available as the separate
[`flygym-gymnasium`](https://github.com/NeLy-EPFL/flygym-gymnasium) package; see
[FlyGym v1 vs. v2 API](migration.md).
