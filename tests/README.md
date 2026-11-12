Install the test dependencies with:

```sh
uv sync --extra dev --extra examples --extra rl --extra warp
```

Run the full suite on a machine with an NVIDIA CUDA GPU and working offscreen
OpenGL rendering:

```sh
uv run --no-sync pytest tests/ -n 4 --tb=short
```

Without CUDA, run the CPU suite and tutorials, plus the Warp tests that can use
its CPU backend:

```sh
uv run --no-sync pytest tests/ -m 'not warp' -n 4 --tb=short
uv run --no-sync pytest tests/warp/ -k 'not TestCheckGpu' --no-cov --tb=short
```

Tutorials require `ffmpeg` on `PATH`. Large model meshes are downloaded from
EPFL on first use; the first run can take several minutes. Rendering needs a
working OpenGL context (Linux CI uses EGL/Mesa). `SKIP_RENDERING_TESTS=1` skips
native rendering tests and tutorials when no context is available.

The Warp CPU run checks API and kernel compatibility, but does not validate
CUDA execution, graph capture, or GPU performance. The two `warp` tutorial
notebooks still need a CUDA machine.

Check the documentation and its generated WASM model assets with:

```sh
uv run --no-sync properdocs build
```

MuJoCo 4 upgrade validation (2026-11-12, macOS ARM64, Python 3.14.8):

- Core/examples/packaging/native rendering: 643 passed, 3 skipped. The two
  rendering checks skipped by their Linux-only markers also passed when called
  directly on this Mac; the remaining skip tests the missing Warp extra.
- Warp simulation, recording, rendering, and utility tests on CPU: 85 passed.
- Non-CUDA tutorial notebooks: 9 passed with `ffmpeg` on `PATH`.
- Documentation build: passed (two existing docstring warnings).
- WASM viewer and game models: compiled and stepped 100 times each using
  MuJoCo 4.0.1, with finite state and advancing simulation time.
- Dependency consistency, lockfile, and lint/format checks for changed Python
  files: passed.

CUDA execution, graph capture, GPU performance, the two CUDA tutorials, and
other operating systems/Python versions require separate validation.
