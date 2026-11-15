# Wave Equation Laboratory — Computational Physics Studio v2.0

Standalone scientific computing application for the two-dimensional damped scalar wave equation. **No financial, machine-learning, or trading integrations.**

## Install (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Features

- Original v0.2 interactive membrane simulator: adjustable wave speed, damping, initial conditions, 3D animation, energy proxy and grid convergence.
- v2.0 computational physics studio: reflecting walls, single/double slits, wave interference with adjustable phase, spatially varying material wave speed, four edge-boundary models, standing-wave modes, Fourier spectra, numerical benchmark and reproducible experiment ZIP.
- Five experiment presets: central pulse, double slit, two sources, material interface and standing mode.
- Numerical and analytical regression tests. Run `python -m pytest -q`.

## Physics

The studio solves

`u_tt + gamma u_t = div(c(x,y)^2 grad u) + f(x,y,t)`

on a uniform square grid with uniform assumed density. The explicit centered-time scheme uses conservative face fluxes and harmonic averages of `c^2`, with a conservative CFL safety factor. Internal barriers use **zero normal flux** (reflecting) rather than clamped displacement. The free outer boundary approximates zero normal flux. Periodic boundaries wrap around. Absorbing boundaries use a **damping sponge**, not a perfectly matched layer. Driving sources add energy, so energy conservation does not apply in those experiments.

The energy chart is calculated from *saved snapshots*, not every integration step; it is a diagnostic proxy that can exhibit sampling error. The Fourier spectrum also uses saved snapshots and can alias frequencies above its Nyquist rate. These distinctions matter when interpreting the results.

## Reproducible research

The studio experiment export includes `settings.json`, `simulation.npz`, `energy.csv`, and `spectrum.csv`. Benchmark CSV provides grid spacing, relative final-time L2 error against the analytical fundamental standing-wave mode, and elapsed runtime. Benchmark timings vary by machine and load.

## Code structure

- `app.py`: Streamlit interface, original and expanded laboratories
- `src/wavesim/core.py`: original constant-speed finite-difference solver
- `src/wavesim/diagnostics.py`: original energy and convergence diagnostics
- `src/wavesim/studio.py`: extended variable-coefficient wave solver, sources, obstacles, boundary models, spectra, benchmarks and presets
- `tests/`: synthetic tests that do not need internet access

## Known limitations

This is an educational scalar wave solver, not a full acoustic, elastic-solid, fluid, or quantum wave model. The 3D plots are membrane displacement surfaces, not volumetric waves. Source forcing is a simplified Gaussian spatial forcing; internal obstacle edges reflect by no-flux conditions. The original v0.2 and studio sections use different models and should not be interpreted as identical solvers. The Fourier-spectral benchmark applies only to the periodic uniform-wave model, not the obstacle or heterogeneous-material experiments.

## Release checklist

1. `python -m pytest -q` passes.
2. Open Streamlit and run each preset.
3. Confirm plots, source controls, energy and spectrum update.
4. Download and inspect experiment ZIP.
5. Run numerical benchmark and inspect its errors.

## v2.3 — MP4-only animation exports

The Fourier-spectral comparison section provides adjustable MP4 export: duration (2–60 seconds), frame rate (12/24/30/60 FPS), playback speed (0.25×/0.5×/1×/2×), and square resolution (360/720/1080). Default is 10 seconds at 30 FPS, 1×, 360×360. The simulation interval is independent of playback duration; 2× repeats the interval twice and 0.5× shows half. Some large combinations are rejected to limit rendering workload. MP4 requires `imageio`, `imageio-ffmpeg`, and Pillow. GIF functionality has been removed.

On Windows run `python -m pytest -q`, then `python -m streamlit run app.py`. Confirm MP4 duration and motion in a media player before publishing.
