import numpy as np
import pytest
from wavesim import Config, simulate
from wavesim.diagnostics import energy_series, mode_exact, convergence_study


def test_exact_mode_initial_condition():
    cfg = Config(nx=21, ny=21, damping=0)
    times, snapshots = simulate(cfg, .1, kind='mode', frames=5)
    np.testing.assert_allclose(snapshots[0], mode_exact(cfg, times)[0], atol=1e-14)


def test_energy_nonnegative_and_finite():
    cfg = Config(nx=21, ny=21, damping=0)
    times, snapshots = simulate(cfg, .1, frames=30)
    t, k, p, e = energy_series(cfg, times, snapshots)
    assert len(t) == len(times)-1
    assert np.all(np.isfinite(e)) and np.all(e >= 0)
    np.testing.assert_allclose(k+p, e)


def test_energy_invalid_times():
    cfg = Config(nx=11, ny=11)
    with pytest.raises(ValueError):
        energy_series(cfg, [0, 0], np.zeros((2, 11, 11)))


def test_mode_exact_rejects_damping():
    with pytest.raises(ValueError):
        mode_exact(Config(damping=.1), [0, .1])


def test_convergence_error_decreases():
    result = convergence_study(grids=(21, 31, 41), duration=.25)
    errors = [r['relative_l2_error'] for r in result]
    assert errors[0] > errors[1] > errors[2]
    assert result[-1]['observed_order'] > 1.5
