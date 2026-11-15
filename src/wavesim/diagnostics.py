"""Numerical diagnostics for the finite-difference membrane solver."""
import numpy as np
from .core import Config, simulate


def energy_series(cfg, times, snapshots):
    """Snapshot-based energy proxy; velocities use snapshot intervals, not solver steps.

    Kinetic energy is approximated at midpoints, potential energy at adjacent
    snapshot midpoints. Accurate conservation assessment needs dense snapshots.
    """
    times = np.asarray(times, dtype=float)
    u = np.asarray(snapshots, dtype=float)
    if len(times) < 2 or u.shape != (len(times), cfg.nx, cfg.ny):
        raise ValueError('Snapshots must match times and grid')
    if not np.all(np.diff(times) > 0):
        raise ValueError('Times must be strictly increasing')
    dt = np.diff(times)[:, None, None]
    velocity = np.diff(u, axis=0) / dt
    midpoint = (u[:-1] + u[1:]) / 2
    grad_x = np.diff(midpoint, axis=1) / cfg.dx
    grad_y = np.diff(midpoint, axis=2) / cfg.dy
    kinetic = .5 * cfg.dx * cfg.dy * np.sum(velocity**2, axis=(1, 2))
    potential = .5 * cfg.speed**2 * cfg.dx * cfg.dy * (
        np.sum(grad_x**2, axis=(1, 2)) + np.sum(grad_y**2, axis=(1, 2)))
    return (times[:-1] + times[1:]) / 2, kinetic, potential, kinetic + potential


def mode_exact(cfg, times, amplitude=1.):
    """Undamped, zero-initial-velocity fundamental standing-wave solution."""
    if cfg.damping != 0:
        raise ValueError('Exact reference requires zero damping')
    x = np.linspace(0, cfg.lx, cfg.nx)
    y = np.linspace(0, cfg.ly, cfg.ny)
    spatial = amplitude * np.outer(np.sin(np.pi*x/cfg.lx), np.sin(np.pi*y/cfg.ly))
    omega = np.pi*cfg.speed*np.sqrt(cfg.lx**-2 + cfg.ly**-2)
    return np.cos(omega*np.asarray(times))[:, None, None] * spatial


def convergence_study(grids=(21, 31, 41, 61), duration=.25, speed=1., safety=.7):
    """Compare numerical standing-wave solution with analytic reference.

    Returns grid spacing, final-time relative L2 error, and observed order.
    """
    rows = []
    for n in grids:
        cfg = Config(nx=int(n), ny=int(n), speed=speed, damping=0, safety=safety)
        times, snapshots = simulate(cfg, duration, kind='mode', frames=2)
        exact = mode_exact(cfg, times)
        err = np.linalg.norm((snapshots[-1] - exact[-1])[1:-1, 1:-1])
        scale = np.linalg.norm(exact[-1][1:-1, 1:-1])
        rows.append((int(n), cfg.dx, err / max(scale, 1e-14)))
    result = []
    for i, (n, dx, err) in enumerate(rows):
        order = float('nan') if i == 0 else np.log(rows[i-1][2] / err) / np.log(rows[i-1][1] / dx)
        result.append({'grid': n, 'dx': dx, 'relative_l2_error': err, 'observed_order': order})
    return result
