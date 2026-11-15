"""Finite-difference solver for a damped 2D wave equation on a rectangular membrane."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Config:
    nx: int = 81
    ny: int = 81
    lx: float = 1.0
    ly: float = 1.0
    speed: float = 1.0
    damping: float = 0.05
    safety: float = 0.85

    def __post_init__(self):
        if min(self.nx, self.ny) < 3 or min(self.lx, self.ly, self.speed) <= 0:
            raise ValueError('Grid sizes must be >= 3 and lengths and speed positive')
        if self.damping < 0 or not 0 < self.safety <= 1:
            raise ValueError('Damping must be nonnegative and safety in (0,1]')

    @property
    def dx(self): return self.lx / (self.nx - 1)
    @property
    def dy(self): return self.ly / (self.ny - 1)
    @property
    def dt(self):
        return self.safety / (self.speed * np.sqrt(1/self.dx**2 + 1/self.dy**2))


def initial_displacement(cfg: Config, kind='gaussian', amplitude=1., width=.09):
    x = np.linspace(0, cfg.lx, cfg.nx)
    y = np.linspace(0, cfg.ly, cfg.ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    if kind == 'gaussian':
        z = amplitude * np.exp(-((X-cfg.lx/2)**2+(Y-cfg.ly/2)**2)/(2*(width*min(cfg.lx,cfg.ly))**2))
    elif kind == 'mode':
        z = amplitude*np.sin(np.pi*X/cfg.lx)*np.sin(np.pi*Y/cfg.ly)
    elif kind == 'double pulse':
        w = width*min(cfg.lx,cfg.ly)
        z = amplitude*(np.exp(-((X-.3*cfg.lx)**2+(Y-.5*cfg.ly)**2)/(2*w*w)) - np.exp(-((X-.7*cfg.lx)**2+(Y-.5*cfg.ly)**2)/(2*w*w)))
    else: raise ValueError('Unknown initial condition')
    z[[0,-1],:] = 0
    z[:,[0,-1]] = 0
    return z


def laplacian(u, dx, dy):
    out = np.zeros_like(u)
    out[1:-1,1:-1] = ((u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1])/dx**2 +
                         (u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])/dy**2)
    return out


def simulate(cfg: Config, duration: float, kind='gaussian', amplitude=1., width=.09, frames=100):
    """Return (times, snapshots), including t=0. Fixed zero-displacement edges.

    Central differences in time and space; initial velocity zero. Damping via
    centered u_t discretization. Snapshots are sampled from solver steps.
    """
    if duration <= 0 or frames < 2 or frames > 1000:
        raise ValueError('duration must be positive and frames between 2 and 1000')
    nsteps = max(1, int(np.ceil(duration/cfg.dt)))
    dt = duration/nsteps
    u0 = initial_displacement(cfg, kind, amplitude, width)
    acc = cfg.speed**2*laplacian(u0, cfg.dx, cfg.dy)
    prev = u0 + .5*dt**2*acc  # u(-dt) with zero initial velocity
    curr = u0.copy()
    indices = np.unique(np.linspace(0,nsteps,min(frames,nsteps+1),dtype=int))
    snaps = [curr.copy()]
    times = [0.]
    selected = set(indices[1:].tolist())
    a = cfg.damping*dt/2
    for k in range(1,nsteps+1):
        nxt = (2*curr - (1-a)*prev + dt**2*cfg.speed**2*laplacian(curr,cfg.dx,cfg.dy))/(1+a)
        nxt[[0,-1],:] = 0
        nxt[:,[0,-1]] = 0
        prev,curr = curr,nxt
        if k in selected:
            snaps.append(curr.copy()); times.append(k*dt)
    return np.array(times), np.stack(snaps)
