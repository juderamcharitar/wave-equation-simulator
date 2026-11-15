"""Periodic, uniform, undamped wave equation: FFT modal solution and FD comparison.

These methods deliberately use a periodic grid without duplicate endpoints. They are
not interchangeable with the variable-material, obstacle or damped studio solver.
"""
import numpy as np


def initial_field(n=64, length=1.0, kind='mode'):
    if n < 12 or length <= 0:
        raise ValueError('Invalid periodic grid')
    x = np.arange(n) * length / n
    X, Y = np.meshgrid(x, x, indexing='ij')
    if kind == 'mode':
        return np.sin(2*np.pi*X/length)*np.sin(2*np.pi*Y/length)
    if kind == 'pulse':
        dx = (X-.35*length+length/2)%length-length/2
        dy = (Y-.5*length+length/2)%length-length/2
        return np.exp(-(dx*dx+dy*dy)/(2*(.07*length)**2))
    raise ValueError('Unknown initial condition')


def spectral_solution(initial, times, speed=1.0, length=1.0):
    """Exact-in-time Fourier evolution for zero initial velocity, periodic square."""
    u = np.asarray(initial, dtype=float)
    t = np.asarray(times, dtype=float)
    if u.ndim != 2 or u.shape[0] != u.shape[1] or u.shape[0] < 12 or t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0) or t[0] < 0 or speed <= 0 or length <= 0:
        raise ValueError('Invalid spectral simulation input')
    k = 2*np.pi*np.fft.fftfreq(u.shape[0], d=length/u.shape[0])
    omega = speed*np.hypot(k[:,None],k[None,:])
    coeff = np.fft.fft2(u)
    return np.stack([np.fft.ifft2(coeff*np.cos(omega*ti)).real for ti in t])


def finite_difference_periodic(initial, times, speed=1.0, length=1.0, safety=.55):
    """Centered explicit finite differences on the SAME periodic grid."""
    u = np.asarray(initial,dtype=float).copy()
    t = np.asarray(times,dtype=float)
    if u.ndim != 2 or u.shape[0] != u.shape[1] or u.shape[0] < 12 or t.ndim != 1 or len(t)<2 or t[0] != 0 or np.any(np.diff(t)<=0) or speed<=0 or length<=0 or not 0<safety<1:
        raise ValueError('Invalid finite difference simulation input')
    h = length/u.shape[0]
    def lap(a):
        return (np.roll(a,1,0)+np.roll(a,-1,0)+np.roll(a,1,1)+np.roll(a,-1,1)-4*a)/h**2
    max_dt = safety*h/(speed*np.sqrt(2))
    out=[u.copy()]
    velocity=np.zeros_like(u)
    current=0.
    for target in t[1:]:
        steps=int(np.ceil((target-current)/max_dt))
        dt=(target-current)/steps
        # Start each interval with synchronized displacement and velocity.
        for _ in range(steps):
            velocity += .5*dt*speed**2*lap(u)
            u += dt*velocity
            velocity += .5*dt*speed**2*lap(u)
        out.append(u.copy())
        current=target
    return np.stack(out)


def compare_solvers(n=64, duration=.4, frames=25, speed=1., kind='mode'):
    if not 12<=n<=256 or duration<=0 or not 2<=frames<=120 or speed<=0:
        raise ValueError('Invalid comparison parameters')
    import time
    times=np.linspace(0,duration,frames)
    field=initial_field(n,kind=kind)
    start=time.perf_counter(); spectral=spectral_solution(field,times,speed); spectral_s=time.perf_counter()-start
    start=time.perf_counter(); fd=finite_difference_periodic(field,times,speed); fd_s=time.perf_counter()-start
    denom=np.linalg.norm(spectral.reshape(frames,-1),axis=1)
    error=np.linalg.norm((fd-spectral).reshape(frames,-1),axis=1)/np.maximum(denom,1e-12)
    return times,spectral,fd,error,{'spectral_seconds':spectral_s,'finite_difference_seconds':fd_s,'final_relative_l2_error':float(error[-1])}
