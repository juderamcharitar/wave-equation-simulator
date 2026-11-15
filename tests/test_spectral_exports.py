import numpy as np
import pytest
from wavesim.spectral import initial_field,spectral_solution,finite_difference_periodic,compare_solvers
from wavesim.exports import export_mp4

def test_initial_mode_periodic():
    u=initial_field(32,kind='mode')
    assert u.shape==(32,32)
    assert np.isclose(u[0,0],0)

def test_spectral_mode_analytic():
    u=initial_field(32,kind='mode')
    t=np.array([0,.1,.2])
    got=spectral_solution(u,t)
    omega=2*np.pi*np.sqrt(2)
    assert np.allclose(got[2],u*np.cos(omega*.2),atol=1e-12)

def test_fd_matches_spectral_small_time():
    t,sp,fd,err,stats=compare_solvers(n=48,duration=.2,frames=12)
    assert err[-1]<.025
    assert stats['spectral_seconds']>=0
    assert np.allclose(sp[0],fd[0])

def test_pulse_finite():
    _,sp,fd,err,_=compare_solvers(n=24,duration=.1,frames=4,kind='pulse')
    assert np.isfinite(sp).all() and np.isfinite(fd).all() and np.isfinite(err).all()

def test_invalid_inputs():
    with pytest.raises(ValueError):initial_field(4)
    with pytest.raises(ValueError):compare_solvers(n=4)
    with pytest.raises(ValueError):spectral_solution(np.zeros((16,16)),[0,0])
    with pytest.raises(ValueError):finite_difference_periodic(np.zeros((16,16)),[0,.1],safety=2)

def test_mp4_header():
    pytest.importorskip('imageio_ffmpeg')
    times=np.linspace(0,.2,4)
    frames=np.stack([initial_field(24)*np.cos(t) for t in times])
    payload=export_mp4(times,frames,fps=12,duration=2)
    assert b'ftyp' in payload[:32]
