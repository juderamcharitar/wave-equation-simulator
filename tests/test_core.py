import numpy as np
import pytest
from wavesim import Config, simulate, initial_displacement, laplacian

def test_invalid_config():
    with pytest.raises(ValueError): Config(nx=2)
    with pytest.raises(ValueError): Config(damping=-1)

def test_cfl():
    c=Config(nx=41,ny=51)
    assert c.speed*c.dt*np.sqrt(1/c.dx**2+1/c.dy**2) <= 1

def test_laplacian_constant_interior():
    c=Config(); a=np.ones((c.nx,c.ny))
    assert np.allclose(laplacian(a,c.dx,c.dy)[1:-1,1:-1],0)

def test_initial_boundaries():
    c=Config()
    for kind in ('gaussian','mode','double pulse'):
        a=initial_displacement(c,kind)
        assert np.all(a[0]==0) and np.all(a[-1]==0)
        assert np.all(a[:,0]==0) and np.all(a[:,-1]==0)

def test_simulation_shape_and_boundaries():
    c=Config(nx=21,ny=25)
    t,z=simulate(c,.2,frames=10)
    assert z.shape==(len(t),21,25)
    assert t[0]==0 and t[-1]==pytest.approx(.2)
    assert np.all(z[:,:,0]==0) and np.all(z[:,:,-1]==0)
    assert np.all(z[:,0,:]==0) and np.all(z[:,-1,:]==0)
    assert np.isfinite(z).all()

def test_zero_amplitude():
    t,z=simulate(Config(nx=21,ny=21),.1,amplitude=0)
    assert np.all(z==0)

def test_mode_short_time_bounded():
    c=Config(nx=41,ny=41,damping=0)
    t,z=simulate(c,.25,kind='mode',frames=20)
    assert np.max(np.abs(z)) <= 1.001

def test_bad_inputs():
    with pytest.raises(ValueError): simulate(Config(),-1)
    with pytest.raises(ValueError): initial_displacement(Config(),'unknown')
