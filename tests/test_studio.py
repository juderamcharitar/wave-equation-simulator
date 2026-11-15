import numpy as np
import pytest
from wavesim.studio import StudioConfig,simulate_studio,geometry,spectrum,energy_proxy,benchmark,preset

def test_preset_definitions():
    for name in ('central pulse','double slit','two sources','material interface','standing mode'):
        assert 'initial' in preset(name)

def test_invalid_parameters():
    with pytest.raises(ValueError):StudioConfig(n=5)
    with pytest.raises(ValueError):StudioConfig(boundary='magic')

def test_obstacles_have_apertures():
    cfg=StudioConfig(n=61,barrier='double slit')
    _,_,_,solid=geometry(cfg)
    assert solid.any() and (~solid[30]).any()

def test_variable_material():
    cfg=StudioConfig(material='left slow')
    _,_,c,_=geometry(cfg)
    assert c.min()<c.max()

def test_fixed_edges():
    t,z,c,s=simulate_studio(StudioConfig(n=31),duration=.2,frames=12)
    assert np.all(z[:,:,0]==0) and np.all(z[:,:,-1]==0)
    assert np.all(np.isfinite(z))

def test_source_drives_zero_initial():
    t,z,_,_=simulate_studio(StudioConfig(n=31,source='single'),duration=.3,frames=15,initial='zero')
    assert np.max(abs(z[-1]))>1e-7

def test_periodic_constant_is_stationary():
    from wavesim.studio import _flux
    cfg=StudioConfig(n=31,boundary='periodic')
    _,_,c,s=geometry(cfg)
    assert np.max(abs(_flux(np.ones_like(c),c,s,cfg.dx,'periodic')))<1e-12

def test_energy_nonnegative():
    cfg=StudioConfig(n=31)
    t,z,c,s=simulate_studio(cfg,.2,frames=12)
    et,k,p=energy_proxy(t,z,c,s,cfg.dx)
    assert np.min(k)>=0 and np.min(p)>=0 and len(et)==len(t)-1

def test_spectrum_peak():
    t=np.arange(512)/128
    f,a=spectrum(t,np.sin(2*np.pi*8*t))
    assert abs(f[np.argmax(a)]-8)<.5

def test_benchmark_error_finite():
    rows=benchmark(grids=(21,31),duration=.15)
    assert all(np.isfinite(r['relative_l2_error']) for r in rows)
    assert rows[1]['relative_l2_error']<rows[0]['relative_l2_error']
