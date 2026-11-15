import numpy as np
import pytest
from wavesim.exports import render_frames, validate_export
from wavesim.spectral import initial_field, spectral_solution

def sample():
    times=np.linspace(0,.3,8)
    frames=spectral_solution(initial_field(24,kind='pulse'),times)
    return times,frames

def test_frame_count_and_resolution():
    t,z=sample()
    frames=list(render_frames(t,z,fps=12,duration=2,resolution=360))
    assert len(frames)==24
    assert frames[0].size==(360,360)

def test_invalid_resource_budget():
    t,z=sample()
    with pytest.raises(ValueError):validate_export(t,z,fps=60,duration=60,resolution=1080)

def test_bad_fps():
    t,z=sample()
    with pytest.raises(ValueError):validate_export(t,z,fps=15)
