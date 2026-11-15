import streamlit as st
import numpy as np
import plotly.graph_objects as go
from wavesim import Config, simulate
from wavesim.diagnostics import energy_series, convergence_study

st.set_page_config(page_title='Wave Equation Laboratory', layout='wide', page_icon='🌊')
st.title('Wave Equation Laboratory')
st.caption('A reproducible 2D finite-difference simulation of a damped membrane with fixed boundaries.')
with st.sidebar:
    st.header('Simulation controls')
    nx = st.slider('Grid resolution (per axis)', 21, 151, 81, step=10)
    speed = st.slider('Wave speed', .2, 3., 1., .1)
    damping = st.slider('Damping', 0., 2., .05, .05)
    duration = st.slider('Simulation duration', .2, 5., 2., .2)
    kind = st.selectbox('Initial displacement', ['gaussian','mode','double pulse'])
    amplitude = st.slider('Amplitude', .1, 2., 1., .1)
    width = st.slider('Pulse width (relative)', .03, .2, .09, .01)
    frames = st.slider('Recorded frames', 20, 160, 80, 10)
    st.info('Fixed edges (Dirichlet boundaries); zero initial velocity. Solver timestep obeys the 2D CFL limit.')

@st.cache_data(show_spinner='Solving wave equation…')
def run(nx,speed,damping,duration,kind,amplitude,width,frames):
    cfg = Config(nx=nx,ny=nx,speed=speed,damping=damping)
    return cfg, *simulate(cfg,duration,kind,amplitude,width,frames)

cfg,times,z = run(nx,speed,damping,duration,kind,amplitude,width,frames)
i = st.slider('Time frame', 0,len(times)-1,0)
col1,col2 = st.columns(2)
with col1:
    st.subheader('3D membrane')
    fig=go.Figure(go.Surface(z=z[i].T, x=np.linspace(0,cfg.lx,cfg.nx), y=np.linspace(0,cfg.ly,cfg.ny), colorscale='Viridis', cmin=-amplitude,cmax=amplitude))
    fig.update_layout(scene=dict(zaxis=dict(range=[-amplitude,amplitude]),aspectratio=dict(x=1,y=1,z=.5)),margin=dict(l=0,r=0,b=0,t=20),height=470)
    st.plotly_chart(fig,use_container_width=True)
with col2:
    st.subheader('2D displacement heatmap')
    fig2=go.Figure(go.Heatmap(z=z[i].T,x=np.linspace(0,cfg.lx,cfg.nx),y=np.linspace(0,cfg.ly,cfg.ny),colorscale='RdBu',zmin=-amplitude,zmax=amplitude,zmid=0))
    fig2.update_layout(height=470,margin=dict(l=20,r=20,b=20,t=20),xaxis_title='x',yaxis_title='y')
    st.plotly_chart(fig2,use_container_width=True)
st.metric('Time',f'{times[i]:.4f} s')
st.metric('Stable timestep (maximum)',f'{cfg.dt:.6f} s')
st.metric('Maximum displacement',f'{np.max(np.abs(z[i])):.5f}')
st.subheader('Center-point displacement over time')
center=z[:,nx//2,nx//2]
st.line_chart({'t':times,'displacement':center},x='t',y='displacement')
import io
buf=io.BytesIO(); np.savez_compressed(buf,times=times,displacements=z,dx=cfg.dx,dy=cfg.dy,speed=cfg.speed,damping=cfg.damping)
st.download_button('Download simulation data (.npz)',buf.getvalue(),'wave_simulation.npz','application/octet-stream')
with st.expander('Numerical method and limitations'):
    st.markdown(r'''We solve $u_{tt}+\gamma u_t=c^2(u_{xx}+u_{yy})$ with second-order centered differences in time and space. The timestep is constrained by $c\Delta t\sqrt{1/\Delta x^2+1/\Delta y^2}\leq1$. Fixed zero boundaries represent a membrane clamped along its edges. This is an educational numerical model, not a fluid or ocean-wave simulation. Damping and discretization introduce approximation error.''')


st.divider()
st.header('Advanced physics & numerical diagnostics (v0.2)')
with st.expander('Animated 3D playback', expanded=False):
    st.caption('Play/Pause is handled by Plotly. For responsiveness, the animation is downsampled to at most 40 frames.')
    indices = np.unique(np.linspace(0, len(times)-1, min(40, len(times)), dtype=int))
    xs = np.linspace(0, cfg.lx, cfg.nx)
    ys = np.linspace(0, cfg.ly, cfg.ny)
    anim = go.Figure(data=[go.Surface(z=z[indices[0]].T, x=xs, y=ys, colorscale='Viridis', cmin=-amplitude, cmax=amplitude)])
    anim.frames = [go.Frame(data=[go.Surface(z=z[j].T, x=xs, y=ys, colorscale='Viridis', cmin=-amplitude, cmax=amplitude)], name=str(j)) for j in indices]
    anim.update_layout(height=540, scene=dict(zaxis=dict(range=[-amplitude, amplitude]), aspectratio=dict(x=1,y=1,z=.5)),
        updatemenus=[dict(type='buttons', showactive=False, buttons=[
            dict(label='Play', method='animate', args=[None, {'frame': {'duration': 80, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 0}}]),
            dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate'}])])])
    st.plotly_chart(anim, use_container_width=True)

with st.expander('Energy diagnostics', expanded=True):
    energy_t, kinetic, potential, total = energy_series(cfg, times, z)
    st.line_chart({'time': energy_t, 'kinetic': kinetic, 'potential': potential, 'total': total}, x='time')
    if len(total) > 1 and total[0] > 0:
        st.metric('Snapshot energy change', f'{100*(total[-1]/total[0]-1):+.2f}%')
    st.caption('Energy is estimated from saved snapshots, not every solver step. Sparse frames can distort kinetic energy and apparent conservation. With damping > 0, physical energy should decay; with damping = 0, numerical and sampling errors remain.')

with st.expander('Grid convergence study (analytic standing-wave benchmark)', expanded=False):
    st.caption('Independent undamped fundamental-mode experiment at fixed physical time; not a convergence claim for the selected pulse.')
    if st.button('Run convergence study'):
        import pandas as pd
        with st.spinner('Solving across four grid resolutions...'):
            rows = convergence_study()
        st.dataframe(rows, use_container_width=True)
        st.line_chart(pd.DataFrame(rows), x='dx', y='relative_l2_error')
        st.download_button('Download convergence results (CSV)', pd.DataFrame(rows).to_csv(index=False), 'convergence.csv', 'text/csv')

st.divider()
st.header('Computational Physics Studio (v2.0)')
st.caption('Independent research laboratory for interference, diffraction, variable materials, resonances, boundary conditions, Fourier spectra, and numerical benchmarks.')
from wavesim.studio import StudioConfig, simulate_studio, energy_proxy, spectrum, benchmark, preset, PRESETS
import json, zipfile

with st.form('studio_setup'):
    p=st.selectbox('Experiment preset',PRESETS)
    defaults=preset(p)
    ca,cb,cc=st.columns(3)
    with ca:
        sn=st.select_slider('Studio grid',options=[31,41,61,81,101],value=61)
        sduration=st.slider('Studio duration (s)',.2,3.,1.,.1)
        sbound=st.selectbox('Boundary condition',['fixed','free','periodic','absorbing'],index=['fixed','free','periodic','absorbing'].index(defaults['boundary']))
        sinit=st.selectbox('Initial displacement',['gaussian','mode','zero'],index=['gaussian','mode','zero'].index(defaults['initial']))
    with cb:
        smat=st.selectbox('Material distribution',['uniform','left slow','radial gradient'],index=['uniform','left slow','radial gradient'].index(defaults['material']))
        sbar=st.selectbox('Obstacle',['none','wall','single slit','double slit'],index=['none','wall','single slit','double slit'].index(defaults['barrier']))
        ssrc=st.selectbox('Driving source',['none','single','double'],index=['none','single','double'].index(defaults['source']))
        sspeed=st.slider('Studio wave speed',.2,3.,1.,.1)
    with cc:
        sfreq=st.slider('Source frequency (Hz)',1.,30.,12.,1.)
        sphase=st.slider('Second source phase (radians)',0.,float(2*np.pi),0.,.1)
        sdamp=st.slider('Studio damping',0.,2.,0.,.05)
        smx=st.slider('Mode index x',1,6,1)
        smy=st.slider('Mode index y',1,6,1)
    run_studio=st.form_submit_button('Run studio experiment',type='primary')

@st.cache_data(show_spinner='Computing studio experiment…',max_entries=8)
def studio_run(parameters,duration,initial):
    conf=StudioConfig(**parameters)
    return simulate_studio(conf,duration,frames=80,initial=initial)

if run_studio:
    st.session_state['studio_params']=dict(n=sn,speed=sspeed,damping=sdamp,boundary=sbound,material=smat,barrier=sbar,source=ssrc,frequency=sfreq,phase=sphase,mode_x=smx,mode_y=smy)
    st.session_state['studio_duration']=sduration
    st.session_state['studio_initial']=sinit
if 'studio_params' in st.session_state:
    pars=st.session_state['studio_params']; duration_s=st.session_state['studio_duration']; initial_s=st.session_state['studio_initial']
    scfg=StudioConfig(**pars)
    try:
        tt,zz,cc_field,solids=studio_run(pars,duration_s,initial_s)
        idx=st.slider('Studio frame',0,len(tt)-1,0)
        left,right=st.columns(2)
        with left:
            st.subheader('Displacement and obstacle map')
            fig=go.Figure(go.Heatmap(z=zz[idx].T,colorscale='RdBu',zmid=0))
            if solids.any():
                fig.add_trace(go.Contour(z=solids.T.astype(float),contours=dict(start=.5,end=.5,size=1),showscale=False,line=dict(color='black',width=3)))
            fig.update_layout(height=430,margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig,use_container_width=True)
        with right:
            st.subheader('3D wave surface')
            fig=go.Figure(go.Surface(z=zz[idx].T,colorscale='Viridis'))
            fig.update_layout(height=430,margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig,use_container_width=True)
        st.caption(f'Frame time: {tt[idx]:.4f} s · Spatial grid: {scfg.n}×{scfg.n} · Source forcing: {scfg.source}')
        et,ke,pe=energy_proxy(tt,zz,cc_field,solids,scfg.dx)
        st.subheader('Sampled energy diagnostics')
        st.line_chart({'time':et,'kinetic':ke,'potential':pe,'total':ke+pe},x='time')
        if scfg.source!='none':st.info('Driven experiments add energy; conservation is not expected.')
        if scfg.boundary=='absorbing':st.caption('Absorbing boundaries use a damping sponge, not a perfectly matched layer.')
        st.caption('Snapshot-based energy is approximate; sparse frame sampling can distort apparent conservation.')
        st.subheader('Fourier spectrum at observation point')
        observation=zz[:,scfg.n//2,3*scfg.n//4]
        frequencies,amplitudes=spectrum(tt,observation)
        st.line_chart({'frequency_hz':frequencies,'amplitude':amplitudes},x='frequency_hz')
        st.caption('Finite observation windows limit frequency resolution; source frequencies above the frame sampling Nyquist limit may alias.')
        st.subheader('Reproducible experiment export')
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('settings.json',json.dumps({'configuration':pars,'duration':duration_s,'initial':initial_s,'equation':'u_tt + gamma*u_t = div(c^2 grad u) + forcing','assumptions':'uniform density; conservative face flux; obstacle no-flux; sponge absorbing boundary'},indent=2))
            raw=io.BytesIO();np.savez_compressed(raw,times=tt,displacement=zz,wave_speed=cc_field,obstacles=solids)
            archive.writestr('simulation.npz',raw.getvalue())
            import pandas as pd
            archive.writestr('energy.csv',pd.DataFrame({'time':et,'kinetic':ke,'potential':pe,'total':ke+pe}).to_csv(index=False))
            archive.writestr('spectrum.csv',pd.DataFrame({'frequency_hz':frequencies,'amplitude':amplitudes}).to_csv(index=False))
        st.download_button('Download reproducible experiment ZIP',buffer.getvalue(),'wave_experiment.zip','application/zip')
    except ValueError as exc:st.error(str(exc))

with st.expander('Numerical solver benchmark and analytical comparison'):
    st.caption('Undamped fundamental standing mode on a fixed square. Timings depend on hardware and background load.')
    if st.button('Run solver benchmark'):
        import pandas as pd
        results=pd.DataFrame(benchmark())
        st.dataframe(results,use_container_width=True)
        st.download_button('Download benchmark CSV',results.to_csv(index=False),'wave_benchmark.csv','text/csv')
with st.expander('Model assumptions and interpretation'):
    st.markdown('''**Equation:** $u_{tt}+\\gamma u_t=\\nabla\\cdot(c(x,y)^2\\nabla u)+f(x,y,t)$. Spatial differences use conservative face fluxes and harmonic stiffness means. Explicit centered time integration uses a conservative CFL safety factor. Fixed edges enforce zero displacement; free edges approximate zero normal flux; periodic edges wrap around; absorbing edges use a damping sponge. Internal solid barriers impose zero normal flux (reflecting obstacles), not clamped internal walls. Variable wave speed represents variable stiffness at uniform density. Source signals are continuous forcing, not prescribed displacement. The model is a simplified scalar membrane/wave laboratory, not a full acoustic, elastic-solid, or fluid solver. The FFT of saved frames may alias high frequencies.''')

st.divider()
st.header('Spectral solver & MP4 exports (v2.3)')
st.caption('A separate, mathematically compatible benchmark: uniform undamped waves on a periodic square, with zero initial velocity. It does not apply FFT evolution to barriers or variable materials.')
from wavesim.spectral import compare_solvers
from wavesim.exports import export_mp4
with st.form('spectral_form'):
    c1,c2,c3=st.columns(3)
    with c1: spec_grid=st.selectbox('Periodic grid size',[24,32,48,64,96],index=2)
    with c2: spec_kind=st.selectbox('Spectral initial field',['mode','pulse'])
    with c3: spec_duration=st.slider('Comparison duration',.1,1.0,.4,.1)
    spec_go=st.form_submit_button('Compare FFT spectral and finite differences')
if spec_go:
    st.session_state.pop('mp4_bytes',None)
    st.session_state['spectral_settings']=(spec_grid,spec_duration,spec_kind)
if 'spectral_settings' in st.session_state:
    sn,sd,sk=st.session_state['spectral_settings']
    with st.spinner('Solving both methods…'):
        s_times,s_fft,s_fd,s_error,s_stats=compare_solvers(n=sn,duration=sd,frames=30,kind=sk)
    st.dataframe(s_stats,use_container_width=True)
    st.line_chart({'time':s_times,'relative_l2_error':s_error},x='time')
    sf=st.slider('Comparison frame',0,len(s_times)-1,len(s_times)-1)
    ca,cb=st.columns(2)
    with ca:st.plotly_chart(go.Figure(go.Heatmap(z=s_fft[sf].T,colorscale='RdBu',zmid=0)).update_layout(title='Fourier spectral'),use_container_width=True)
    with cb:st.plotly_chart(go.Figure(go.Heatmap(z=s_fd[sf].T,colorscale='RdBu',zmid=0)).update_layout(title='Finite differences'),use_container_width=True)
    st.caption('The Fourier result is exact in time for this periodic, constant-speed, undamped model (up to floating-point error). The finite-difference result has spatial and temporal discretization error. Benchmarks are hardware-dependent.')
    export_method=st.radio('Animation source',['Spectral comparison','Finite difference comparison'],horizontal=True)
    animation_fields=s_fft if export_method=='Spectral comparison' else s_fd
    st.subheader('Animation export controls')
    ec1,ec2,ec3,ec4=st.columns(4)
    with ec1: video_duration=st.slider('Video duration (seconds)',2,60,10)
    with ec2: fps=st.selectbox('Frames per second',[12,24,30,60],index=2)
    with ec3: playback_speed=st.selectbox('Playback speed',[.25,.5,1.,2.],index=2,format_func=lambda x:f'{x:g}×')
    with ec4: resolution=st.selectbox('Resolution (square pixels)',[360,720,1080],index=0)
    st.caption('Playback duration and speed are independent of simulation duration. At 1×, the full simulated interval is stretched to the chosen video length; 2× loops twice, 0.5× plays half. MP4 exports have a resource budget to prevent accidental memory exhaustion.')
    settings=(sn,sd,sk,export_method,video_duration,fps,playback_speed,resolution)
    if st.session_state.get('export_settings')!=settings:
        st.session_state['export_settings']=settings
        st.session_state.pop('mp4_bytes',None)
    if st.button('Prepare MP4 animation'):
        try:
            with st.spinner('Encoding MP4…'):
                st.session_state['mp4_bytes']=export_mp4(s_times,animation_fields,fps,video_duration,playback_speed,resolution)
        except (RuntimeError,ValueError,OSError,MemoryError) as exc:st.error(f'MP4 encoding failed: {exc}')
    if 'mp4_bytes' in st.session_state:
        st.download_button('Download MP4 animation',st.session_state['mp4_bytes'],'wave_animation.mp4','video/mp4')
