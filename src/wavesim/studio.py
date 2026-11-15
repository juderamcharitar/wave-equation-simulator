"""Extended 2-D wave laboratory: physical media, sources, boundaries and experiments.

Solve u_tt + gamma*u_t = div(c(x,y)^2 grad u) + f on a uniform grid.
Density is assumed uniform; variable c represents spatially varying stiffness.
"""
from dataclasses import dataclass
import numpy as np
from .core import Config

BOUNDARIES = ('fixed', 'free', 'periodic', 'absorbing')
PRESETS = ('central pulse', 'double slit', 'two sources', 'material interface', 'standing mode')

@dataclass(frozen=True)
class StudioConfig:
    n: int = 81
    length: float = 1.0
    speed: float = 1.0
    damping: float = 0.0
    safety: float = 0.65
    boundary: str = 'fixed'
    material: str = 'uniform'
    barrier: str = 'none'
    source: str = 'none'
    frequency: float = 12.0
    phase: float = 0.0
    source_amplitude: float = 5.0
    mode_x: int = 1
    mode_y: int = 1
    
    def __post_init__(self):
        if self.n < 11 or self.length <= 0 or self.speed <= 0 or self.damping < 0:
            raise ValueError('Invalid geometry or physical parameters')
        if not 0 < self.safety <= 0.9 or self.boundary not in BOUNDARIES:
            raise ValueError('Invalid safety factor or boundary')
        if self.material not in ('uniform','left slow','radial gradient') or self.barrier not in ('none','wall','single slit','double slit'):
            raise ValueError('Invalid material or obstacle')
        if self.source not in ('none','single','double') or self.frequency <= 0 or self.mode_x < 1 or self.mode_y < 1:
            raise ValueError('Invalid source or mode')

    @property
    def dx(self): return self.length/(self.n-1)


def geometry(cfg):
    x=np.linspace(0,cfg.length,cfg.n)
    X,Y=np.meshgrid(x,x,indexing='ij')
    c=np.full((cfg.n,cfg.n),cfg.speed)
    if cfg.material=='left slow': c[X < cfg.length/2]*=.55
    if cfg.material=='radial gradient':
        r=np.hypot(X-cfg.length/2,Y-cfg.length/2)/(cfg.length/np.sqrt(2))
        c*=.6+.4*r
    solid=np.zeros_like(c,dtype=bool)
    if cfg.barrier!='none':
        # 3-cell-wide vertical barrier with optional apertures
        strip=np.abs(X-cfg.length*.5)<=1.5*cfg.dx
        solid[:]=strip
        if cfg.barrier=='single slit':
            solid[strip & (np.abs(Y-cfg.length*.5)<cfg.length*.075)]=False
        elif cfg.barrier=='double slit':
            aperture=(np.abs(Y-cfg.length*.35)<cfg.length*.045)|(np.abs(Y-cfg.length*.65)<cfg.length*.045)
            solid[strip & aperture]=False
    return X,Y,c,solid


def _flux(u,c,solid,h,boundary):
    """Conservative flux divergence with harmonic face stiffness and zero obstacle flux."""
    a=c*c
    if boundary=='periodic':
        out=np.zeros_like(u)
        for axis in (0,1):
            up=np.roll(u,-1,axis=axis)
            ap=np.roll(a,-1,axis=axis)
            face=2*a*ap/np.maximum(a+ap,1e-30)
            face=np.where(solid|np.roll(solid,-1,axis=axis),0.,face)
            f=face*(up-u)/h**2
            out+=f-np.roll(f,1,axis=axis)
    else:
        out=np.zeros_like(u)
        for axis in (0,1):
            u1=np.take(u,range(1,u.shape[axis]),axis=axis)
            u0=np.take(u,range(u.shape[axis]-1),axis=axis)
            a1=np.take(a,range(1,a.shape[axis]),axis=axis)
            a0=np.take(a,range(a.shape[axis]-1),axis=axis)
            s1=np.take(solid,range(1,solid.shape[axis]),axis=axis)
            s0=np.take(solid,range(solid.shape[axis]-1),axis=axis)
            face=2*a0*a1/np.maximum(a0+a1,1e-30)
            f=np.where(s0|s1,0.,face*(u1-u0)/h**2)
            lo=[slice(None)]*2; hi=[slice(None)]*2
            lo[axis]=slice(0,-1); hi[axis]=slice(1,None)
            out[tuple(lo)]+=f; out[tuple(hi)]-=f
    out[solid]=0
    return out


def _enforce(u,solid,boundary):
    u[solid]=0.
    if boundary=='fixed':
        u[0,:]=0;u[-1,:]=0;u[:,0]=0;u[:,-1]=0
    return u


def _forcing(cfg,X,Y,t):
    if cfg.source=='none':return np.zeros_like(X)
    width=.025*cfg.length
    p=np.exp(-((X-.2*cfg.length)**2+(Y-.5*cfg.length)**2)/(2*width**2))
    wave=np.sin(2*np.pi*cfg.frequency*t)
    if cfg.source=='double':
        q=np.exp(-((X-.2*cfg.length)**2+(Y-.7*cfg.length)**2)/(2*width**2))
        p=p*wave+q*np.sin(2*np.pi*cfg.frequency*t+cfg.phase)
    else:p*=wave
    return cfg.source_amplitude*p


def simulate_studio(cfg,duration=1.0,frames=60,initial='gaussian'):
    if duration<=0 or not 2<=frames<=200:raise ValueError('Invalid duration or frame count')
    X,Y,c,solid=geometry(cfg)
    dtmax=cfg.safety*cfg.dx/(np.sqrt(2)*np.max(c))
    steps=int(np.ceil(duration/dtmax)); dt=duration/steps
    if steps>20000:raise ValueError('Too many steps; reduce duration or resolution')
    if initial=='gaussian':
        u=np.exp(-((X-.25*cfg.length)**2+(Y-.5*cfg.length)**2)/(2*(.06*cfg.length)**2))
    elif initial=='mode':
        u=np.sin(cfg.mode_x*np.pi*X/cfg.length)*np.sin(cfg.mode_y*np.pi*Y/cfg.length)
    elif initial=='zero':u=np.zeros_like(X)
    else:raise ValueError('Unknown initial condition')
    u=_enforce(u,solid,cfg.boundary)
    a=_flux(u,c,solid,cfg.dx,cfg.boundary)+_forcing(cfg,X,Y,0)
    prev=_enforce(u+.5*dt*dt*a,solid,cfg.boundary)
    selected=set(np.unique(np.linspace(0,steps,min(frames,steps+1),dtype=int))[1:])
    times=[0.]; snaps=[u.copy()]; energies=[]
    # first-order absorbing sponge, clearly distinguished from exact ABC/PML
    if cfg.boundary=='absorbing':
        d=np.minimum.reduce([X,cfg.length-X,Y,cfg.length-Y])
        sponge=18*cfg.speed/cfg.length*np.maximum(0,1-d/(.16*cfg.length))**2
    else:sponge=np.zeros_like(u)
    for k in range(1,steps+1):
        lap=_flux(u,c,solid,cfg.dx,cfg.boundary)
        g=cfg.damping+sponge
        aa=.5*dt*g
        nxt=(2*u-(1-aa)*prev+dt*dt*(lap+_forcing(cfg,X,Y,(k-1)*dt)))/(1+aa)
        nxt=_enforce(nxt,solid,cfg.boundary)
        prev,u=u,nxt
        if k in selected:
            times.append(k*dt);snaps.append(u.copy())
    return np.asarray(times),np.asarray(snaps),c,solid


def energy_proxy(times,snaps,c,solid,dx):
    """Snapshot-sampled energy estimate; sparse sampling causes kinetic aliasing."""
    v=np.diff(snaps,axis=0)/np.diff(times)[:,None,None]
    mid=(snaps[:-1]+snaps[1:])*.5
    gx=np.diff(mid,axis=1)/dx; gy=np.diff(mid,axis=2)/dx
    fx=(c[:-1,:]**2+c[1:,:]**2)*.5
    fy=(c[:,:-1]**2+c[:,1:]**2)*.5
    mx=~(solid[:-1,:]|solid[1:,:]);my=~(solid[:,:-1]|solid[:,1:])
    kinetic=.5*dx**2*np.sum(v*v,axis=(1,2))
    potential=.5*dx**2*(np.sum(fx[None]*gx*gx*mx[None],axis=(1,2))+np.sum(fy[None]*gy*gy*my[None],axis=(1,2)))
    return (times[:-1]+times[1:])/2,kinetic,potential


def spectrum(times,signal):
    times=np.asarray(times); signal=np.asarray(signal)
    if len(times)<4:raise ValueError('Need at least four samples')
    dt=np.diff(times)
    if not np.allclose(dt,dt[0],rtol=1e-5,atol=1e-10):
        signal=np.interp(np.linspace(times[0],times[-1],len(times)),times,signal)
        dt=np.array([(times[-1]-times[0])/(len(times)-1)])
    freq=np.fft.rfftfreq(len(signal),dt[0]); amp=np.abs(np.fft.rfft((signal-np.mean(signal))*np.hanning(len(signal))))
    return freq,amp


def benchmark(grids=(31,51,71),duration=.25):
    import time
    rows=[]
    for n in grids:
        cfg=StudioConfig(n=n,source='none',boundary='fixed',material='uniform',barrier='none')
        start=time.perf_counter()
        times,z,_,_=simulate_studio(cfg,duration,frames=2,initial='mode')
        elapsed=time.perf_counter()-start
        X,Y,_,_=geometry(cfg)
        omega=np.pi*cfg.speed*np.sqrt(2)/cfg.length
        exact=np.sin(np.pi*X/cfg.length)*np.sin(np.pi*Y/cfg.length)*np.cos(omega*times[-1])
        err=np.linalg.norm(z[-1]-exact)/max(np.linalg.norm(exact),1e-12)
        rows.append({'grid':n,'dx':cfg.dx,'relative_l2_error':float(err),'elapsed_seconds':elapsed})
    return rows


def preset(name):
    if name=='central pulse':return dict(initial='gaussian',barrier='none',source='none',material='uniform',boundary='fixed')
    if name=='double slit':return dict(initial='zero',barrier='double slit',source='single',material='uniform',boundary='absorbing')
    if name=='two sources':return dict(initial='zero',barrier='none',source='double',material='uniform',boundary='absorbing')
    if name=='material interface':return dict(initial='gaussian',barrier='none',source='none',material='left slow',boundary='absorbing')
    if name=='standing mode':return dict(initial='mode',barrier='none',source='none',material='uniform',boundary='fixed')
    raise ValueError('Unknown preset')
