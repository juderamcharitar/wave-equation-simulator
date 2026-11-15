"""Bounded scientific animation encoding with independent simulation and playback clocks."""
import numpy as np

FPS_OPTIONS = (12, 24, 30, 60)
RESOLUTIONS = (360, 720, 1080)
MAX_MP4_PIXELS = 900_000_000  # constrain render workload


def validate_export(times, fields, fps=30, duration=10, speed=1., resolution=360):
    t, z = np.asarray(times, dtype=float), np.asarray(fields, dtype=float)
    if z.ndim != 3 or t.ndim != 1 or len(t) != len(z) or len(t) < 2 or min(z.shape[1:]) < 2:
        raise ValueError('Expected at least two 2D simulation frames')
    if not np.isfinite(z).all() or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Nonfinite data or invalid simulation times')
    if fps not in FPS_OPTIONS or resolution not in RESOLUTIONS or not 2 <= duration <= 60 or speed not in (.25,.5,1.,2.):
        raise ValueError('Unsupported export settings')
    count = int(round(fps * duration))
    if count < 2 or count * resolution**2 > MAX_MP4_PIXELS:
        raise ValueError('Export exceeds safe resource budget; reduce duration, FPS, or resolution')
    return t, z, count


def render_frames(times, fields, fps=30, duration=10, speed=1., resolution=360):
    """Resample physical time over a playback window; speed controls loops/slow motion.

    speed=1 traverses the original simulation once across the requested video duration.
    speed=2 traverses it twice; speed=.5 traverses half of it. Looping wraps to start.
    """
    from PIL import Image, ImageDraw
    t, z, count = validate_export(times, fields, fps, duration, speed, resolution)
    # fixed scale prevents artificial amplitude flicker between frames
    scale = max(float(np.max(np.abs(z))), 1e-12)
    total = t[-1] - t[0]
    for i in range(count):
        fraction = (i / count) * speed
        # at normal speed the last video frame is just before the final instant
        sim_t = t[0] + (fraction % 1.) * total
        upper = min(int(np.searchsorted(t, sim_t, side='right')), len(t)-1)
        lower = max(upper-1, 0)
        alpha = (sim_t-t[lower])/(t[upper]-t[lower]) if upper>lower else 0.
        v = np.clip(((1-alpha)*z[lower]+alpha*z[upper])/scale,-1,1)
        # neutral dark background with positive red / negative blue
        r = np.where(v>0, 50+205*v, 50)
        g = 50+30*(1-np.abs(v))
        b = np.where(v<0, 50-205*v, 50)
        rgb=np.stack([r,g,b],axis=-1).astype('uint8')
        im=Image.fromarray(rgb,'RGB').resize((resolution,resolution),Image.Resampling.BILINEAR)
        ImageDraw.Draw(im).text((8,8),f't={sim_t:.3f}s  playback={i/fps:.2f}s',fill=(255,255,255),stroke_width=1,stroke_fill=(0,0,0))
        yield im


def export_mp4(times, fields, fps=30, duration=10, speed=1., resolution=360):
    validate_export(times,fields,fps,duration,speed,resolution)
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError('MP4 export requires imageio and imageio-ffmpeg') from exc
    import tempfile,os
    fd,path=tempfile.mkstemp(suffix='.mp4');os.close(fd)
    try:
        with imageio.get_writer(path,format='FFMPEG',fps=fps,codec='libx264',macro_block_size=16,output_params=['-pix_fmt','yuv420p']) as writer:
            for im in render_frames(times,fields,fps,duration,speed,resolution):
                writer.append_data(np.asarray(im))
        with open(path,'rb') as f:return f.read()
    finally:
        if os.path.exists(path):os.unlink(path)
