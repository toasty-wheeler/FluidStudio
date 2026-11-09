//! Export: a worker re-simulates from t=0, the GPU renders each frame (same
//! renderer as the viewport) and a writer thread pipes raw frames into ffmpeg
//! (or writes PNGs).

use crate::bake::total_frames;
use crate::gl::{GpuExport, RenderParams};
use crate::sim::{Emitter, FluidSim, Frame, SimParams};
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Format {
    Mp4,
    WebmAlpha,
    MovProres,
    Png,
}

impl Format {
    pub const ALL: [Format; 4] = [Format::Mp4, Format::WebmAlpha, Format::MovProres, Format::Png];
    pub fn name(self) -> &'static str {
        match self {
            Format::Mp4 => "MP4 video (H.264)",
            Format::WebmAlpha => "WebM video (VP9, supports transparency)",
            Format::MovProres => "MOV for editing (ProRes 4444, supports transparency)",
            Format::Png => "PNG image sequence",
        }
    }
    pub fn needs_ffmpeg(self) -> bool {
        self != Format::Png
    }
    pub fn supports_alpha(self) -> bool {
        self != Format::Mp4
    }
}

/// Saved with the project. `folder` and `ffmpeg` may be relative to the program folder.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ExportSettings {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub transparent: bool,
    pub format: Format,
    pub ffmpeg: String,
    pub folder: String,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            fps: 60,
            transparent: false,
            format: Format::Mp4,
            ffmpeg: String::new(),
            folder: "Exports".into(),
        }
    }
}

#[derive(Default)]
pub struct Progress {
    pub done: AtomicUsize,
    pub total: AtomicUsize,
    pub cancel: AtomicBool,
    pub finished: AtomicBool,
    pub msg: Mutex<String>,
    pub out_path: Mutex<Option<PathBuf>>,
}

fn no_window(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
}

/// Starts the sim + writer threads and returns the GPU job for the paint callback.
pub fn start(params: &SimParams, emitters: &[Emitter], s: ExportSettings, mut rp: RenderParams) -> (Arc<Progress>, GpuExport) {
    let prog = Arc::new(Progress::default());
    let (w, h) = ((s.width.max(16) & !1) as i32, (s.height.max(16) & !1) as i32);
    let total = total_frames(params);
    let step = if s.fps == 30 { 2 } else { 1 };
    prog.total.store(total / step + 1, Ordering::SeqCst);
    *prog.msg.lock().unwrap() = "Rendering…".into();
    rp.transparent = s.transparent && s.format.supports_alpha();

    // simulation thread
    let (ftx, frx) = sync_channel::<Arc<Frame>>(3);
    let (p2, e2, cancel_prog) = (params.clone(), emitters.to_vec(), prog.clone());
    std::thread::spawn(move || {
        let mut sim = FluidSim::new();
        sim.configure(&p2, &e2);
        if ftx.send(Arc::new(sim.snapshot())).is_err() {
            return;
        }
        for f in 1..=total {
            if cancel_prog.cancel.load(Ordering::SeqCst) {
                return;
            }
            sim.step_frame();
            if f % step == 0 && ftx.send(Arc::new(sim.snapshot())).is_err() {
                return;
            }
        }
    });

    // writer thread
    let (ptx, prx) = sync_channel::<Vec<u8>>(3);
    let p3 = prog.clone();
    std::thread::spawn(move || {
        let r = writer(prx, w as u32, h as u32, &s, &p3);
        let cancelled = p3.cancel.load(Ordering::SeqCst);
        *p3.msg.lock().unwrap() = match r {
            Ok(m) if cancelled => format!("Cancelled. {m}"),
            Ok(m) => m,
            Err(e) => e,
        };
        p3.finished.store(true, Ordering::SeqCst);
    });

    (prog, GpuExport { rx: frx, tx: ptx, params: rp, w, h })
}

fn writer(rx: Receiver<Vec<u8>>, w: u32, h: u32, s: &ExportSettings, prog: &Progress) -> Result<String, String> {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let folder = crate::paths::resolve(&s.folder);
    std::fs::create_dir_all(&folder).map_err(|e| format!("Can't create folder: {e}"))?;
    let alpha = s.transparent && s.format.supports_alpha();

    if s.format == Format::Png {
        let dir = folder.join(format!("fluid_{secs}"));
        std::fs::create_dir_all(&dir).map_err(|e| format!("Can't create folder: {e}"))?;
        *prog.out_path.lock().unwrap() = Some(dir.clone());
        let mut k = 0usize;
        for mut px in rx {
            if !alpha {
                px.chunks_exact_mut(4).for_each(|p| p[3] = 255);
            }
            let img = image::RgbaImage::from_raw(w, h, px).ok_or("bad frame size")?;
            img.save(dir.join(format!("frame_{k:05}.png"))).map_err(|e| format!("Writing PNG failed: {e}"))?;
            k += 1;
            prog.done.store(k, Ordering::SeqCst);
        }
        return Ok(format!("Saved {k} PNG frames."));
    }

    let (ext, codec): (&str, &[&str]) = match s.format {
        Format::Mp4 => ("mp4", &["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]),
        Format::WebmAlpha if alpha => ("webm", &["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "22", "-row-mt", "1", "-pix_fmt", "yuva420p"]),
        Format::WebmAlpha => ("webm", &["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "22", "-row-mt", "1", "-pix_fmt", "yuv420p"]),
        Format::MovProres if alpha => ("mov", &["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"]),
        Format::MovProres => ("mov", &["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]),
        Format::Png => unreachable!(),
    };
    let out = folder.join(format!("fluid_{secs}.{ext}"));
    *prog.out_path.lock().unwrap() = Some(out.clone());
    let mut cmd = Command::new(crate::paths::resolve(&s.ffmpeg));
    cmd.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", &format!("{w}x{h}"), "-framerate", &s.fps.to_string(), "-i", "-"])
        .args(codec)
        .arg(&out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    no_window(&mut cmd);
    let mut child = cmd.spawn().map_err(|e| format!("Couldn't start ffmpeg ({}): {e}", s.ffmpeg))?;
    let mut stdin = child.stdin.take().unwrap();
    let mut k = 0usize;
    let mut pipe_err = None;
    for px in rx {
        if let Err(e) = stdin.write_all(&px) {
            pipe_err = Some(e);
            break;
        }
        k += 1;
        prog.done.store(k, Ordering::SeqCst);
    }
    drop(stdin);
    *prog.msg.lock().unwrap() = "Finishing video…".into();
    let o = child.wait_with_output().map_err(|e| e.to_string())?;
    if !o.status.success() || pipe_err.is_some() {
        let err = String::from_utf8_lossy(&o.stderr);
        return Err(format!("ffmpeg failed: {}", err.lines().last().unwrap_or("unknown error")));
    }
    Ok(format!("Saved {} ({k} frames).", out.file_name().unwrap_or_default().to_string_lossy()))
}

/// ffmpeg bundled next to the program first, then PATH, then common spots.
pub fn find_ffmpeg() -> Option<String> {
    if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
        let p = dir.join("ffmpeg.exe");
        if p.exists() {
            return Some(crate::paths::relativize(&p));
        }
    }
    let mut cmd = Command::new("where");
    cmd.arg("ffmpeg");
    no_window(&mut cmd);
    if let Ok(o) = cmd.output() {
        if o.status.success() {
            if let Some(line) = String::from_utf8_lossy(&o.stdout).lines().next() {
                return Some(line.trim().to_string());
            }
        }
    }
    let home = std::env::var("USERPROFILE").unwrap_or_default();
    [
        format!("{home}\\scoop\\apps\\ffmpeg\\current\\bin\\ffmpeg.exe"),
        "C:\\ffmpeg\\bin\\ffmpeg.exe".to_string(),
        "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe".to_string(),
    ]
    .into_iter()
    .find(|p| std::path::Path::new(p).exists())
}
