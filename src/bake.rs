//! Background thread that simulates ahead and caches every frame so the
//! timeline can be scrubbed and replayed instantly.

use crate::sim::{Emitter, FluidSim, Frame, SimParams, FPS};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

pub struct Cache {
    pub gen: u64,
    pub total: usize,
    pub frames: Vec<Option<Arc<Frame>>>,
    pub stride: usize,
    pub frontier: usize,
    pub latest: Arc<Frame>,
    pub bytes: usize,
    pub budget: usize,
    pub particles: usize,
    pub step_ms: f32,
}

impl Cache {
    fn new() -> Self {
        Self {
            gen: 0,
            total: 0,
            frames: vec![Some(Arc::new(Frame::empty()))],
            stride: 1,
            frontier: 0,
            latest: Arc::new(Frame::empty()),
            bytes: 0,
            budget: 2048 << 20,
            particles: 0,
            step_ms: 0.0,
        }
    }

    /// Frame to show for frame index `f` (nearest cached frame at or before it).
    pub fn get(&self, f: usize) -> Arc<Frame> {
        if f >= self.frontier {
            return self.latest.clone();
        }
        let g = f - f % self.stride;
        self.frames.get(g).and_then(|x| x.clone()).unwrap_or_else(|| self.latest.clone())
    }

    fn push(&mut self, f: usize, fr: Arc<Frame>) {
        self.frontier = f;
        self.latest = fr.clone();
        if f % self.stride == 0 && f < self.frames.len() {
            self.bytes += fr.bytes();
            self.frames[f] = Some(fr);
        }
        // Over budget: keep every other frame (preview gets choppier, never runs out of RAM).
        while self.bytes > self.budget && self.stride < 4096 {
            self.stride *= 2;
            let s = self.stride;
            let mut freed = 0;
            for (i, slot) in self.frames.iter_mut().enumerate() {
                if i % s != 0 {
                    if let Some(x) = slot.take() {
                        freed += x.bytes();
                    }
                }
            }
            self.bytes -= freed;
        }
    }

    pub fn done(&self) -> bool {
        self.frontier >= self.total
    }
}

struct Job {
    gen: u64,
    params: SimParams,
    emitters: Vec<Emitter>,
    total: usize,
}

pub struct Baker {
    tx: mpsc::Sender<Job>,
    pub cache: Arc<Mutex<Cache>>,
}

pub fn total_frames(p: &SimParams) -> usize {
    (p.duration.max(0.1) as f64 * FPS).round() as usize
}

impl Baker {
    pub fn new(ctx: eframe::egui::Context) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let cache = Arc::new(Mutex::new(Cache::new()));
        let c2 = cache.clone();
        std::thread::Builder::new()
            .name("fluid-baker".into())
            .spawn(move || worker(rx, c2, ctx))
            .expect("spawn baker");
        Self { tx, cache }
    }

    pub fn restart(&self, params: &SimParams, emitters: &[Emitter], budget_mb: usize) {
        let total = total_frames(params);
        let gen = {
            let mut c = self.cache.lock().unwrap();
            c.gen += 1;
            c.total = total;
            c.frames = vec![None; total + 1];
            c.frames[0] = Some(Arc::new(Frame::empty()));
            c.stride = 1;
            c.frontier = 0;
            c.latest = Arc::new(Frame::empty());
            c.bytes = 0;
            c.budget = budget_mb.max(64) << 20;
            c.particles = 0;
            c.gen
        };
        let _ = self.tx.send(Job { gen, params: params.clone(), emitters: emitters.to_vec(), total });
    }
}

fn worker(rx: mpsc::Receiver<Job>, cache: Arc<Mutex<Cache>>, ctx: eframe::egui::Context) {
    let mut sim = FluidSim::new();
    let mut job: Option<Job> = None;
    let mut frame = 0usize;
    let mut last_repaint = Instant::now();
    loop {
        // Get the newest job (block only when idle).
        let mut incoming = if job.is_some() {
            match rx.try_recv() {
                Ok(j) => Some(j),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => return,
            }
        } else {
            match rx.recv() {
                Ok(j) => Some(j),
                Err(_) => return,
            }
        };
        while let Ok(j) = rx.try_recv() {
            incoming = Some(j);
        }
        if let Some(j) = incoming {
            sim.configure(&j.params, &j.emitters);
            frame = 0;
            job = Some(j);
        }
        let Some(j) = &job else { continue };
        if frame >= j.total {
            job = None;
            ctx.request_repaint();
            continue;
        }
        let t0 = Instant::now();
        sim.step_frame();
        frame += 1;
        let fr = Arc::new(sim.snapshot());
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        {
            let mut c = cache.lock().unwrap();
            if c.gen != j.gen {
                continue;
            }
            c.push(frame, fr);
            c.particles = sim.len();
            c.step_ms = c.step_ms * 0.9 + ms * 0.1;
        }
        if last_repaint.elapsed().as_millis() > 15 || frame >= j.total {
            ctx.request_repaint();
            last_repaint = Instant::now();
        }
    }
}
