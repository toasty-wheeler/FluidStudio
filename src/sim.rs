//! Particle fluid solver: Clavet-style double-density relaxation in 3D,
//! Jacobi-parallelised with rayon. Fully deterministic for a given input.

use glam::{Vec3, Vec4};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

pub const FPS: f64 = 60.0;
pub const POS_SCALE: f32 = 1000.0; // frame cache stores millimetres in i16
const K: usize = 64; // max neighbours per particle
const LIMIT: f32 = 32.0; // particles leaving this box are removed

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SimParams {
    pub duration: f32,
    pub gravity: f32,
    pub h: f32,
    pub stiffness: f32,
    pub near_stiffness: f32,
    pub rest_density: f32,
    pub viscosity: f32,
    pub color_mix: f32,
    pub lifetime: f32,
    pub max_particles: usize,
    pub substeps: u32,
    pub floor: bool,
    pub friction: f32,
    pub container: bool,
    pub container_size: f32,
    /// Half-size of a finite floor (0 = infinite). Set from the ground image.
    pub floor_half: [f32; 2],
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            duration: 8.0,
            gravity: 9.81,
            h: 0.15,
            stiffness: 0.5,
            near_stiffness: 1.0,
            rest_density: 3.0,
            viscosity: 0.1,
            color_mix: 0.3,
            lifetime: 0.0,
            max_particles: 40_000,
            substeps: 3,
            floor: true,
            friction: 0.15,
            container: false,
            container_size: 3.0,
            floor_half: [0.0, 0.0],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Emitter {
    pub id: u32,
    pub name: String,
    pub enabled: bool,
    pub pos: [f32; 3],
    pub yaw: f32,   // degrees
    pub pitch: f32, // degrees
    pub start: f32,
    pub length: f32,
    pub rate: f32, // particles per second
    pub speed: f32,
    pub spread: f32, // full cone angle, degrees
    pub radius: f32, // nozzle radius
    pub color: [u8; 3],
    pub transparency: f32, // 0..100 %
}

impl Default for Emitter {
    fn default() -> Self {
        Self {
            id: 0,
            name: "Point".into(),
            enabled: true,
            pos: [0.0, 1.0, 0.0],
            yaw: 0.0,
            pitch: 40.0,
            start: 0.0,
            length: 2.0,
            rate: 600.0,
            speed: 6.0,
            spread: 8.0,
            radius: 0.08,
            color: [47, 140, 255],
            transparency: 15.0,
        }
    }
}

impl Emitter {
    pub fn dir(&self) -> Vec3 {
        dir_from(self.yaw, self.pitch)
    }
    pub fn origin(&self) -> Vec3 {
        Vec3::from(self.pos)
    }
}

pub fn dir_from(yaw_deg: f32, pitch_deg: f32) -> Vec3 {
    let (y, p) = (yaw_deg.to_radians(), pitch_deg.to_radians());
    Vec3::new(p.cos() * y.sin(), p.sin(), p.cos() * y.cos())
}

pub fn yaw_pitch_from(d: Vec3) -> (f32, f32) {
    let d = d.normalize_or_zero();
    (d.x.atan2(d.z).to_degrees(), d.y.clamp(-1.0, 1.0).asin().to_degrees())
}

/// Compact snapshot of one animation frame.
pub struct Frame {
    pub pos: Vec<[i16; 3]>,
    pub col: Vec<[u8; 4]>,
}

impl Frame {
    pub fn empty() -> Self {
        Self { pos: Vec::new(), col: Vec::new() }
    }
    pub fn len(&self) -> usize {
        self.pos.len()
    }
    pub fn bytes(&self) -> usize {
        self.pos.len() * 10 + 64
    }
}

struct Prep {
    pos: Vec3,
    d: Vec3,
    u: Vec3,
    w: Vec3,
    start: f64,
    end: f64,
    rate: f64,
    speed: f32,
    half_spread: f32,
    radius: f32,
    col: Vec4,
}

/// Rest volume of one particle (equilibrium spacing is ~0.48 h).
pub fn particle_volume(h: f32) -> f32 {
    (0.48 * h).powi(3)
}

/// Stream radius needed so particles leave the nozzle near rest density.
pub fn effective_radius(e: &Emitter, h: f32) -> f32 {
    let need = (e.rate.max(0.0) * particle_volume(h) / (std::f32::consts::PI * e.speed.max(0.3))).sqrt();
    need.max(e.radius)
}

fn prep(e: &Emitter, h: f32) -> Prep {
    let d = e.dir();
    let u = if d.y.abs() > 0.99 { Vec3::X } else { Vec3::new(d.z, 0.0, -d.x).normalize() };
    let w = d.cross(u);
    Prep {
        pos: e.origin(),
        d,
        u,
        w,
        start: e.start as f64,
        end: (e.start + e.length) as f64,
        rate: e.rate.max(0.0) as f64,
        speed: e.speed,
        half_spread: e.spread.to_radians() * 0.5,
        radius: effective_radius(e, h),
        col: Vec4::new(
            e.color[0] as f32 / 255.0,
            e.color[1] as f32 / 255.0,
            e.color[2] as f32 / 255.0,
            (1.0 - e.transparency / 100.0).clamp(0.0, 1.0),
        ),
    }
}

pub struct FluidSim {
    p: SimParams,
    em: Vec<Prep>,
    acc: Vec<f64>,
    seed: u32,
    t: f64,
    frame: u64,
    pub pos: Vec<Vec3>,
    prev: Vec<Vec3>,
    vel: Vec<Vec3>,
    pub col: Vec<Vec4>,
    age: Vec<f32>,
    // scratch
    cell: Vec<[i32; 3]>,
    hash: Vec<u32>,
    bstart: Vec<u32>,
    bcur: Vec<u32>,
    sorted: Vec<u32>,
    nb: Vec<u32>,
    nbc: Vec<u8>,
    press: Vec<f32>,
    press_n: Vec<f32>,
    tmp3: Vec<Vec3>,
    tmp4: Vec<Vec4>,
}

#[inline]
fn hash3(x: i32, y: i32, z: i32, mask: u32) -> u32 {
    ((x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663) ^ (z as u32).wrapping_mul(83492791)) & mask
}

impl FluidSim {
    pub fn new() -> Self {
        Self {
            p: SimParams::default(),
            em: Vec::new(),
            acc: Vec::new(),
            seed: 0x2545_F491,
            t: 0.0,
            frame: 0,
            pos: Vec::new(),
            prev: Vec::new(),
            vel: Vec::new(),
            col: Vec::new(),
            age: Vec::new(),
            cell: Vec::new(),
            hash: Vec::new(),
            bstart: Vec::new(),
            bcur: Vec::new(),
            sorted: Vec::new(),
            nb: Vec::new(),
            nbc: Vec::new(),
            press: Vec::new(),
            press_n: Vec::new(),
            tmp3: Vec::new(),
            tmp4: Vec::new(),
        }
    }

    pub fn configure(&mut self, p: &SimParams, emitters: &[Emitter]) {
        self.p = p.clone();
        self.p.h = self.p.h.max(0.02);
        self.em = emitters.iter().filter(|e| e.enabled).map(|e| prep(e, self.p.h)).collect();
        self.acc = vec![0.0; self.em.len()];
        self.seed = 0x2545_F491;
        self.t = 0.0;
        self.frame = 0;
        self.pos.clear();
        self.prev.clear();
        self.vel.clear();
        self.col.clear();
        self.age.clear();
    }

    pub fn len(&self) -> usize {
        self.pos.len()
    }

    fn rand(&mut self) -> f32 {
        // mulberry32
        self.seed = self.seed.wrapping_add(0x6D2B_79F5);
        let mut t = self.seed;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64 / 4294967296.0) as f32
    }

    fn emit(&mut self, dt: f32) {
        let t = self.t;
        for k in 0..self.em.len() {
            let (start, end, rate) = (self.em[k].start, self.em[k].end, self.em[k].rate);
            if t < start || t >= end {
                continue;
            }
            self.acc[k] += rate * dt as f64;
            while self.acc[k] >= 1.0 {
                self.acc[k] -= 1.0;
                if self.pos.len() >= self.p.max_particles {
                    self.acc[k] = 0.0;
                    break;
                }
                let th = self.em[k].half_spread * self.rand().sqrt();
                let ph = self.rand() * std::f32::consts::TAU;
                let rr = self.em[k].radius * self.rand().sqrt();
                let ph2 = self.rand() * std::f32::consts::TAU;
                let spd = self.em[k].speed * (0.97 + 0.06 * self.rand());
                let lead = spd * dt * self.rand();
                let e = &self.em[k];
                let dir = e.d * th.cos() + (e.u * ph.cos() + e.w * ph.sin()) * th.sin();
                let off = (e.u * ph2.cos() + e.w * ph2.sin()) * rr;
                let p = e.pos + off + dir * lead;
                self.pos.push(p);
                self.prev.push(p);
                self.vel.push(dir * spd);
                self.col.push(e.col);
                self.age.push(0.0);
            }
        }
    }

    fn build_neighbors(&mut self) {
        let n = self.pos.len();
        let h = self.p.h;
        let inv = 1.0 / h;
        let h2 = h * h;
        let mut ts = 1024usize;
        while ts < n * 2 {
            ts <<= 1;
        }
        if self.bstart.len() != ts + 1 {
            self.bstart = vec![0; ts + 1];
            self.bcur = vec![0; ts];
        }
        let mask = (ts - 1) as u32;
        self.cell.resize(n, [0; 3]);
        self.hash.resize(n, 0);
        self.sorted.resize(n, 0);
        self.cell
            .par_iter_mut()
            .zip(self.hash.par_iter_mut())
            .zip(self.pos.par_iter())
            .for_each(|((c, hs), p)| {
                let cc = [(p.x * inv).floor() as i32, (p.y * inv).floor() as i32, (p.z * inv).floor() as i32];
                *c = cc;
                *hs = hash3(cc[0], cc[1], cc[2], mask);
            });
        self.bstart.fill(0);
        for &hs in &self.hash {
            self.bstart[hs as usize + 1] += 1;
        }
        for k in 0..ts {
            self.bstart[k + 1] += self.bstart[k];
        }
        self.bcur.copy_from_slice(&self.bstart[..ts]);
        for i in 0..n {
            let b = self.hash[i] as usize;
            self.sorted[self.bcur[b] as usize] = i as u32;
            self.bcur[b] += 1;
        }

        self.nb.resize(n * K, 0);
        self.nbc.resize(n, 0);
        let (pos, cell, bstart, sorted) = (&self.pos, &self.cell, &self.bstart, &self.sorted);
        self.nb
            .par_chunks_mut(K)
            .zip(self.nbc.par_iter_mut())
            .enumerate()
            .for_each(|(i, (slots, cnt))| {
                let pi = pos[i];
                let c = cell[i];
                let mut m = 0usize;
                'outer: for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            let (nx, ny, nz) = (c[0] + dx, c[1] + dy, c[2] + dz);
                            let b = hash3(nx, ny, nz, mask) as usize;
                            for s in bstart[b]..bstart[b + 1] {
                                let j = sorted[s as usize] as usize;
                                if j == i {
                                    continue;
                                }
                                let cj = cell[j];
                                if cj[0] != nx || cj[1] != ny || cj[2] != nz {
                                    continue;
                                }
                                if (pos[j] - pi).length_squared() < h2 {
                                    slots[m] = j as u32;
                                    m += 1;
                                    if m == K {
                                        break 'outer;
                                    }
                                }
                            }
                        }
                    }
                }
                *cnt = m as u8;
            });
    }

    fn substep(&mut self, dt: f32) {
        self.emit(dt);
        let n = self.pos.len();
        if n == 0 {
            self.t += dt as f64;
            return;
        }
        let p = self.p.clone();
        let g = p.gravity * dt;

        // integrate
        self.pos
            .par_iter_mut()
            .zip(self.prev.par_iter_mut())
            .zip(self.vel.par_iter_mut())
            .zip(self.age.par_iter_mut())
            .for_each(|(((x, q), v), a)| {
                v.y -= g;
                *q = *x;
                *x += *v * dt;
                *a += dt;
            });

        self.build_neighbors();
        let h = p.h;
        let inv_h = 1.0 / h;

        // density -> pressure
        self.press.resize(n, 0.0);
        self.press_n.resize(n, 0.0);
        {
            let (pos, nb, nbc) = (&self.pos, &self.nb, &self.nbc);
            let (k1, k2, r0) = (p.stiffness, p.near_stiffness, p.rest_density);
            self.press
                .par_iter_mut()
                .zip(self.press_n.par_iter_mut())
                .enumerate()
                .for_each(|(i, (pr, pn))| {
                    let pi = pos[i];
                    let (mut s, mut sn) = (0.0f32, 0.0f32);
                    for k in 0..nbc[i] as usize {
                        let j = nb[i * K + k] as usize;
                        let q = (pos[j] - pi).length() * inv_h;
                        if q < 1.0 {
                            let a = 1.0 - q;
                            s += a * a;
                            sn += a * a * a;
                        }
                    }
                    *pr = k1 * (s - r0);
                    *pn = k2 * sn;
                });
        }

        // pressure displacement (Jacobi)
        self.tmp3.resize(n, Vec3::ZERO);
        {
            let (pos, nb, nbc, pr, pn) = (&self.pos, &self.nb, &self.nbc, &self.press, &self.press_n);
            let scale = h * 0.05;
            let (dmax, dmin, total_max) = (h * 0.05, -h * 0.02, h * 0.08);
            self.tmp3.par_iter_mut().enumerate().for_each(|(i, out)| {
                let pi = pos[i];
                let mut acc = Vec3::ZERO;
                for k in 0..nbc[i] as usize {
                    let j = nb[i * K + k] as usize;
                    let e = pos[j] - pi;
                    let r = e.length();
                    if r >= h || r < 1e-7 {
                        continue;
                    }
                    let a = 1.0 - r * inv_h;
                    let d = (((pr[i] + pr[j]) * a + (pn[i] + pn[j]) * a * a) * scale).clamp(dmin, dmax);
                    acc -= e * (d / r);
                }
                let l = acc.length();
                if l > total_max {
                    acc *= total_max / l;
                }
                *out = acc;
            });
        }

        // apply displacement, collide, derive velocity
        {
            let pr_r = h * 0.2;
            let b = p.container_size - pr_r;
            let (floor, container, fric) = (p.floor, p.container, p.friction);
            let [fhx, fhz] = p.floor_half;
            let inv_dt = 1.0 / dt;
            let disp = &self.tmp3;
            self.pos
                .par_iter_mut()
                .zip(self.prev.par_iter())
                .zip(self.vel.par_iter_mut())
                .zip(disp.par_iter())
                .for_each(|(((x, q), v), d)| {
                    *x += *d;
                    let on_floor = fhx <= 0.0 || (x.x.abs() <= fhx && x.z.abs() <= fhz);
                    if floor && on_floor && x.y < pr_r && q.y >= -pr_r {
                        x.y = pr_r;
                        x.x -= (x.x - q.x) * fric;
                        x.z -= (x.z - q.z) * fric;
                    }
                    if container {
                        x.x = x.x.clamp(-b, b);
                        x.z = x.z.clamp(-b, b);
                    }
                    let mut nv = (*x - *q) * inv_dt;
                    let s2 = nv.length_squared();
                    if s2 > 900.0 {
                        nv *= 30.0 / s2.sqrt();
                    }
                    *v = nv;
                });
        }

        // viscosity + colour mixing (Jacobi)
        self.tmp4.resize(n, Vec4::ZERO);
        {
            let (pos, vel, col, nb, nbc) = (&self.pos, &self.vel, &self.col, &self.nb, &self.nbc);
            let xs = p.viscosity * 0.3;
            let radial = 0.3 + p.viscosity * 0.5;
            let mix = p.color_mix * 0.05;
            self.tmp3
                .par_iter_mut()
                .zip(self.tmp4.par_iter_mut())
                .enumerate()
                .for_each(|(i, (nv, nc))| {
                    let (pi, vi, ci) = (pos[i], vel[i], col[i]);
                    let (mut dv, mut sm, mut cm, mut ws) = (Vec3::ZERO, Vec3::ZERO, Vec4::ZERO, 0.0f32);
                    for k in 0..nbc[i] as usize {
                        let j = nb[i * K + k] as usize;
                        let e = pos[j] - pi;
                        let r = e.length();
                        if r >= h || r < 1e-7 {
                            continue;
                        }
                        let a = 1.0 - r * inv_h;
                        let nrm = e / r;
                        let rel = vel[j] - vi;
                        let u = -rel.dot(nrm);
                        if u > 0.0 {
                            dv -= nrm * (a * radial * u * 0.5);
                        }
                        sm += rel * a;
                        cm += (col[j] - ci) * a;
                        ws += a;
                    }
                    let w = ws.max(1.0);
                    *nv = vi + dv / w + sm * (xs / w);
                    *nc = ci + cm * (mix / w);
                });
        }
        std::mem::swap(&mut self.vel, &mut self.tmp3);
        std::mem::swap(&mut self.col, &mut self.tmp4);

        // remove dead / escaped particles
        let life = p.lifetime;
        let mut i = self.pos.len();
        while i > 0 {
            i -= 1;
            let x = self.pos[i];
            let dead = (life > 0.0 && self.age[i] > life)
                || x.y < -5.0
                || x.y > LIMIT
                || x.x.abs() > LIMIT
                || x.z.abs() > LIMIT
                || !x.is_finite();
            if dead {
                self.pos.swap_remove(i);
                self.prev.swap_remove(i);
                self.vel.swap_remove(i);
                self.col.swap_remove(i);
                self.age.swap_remove(i);
            }
        }
        self.t += dt as f64;
    }

    /// Advance exactly one 1/60 s frame.
    pub fn step_frame(&mut self) {
        let sub = self.p.substeps.max(1);
        let dt = (1.0 / (FPS * sub as f64)) as f32;
        for _ in 0..sub {
            self.substep(dt);
        }
        self.frame += 1;
        self.t = self.frame as f64 / FPS;
    }

    pub fn snapshot(&self) -> Frame {
        let q = |v: f32| (v * POS_SCALE).round().clamp(-32767.0, 32767.0) as i16;
        let c = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        Frame {
            pos: self.pos.par_iter().map(|p| [q(p.x), q(p.y), q(p.z)]).collect(),
            col: self.col.par_iter().map(|k| [c(k.x), c(k.y), c(k.z), c(k.w)]).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fountain_is_stable() {
        let mut sim = FluidSim::new();
        let p = SimParams::default();
        let em = vec![
            Emitter { pos: [-2.0, 1.0, 0.0], yaw: 90.0, pitch: 35.0, rate: 1500.0, length: 3.0, ..Default::default() },
            Emitter { pos: [2.0, 1.0, 0.0], yaw: -90.0, pitch: 35.0, rate: 1500.0, length: 3.0, start: 0.5, color: [255, 60, 90], ..Default::default() },
        ];
        sim.configure(&p, &em);
        let t0 = std::time::Instant::now();
        for f in 1..=360 {
            sim.step_frame();
            if f % 30 == 0 {
                let n = sim.len();
                let maxv = sim.vel.iter().map(|v| v.length()).fold(0.0f32, f32::max);
                let meanv = sim.vel.iter().map(|v| v.length()).sum::<f32>() / n.max(1) as f32;
                let meannb = sim.nbc.iter().map(|&c| c as f32).sum::<f32>() / n.max(1) as f32;
                let maxy = sim.pos.iter().map(|p| p.y).fold(0.0f32, f32::max);
                let ext = sim.pos.iter().map(|p| p.x.abs().max(p.z.abs())).fold(0.0f32, f32::max);
                println!(
                    "t={:.1}s n={n} maxv={maxv:.2} meanv={meanv:.2} nb={meannb:.1} maxy={maxy:.2} extent={ext:.2} ({:.1} ms/frame)",
                    f as f32 / 60.0,
                    t0.elapsed().as_secs_f32() * 1000.0 / f as f32
                );
            }
        }
        assert!(sim.len() > 5000);
        assert!(sim.pos.iter().all(|p| p.is_finite()));
    }
}
