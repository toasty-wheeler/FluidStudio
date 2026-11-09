//! Orbit camera and world <-> screen helpers.

use eframe::egui::{pos2, Pos2, Rect};
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use serde::{Deserialize, Serialize};

const W_NEAR: f32 = 0.06;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct OrbitCam {
    pub target: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub dist: f32,
    pub fov: f32,
}

impl Default for OrbitCam {
    fn default() -> Self {
        Self { target: [0.0, 0.6, 0.0], yaw: 0.5, pitch: 0.38, dist: 9.0, fov: 45.0 }
    }
}

impl OrbitCam {
    pub fn eye(&self) -> Vec3 {
        let (y, p) = (self.yaw, self.pitch);
        Vec3::from(self.target) + Vec3::new(p.cos() * y.sin(), p.sin(), p.cos() * y.cos()) * self.dist
    }
    pub fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), Vec3::from(self.target), Vec3::Y)
    }
    pub fn proj(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh_gl(self.fov.to_radians(), aspect.max(0.01), 0.05, 400.0)
    }
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.53, 1.53);
    }
    pub fn pan(&mut self, dx: f32, dy: f32, view_h: f32) {
        let scale = self.dist * (self.fov.to_radians() * 0.5).tan() * 2.0 / view_h.max(1.0);
        let fwd = (Vec3::from(self.target) - self.eye()).normalize();
        let right = fwd.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(fwd);
        let t = Vec3::from(self.target) + (-right * dx + up * dy) * scale;
        self.target = t.into();
    }
    pub fn zoom(&mut self, scroll: f32) {
        self.dist = (self.dist * (-scroll * 0.0015).exp()).clamp(0.3, 150.0);
    }
}

pub struct Xf {
    #[allow(dead_code)]
    pub view: Mat4,
    pub proj: Mat4,
    pub vp: Mat4,
    inv_vp: Mat4,
    pub rect: Rect,
    pub eye: Vec3,
}

impl Xf {
    pub fn new(cam: &OrbitCam, rect: Rect) -> Self {
        let view = cam.view();
        let proj = cam.proj(rect.width() / rect.height().max(1.0));
        let vp = proj * view;
        Self { view, proj, vp, inv_vp: vp.inverse(), rect, eye: cam.eye() }
    }
    fn clip(&self, p: Vec3) -> Vec4 {
        self.vp * p.extend(1.0)
    }
    fn to_screen(&self, c: Vec4) -> Pos2 {
        let n = c.xyz() / c.w;
        pos2(
            self.rect.left() + (n.x * 0.5 + 0.5) * self.rect.width(),
            self.rect.top() + (0.5 - n.y * 0.5) * self.rect.height(),
        )
    }
    pub fn project(&self, p: Vec3) -> Option<Pos2> {
        let c = self.clip(p);
        (c.w >= W_NEAR).then(|| self.to_screen(c))
    }
    /// Project a 3D segment, clipping it against the near plane.
    pub fn segment(&self, a: Vec3, b: Vec3) -> Option<[Pos2; 2]> {
        let (mut ca, mut cb) = (self.clip(a), self.clip(b));
        if ca.w < W_NEAR && cb.w < W_NEAR {
            return None;
        }
        if ca.w < W_NEAR {
            let t = (W_NEAR - ca.w) / (cb.w - ca.w);
            ca = ca + (cb - ca) * t;
        } else if cb.w < W_NEAR {
            let t = (W_NEAR - cb.w) / (ca.w - cb.w);
            cb = cb + (ca - cb) * t;
        }
        Some([self.to_screen(ca), self.to_screen(cb)])
    }
    pub fn ray(&self, p: Pos2) -> (Vec3, Vec3) {
        let x = (p.x - self.rect.left()) / self.rect.width() * 2.0 - 1.0;
        let y = 1.0 - (p.y - self.rect.top()) / self.rect.height() * 2.0;
        let a = self.inv_vp.project_point3(Vec3::new(x, y, -1.0));
        let b = self.inv_vp.project_point3(Vec3::new(x, y, 1.0));
        (a, (b - a).normalize())
    }
    /// Screen pixels per world unit at point p.
    pub fn px_per_unit(&self, p: Vec3) -> f32 {
        let c = self.clip(p);
        self.rect.height() * 0.5 * self.proj.y_axis.y / c.w.max(W_NEAR)
    }
}

pub fn ray_plane(o: Vec3, d: Vec3, point: Vec3, normal: Vec3) -> Option<Vec3> {
    let den = d.dot(normal);
    if den.abs() < 1e-5 {
        return None;
    }
    let t = (point - o).dot(normal) / den;
    (t > 0.0).then(|| o + d * t)
}

pub fn dist_to_segment(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    (a + ab * t - p).length()
}
