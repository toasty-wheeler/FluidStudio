#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod bake;
mod bg;
mod export;
mod gl;
mod paths;
mod sim;
mod view;

use bake::Baker;
use bg::{BackMode, BgSettings, Fit, FloorMode};
use eframe::egui::{
    self, pos2, vec2, Align, Align2, Button, Color32, CursorIcon, DragValue, FontId, Key, Layout, Modifiers, PointerButton, Pos2,
    Rect, RichText, ScrollArea, Sense, Shape, Slider, Stroke, Ui,
};
use eframe::{egui_glow, glow};
use export::{ExportSettings, Format, Progress};
use glam::Vec3;
use image::RgbaImage;
use serde::{Deserialize, Serialize};
use sim::{Emitter, SimParams, FPS};
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use view::{dist_to_segment, ray_plane, OrbitCam, Xf};

const AIM_LEN: f32 = 1.0;
const AXIS_PX: f32 = 70.0;
const PLAYHEAD: Color32 = Color32::from_rgb(255, 196, 0);
const PALETTE: [[u8; 3]; 8] = [
    [30, 120, 255],
    [255, 50, 90],
    [40, 200, 100],
    [255, 170, 20],
    [150, 80, 255],
    [0, 190, 200],
    [240, 240, 240],
    [40, 40, 40],
];

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
struct Project {
    params: SimParams,
    emitters: Vec<Emitter>,
    bg: BgSettings,
    cam: OrbitCam,
    droplet_scale: f32,
    smooth: f32,
    refraction: f32,
    shine: f32,
    show_helpers: bool,
    show_floor: bool,
    show_grid: bool,
    loop_play: bool,
    cache_mb: usize,
    next_id: u32,
    export: ExportSettings,
}

impl Default for Project {
    fn default() -> Self {
        let emitters = vec![
            Emitter {
                id: 1,
                name: "Blue jet".into(),
                pos: [-2.2, 1.0, 0.0],
                yaw: 90.0,
                pitch: 35.0,
                start: 0.0,
                length: 2.5,
                color: PALETTE[0],
                ..Default::default()
            },
            Emitter {
                id: 2,
                name: "Pink jet".into(),
                pos: [2.2, 1.0, 0.3],
                yaw: -90.0,
                pitch: 35.0,
                start: 1.0,
                length: 2.5,
                color: PALETTE[1],
                ..Default::default()
            },
        ];
        Self {
            params: SimParams::default(),
            emitters,
            bg: BgSettings::default(),
            cam: OrbitCam::default(),
            droplet_scale: 1.0,
            smooth: 1.0,
            refraction: 0.04,
            shine: 1.0,
            show_helpers: true,
            show_floor: true,
            show_grid: false,
            loop_play: false,
            cache_mb: 2048,
            next_id: 3,
            export: ExportSettings::default(),
        }
    }
}

#[derive(Clone, Copy)]
enum Hit {
    Axis(u32, usize),
    Tip(u32),
    Base(u32),
}

#[derive(Clone, Copy)]
enum Drag {
    None,
    Orbit,
    Pan,
    Axis { id: u32, axis: usize, start: Vec3, press: Pos2 },
    Plane { id: u32, y: f32, offset: Vec3 },
    Aim { id: u32, plane_p: Vec3, plane_n: Vec3 },
}

#[derive(Clone, Copy)]
struct TlDrag {
    id: u32,
    mode: u8, // 0 move, 1 left edge, 2 right edge
    s0: f32,
    l0: f32,
    px: f32,
}

struct App {
    proj: Project,
    selected: Option<u32>,
    add_mode: bool,
    new_height: f32,
    playing: bool,
    playhead: f32,
    baker: Baker,
    applied_key: String,
    seen_key: String,
    changed_at: f64,
    gl: Arc<Mutex<gl::GlShared>>,
    upload_key: (u64, usize),
    drag: Drag,
    tl_drag: Option<TlDrag>,
    floor_img: Option<(String, Arc<RgbaImage>)>,
    back_img: Option<(String, Arc<RgbaImage>)>,
    floor_key: u64,
    back_key: u64,
    ground_avg: [f32; 3],
    img_error: Option<String>,
    view_h: f32,
    view_size: [u32; 2],
    export_open: bool,
    export_prog: Option<Arc<Progress>>,
    toast: Option<(String, f64)>,
    selftest_stage: u8,
    last_saved: String,
    last_save_check: f64,
}

fn settings_path() -> std::path::PathBuf {
    paths::data_dir().join(paths::SETTINGS_FILE)
}

fn load_settings() -> Project {
    let read = |p: std::path::PathBuf| std::fs::read_to_string(p).ok().and_then(|s| serde_json::from_str::<Project>(&s).ok());
    // portable settings first, then the old %APPDATA% autosave (one-time migration)
    read(settings_path()).or_else(|| read(paths::legacy_settings())).unwrap_or_default()
}

/// Copy any images that live outside the program folder into it, so the folder is self-contained.
fn make_images_portable(p: &mut Project) {
    for slot in [&mut p.bg.floor_image, &mut p.bg.back_image] {
        if let Some(path) = slot.clone() {
            let real = paths::resolve(&path);
            if real.is_absolute() && !real.starts_with(paths::data_dir()) && real.exists() {
                if let Ok(rel) = paths::import_image(&real) {
                    *slot = Some(rel);
                }
            }
        }
    }
}

fn c32(c: [u8; 3]) -> Color32 {
    Color32::from_rgb(c[0], c[1], c[2])
}

fn v3(c: [u8; 3]) -> Vec3 {
    Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32) / 255.0
}

fn hash_of<T: Hash>(t: &T) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    t.hash(&mut h);
    h.finish() | 1
}

fn load_image(path: &str) -> Result<Arc<RgbaImage>, String> {
    let img = image::open(paths::resolve(path)).map_err(|e| format!("Can't open image {path}: {e}"))?.to_rgba8();
    let img = if img.width() > 8192 || img.height() > 8192 {
        let s = 8192.0 / img.width().max(img.height()) as f32;
        image::imageops::resize(&img, (img.width() as f32 * s) as u32, (img.height() as f32 * s) as u32, image::imageops::FilterType::Triangle)
    } else {
        img
    };
    Ok(Arc::new(img))
}

/// Pick an image and copy it into the program's Images folder. Returns the stored (relative) path.
fn pick_image() -> Option<Result<String, String>> {
    rfd::FileDialog::new()
        .add_filter("Images", &["png", "jpg", "jpeg", "bmp", "webp", "gif"])
        .set_directory(paths::sub("Images"))
        .pick_file()
        .map(|p| paths::import_image(&p))
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let ctx = cc.egui_ctx.clone();
        ctx.set_visuals(egui::Visuals::dark());
        let proj: Project = if std::env::var("FLUID_SELFTEST").is_ok() {
            let mut p = Project::default();
            if std::env::var("FLUID_SELFTEST_CLOSE").is_ok() {
                p.emitters[0].transparency = 75.0;
            }
            if let Ok(img) = std::env::var("FLUID_SELFTEST_FLOOR") {
                p.bg.floor_mode = FloorMode::Image;
                p.bg.floor_image = Some(img);
                p.bg.floor_tiled = std::env::var("FLUID_SELFTEST_TILED").is_ok();
                p.bg.floor_image_m = 5.0;
                p.cam.dist = 11.0;
            }
            make_images_portable(&mut p);
            p
        } else {
            let mut p = load_settings();
            make_images_portable(&mut p);
            p
        };
        let mut proj = proj;
        // ffmpeg: keep a saved path only if it still exists, otherwise auto-detect (bundled one first)
        if proj.export.ffmpeg.is_empty() || !paths::resolve(&proj.export.ffmpeg).exists() {
            proj.export.ffmpeg = export::find_ffmpeg().unwrap_or_default();
        }
        if proj.export.ffmpeg.is_empty() && proj.export.format.needs_ffmpeg() {
            proj.export.format = Format::Png;
        }
        Self {
            selected: proj.emitters.first().map(|e| e.id),
            proj,
            add_mode: false,
            new_height: 1.0,
            playing: false,
            playhead: 0.0,
            baker: Baker::new(ctx.clone()),
            applied_key: String::new(),
            seen_key: String::new(),
            changed_at: 0.0,
            gl: Arc::new(Mutex::new(gl::GlShared::default())),
            upload_key: (u64::MAX, 0),
            drag: Drag::None,
            tl_drag: None,
            floor_img: None,
            back_img: None,
            floor_key: 0,
            back_key: 0,
            ground_avg: [0.5; 3],
            img_error: None,
            view_h: 800.0,
            view_size: [1280, 720],
            export_open: false,
            export_prog: None,
            toast: None,
            selftest_stage: 0,
            last_saved: String::new(),
            last_save_check: 0.0,
        }
    }

    // ---------------------------------------------------------------- helpers

    fn sel_index(&self) -> Option<usize> {
        let id = self.selected?;
        self.proj.emitters.iter().position(|e| e.id == id)
    }

    fn index_of(&self, id: u32) -> Option<usize> {
        self.proj.emitters.iter().position(|e| e.id == id)
    }

    fn notify(&mut self, ctx: &egui::Context, msg: impl Into<String>) {
        self.toast = Some((msg.into(), ctx.input(|i| i.time) + 3.0));
    }

    fn add_emitter(&mut self, pos: Vec3) {
        let id = self.proj.next_id;
        self.proj.next_id += 1;
        let dur = self.proj.params.duration;
        let yaw = if pos.x.abs() + pos.z.abs() > 0.1 { (-pos.x).atan2(-pos.z).to_degrees() } else { 0.0 };
        let start = self.playhead.min((dur - 0.1).max(0.0));
        let e = Emitter {
            id,
            name: format!("Point {id}"),
            pos: pos.into(),
            yaw,
            pitch: 40.0,
            start,
            length: 2.0f32.min(dur - start).max(0.05),
            color: PALETTE[self.proj.emitters.len() % PALETTE.len()],
            ..Default::default()
        };
        self.proj.emitters.push(e);
        self.selected = Some(id);
    }

    fn duplicate_selected(&mut self) {
        if let Some(i) = self.sel_index() {
            let mut e = self.proj.emitters[i].clone();
            e.id = self.proj.next_id;
            self.proj.next_id += 1;
            e.name = format!("{} copy", e.name);
            e.pos[0] += 0.5;
            self.selected = Some(e.id);
            self.proj.emitters.push(e);
        }
    }

    fn delete_selected(&mut self) {
        if let Some(i) = self.sel_index() {
            self.proj.emitters.remove(i);
            self.selected = self.proj.emitters.get(i.min(self.proj.emitters.len().saturating_sub(1))).map(|e| e.id);
        }
    }

    fn toggle_play(&mut self) {
        if !self.playing && self.playhead >= self.proj.params.duration - 1e-3 {
            self.playhead = 0.0;
        }
        self.playing = !self.playing;
    }

    fn save_project(&mut self, ctx: &egui::Context) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("Fluid project", &["json"])
            .set_directory(paths::sub("Projects"))
            .set_file_name("fluid_project.json")
            .save_file()
        {
            match serde_json::to_string_pretty(&self.proj).map_err(|e| e.to_string()).and_then(|s| std::fs::write(&path, s).map_err(|e| e.to_string())) {
                Ok(()) => self.notify(ctx, format!("Saved {}", path.display())),
                Err(e) => self.notify(ctx, format!("Save failed: {e}")),
            }
        }
    }

    fn open_project(&mut self, ctx: &egui::Context) {
        if let Some(path) = rfd::FileDialog::new().add_filter("Fluid project", &["json"]).set_directory(paths::sub("Projects")).pick_file() {
            match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| serde_json::from_str::<Project>(&s).map_err(|e| e.to_string())) {
                Ok(mut p) => {
                    make_images_portable(&mut p);
                    // keep this PC's ffmpeg if the project's one isn't here
                    if p.export.ffmpeg.is_empty() || !paths::resolve(&p.export.ffmpeg).exists() {
                        p.export.ffmpeg = self.proj.export.ffmpeg.clone();
                    }
                    self.proj = p;
                    self.selected = self.proj.emitters.first().map(|e| e.id);
                    self.playhead = 0.0;
                    self.playing = false;
                    self.notify(ctx, format!("Opened {}", path.display()));
                }
                Err(e) => self.notify(ctx, format!("Open failed: {e}")),
            }
        }
    }

    /// Write settings.json next to the program (only when something changed).
    fn autosave(&mut self) {
        if std::env::var("FLUID_SELFTEST").is_ok() && std::env::var("FLUID_SELFTEST_SAVE").is_err() {
            return;
        }
        let r = serde_json::to_string_pretty(&self.proj);
        if let Err(e) = &r {
            eprintln!("autosave: can't serialize settings: {e}");
        }
        if let Ok(s) = r {
            if s != self.last_saved {
                // write to a temp file first so a crash mid-write can't corrupt the settings
                let p = settings_path();
                let tmp = p.with_extension("json.tmp");
                if std::fs::write(&tmp, &s).is_ok() && std::fs::rename(&tmp, &p).is_ok() {
                    self.last_saved = s;
                }
            }
        }
    }

    fn periodic_autosave(&mut self, ctx: &egui::Context) {
        let now = ctx.input(|i| i.time);
        // don't write while the user is dragging something
        if now - self.last_save_check > 3.0 && !ctx.input(|i| i.pointer.any_down()) {
            self.last_save_check = now;
            self.autosave();
        }
    }

    /// Keep the GPU textures in sync with ground/backdrop settings.
    fn sync_textures(&mut self) {
        let bg = &self.proj.bg;
        for (want, slot) in [(&bg.floor_image, &mut self.floor_img), (&bg.back_image, &mut self.back_img)] {
            if let Some(path) = want {
                if slot.as_ref().map(|s| &s.0) != Some(path) {
                    match load_image(path) {
                        Ok(img) => *slot = Some((path.clone(), img)),
                        Err(e) => {
                            self.img_error = Some(e);
                            *slot = Some((path.clone(), Arc::new(bg::checker_image([128; 3], [96; 3]))));
                        }
                    }
                }
            }
        }
        let floor_key = match (bg.floor_mode, &self.floor_img) {
            (FloorMode::Image, Some((p, _))) => hash_of(&("img", p)),
            _ => hash_of(&("checker", bg.c1, bg.c2)),
        };
        let back_key = self.back_img.as_ref().map(|(p, _)| hash_of(p)).unwrap_or(0);
        let mut g = self.gl.lock().unwrap();
        if floor_key != self.floor_key {
            let img = match (bg.floor_mode, &self.floor_img) {
                (FloorMode::Image, Some((_, img))) => img.clone(),
                _ => Arc::new(bg::checker_image(bg.c1, bg.c2)),
            };
            self.ground_avg = bg::average_color(&img);
            g.floor_src = Some(gl::TexSrc { key: floor_key, img });
            self.floor_key = floor_key;
        }
        if back_key != self.back_key {
            if let Some((_, img)) = &self.back_img {
                g.back_src = Some(gl::TexSrc { key: back_key, img: img.clone() });
            }
            self.back_key = back_key;
        }
    }

    /// Size in metres of the ground image (when the ground uses an image).
    fn floor_image_size(&self) -> Option<[f32; 2]> {
        let bg = &self.proj.bg;
        match (bg.floor_mode, &self.floor_img) {
            (FloorMode::Image, Some((_, img))) => {
                let m = bg.floor_image_m.max(0.05);
                Some([m, m * img.height() as f32 / img.width().max(1) as f32])
            }
            _ => None,
        }
    }

    /// Half-size of the ground when it is a single (untiled) image; [0,0] = endless.
    fn floor_half(&self) -> [f32; 2] {
        match self.floor_image_size() {
            Some([w, d]) if !self.proj.bg.floor_tiled => [w * 0.5, d * 0.5],
            _ => [0.0, 0.0],
        }
    }

    /// Simulation settings with the ground's extent applied.
    fn sim_params(&self) -> SimParams {
        SimParams { floor_half: self.floor_half(), ..self.proj.params.clone() }
    }

    /// Render parameters for a w x h image (points scale for 1:1 / tile backdrop).
    fn render_params(&self, cam: &OrbitCam, w: f32, h: f32, points_scale: f32) -> gl::RenderParams {
        let bg = &self.proj.bg;
        let aspect = w / h.max(1.0);
        let mut back = gl::BackdropU {
            mode: match bg.back_mode {
                BackMode::Gradient => 0,
                BackMode::Solid => 1,
                BackMode::Image => 2,
            },
            top: v3(bg.top),
            horizon: v3(bg.horizon),
            solid: v3(bg.solid),
            rect: [0.0, 0.0, 1.0, 1.0],
            tile: bg.fit == Fit::Tile,
        };
        if back.mode == 2 {
            if let Some((_, img)) = &self.back_img {
                let [x, y, rw, rh] = bg::layout(bg.fit, w, h, img.width() as f32, img.height() as f32, points_scale);
                back.rect = [x / w, y / h, rw / w, rh / h];
            } else {
                back.mode = 0;
            }
        }
        if back.mode == 1 {
            back.horizon = back.solid;
            back.top = back.solid;
        }
        let (tile, center) = match self.floor_image_size() {
            Some(sz) => (sz, 0.5), // image centred on the origin
            None => ([bg.cell.max(0.01) * 2.0; 2], 0.0),
        };
        gl::RenderParams {
            view: cam.view(),
            proj: cam.proj(aspect),
            eye: cam.eye(),
            target: Vec3::from(cam.target),
            radius: self.proj.params.h * 0.55 * self.proj.droplet_scale,
            h: self.proj.params.h,
            smooth: self.proj.smooth,
            refraction: self.proj.refraction,
            shine: self.proj.shine,
            transparent: false,
            show_floor: self.proj.show_floor,
            floor_tile: tile,
            floor_center: center,
            floor_half: self.floor_half(),
            ground: Vec3::from(self.ground_avg) * 0.8,
            back,
        }
    }

    // --------------------------------------------------------------- per-frame

    fn handle_keys(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        let (space, a, del, home, end, esc, dup, save) = ctx.input_mut(|i| {
            (
                i.consume_key(Modifiers::NONE, Key::Space),
                i.consume_key(Modifiers::NONE, Key::A),
                i.consume_key(Modifiers::NONE, Key::Delete),
                i.consume_key(Modifiers::NONE, Key::Home),
                i.consume_key(Modifiers::NONE, Key::End),
                i.consume_key(Modifiers::NONE, Key::Escape),
                i.consume_key(Modifiers::COMMAND, Key::D),
                i.consume_key(Modifiers::COMMAND, Key::S),
            )
        });
        if space {
            self.toggle_play();
        }
        if a {
            self.add_mode = !self.add_mode;
        }
        if del {
            self.delete_selected();
        }
        if home {
            self.playhead = 0.0;
        }
        if end {
            self.playhead = self.proj.params.duration;
            self.playing = false;
        }
        if esc {
            if self.add_mode {
                self.add_mode = false;
            } else {
                self.selected = None;
            }
        }
        if dup {
            self.duplicate_selected();
        }
        if save {
            self.save_project(ctx);
        }
    }

    fn advance_playback(&mut self, ctx: &egui::Context) {
        let dt = ctx.input(|i| i.stable_dt).min(0.1);
        let (frontier, total) = {
            let c = self.baker.cache.lock().unwrap();
            (c.frontier, c.total)
        };
        let dur = self.proj.params.duration.max(0.1);
        if self.playing {
            self.playhead += dt;
            let ready = frontier as f32 / FPS as f32;
            if frontier < total && self.playhead > ready {
                self.playhead = ready; // wait for the simulation to catch up
            }
            if self.playhead >= dur {
                if self.proj.loop_play {
                    self.playhead = 0.0;
                } else {
                    self.playhead = dur;
                    self.playing = false;
                }
            }
            ctx.request_repaint();
        }
        self.playhead = self.playhead.clamp(0.0, dur);
    }

    fn check_dirty(&mut self, ctx: &egui::Context) {
        let sp = self.sim_params();
        let key = serde_json::to_string(&(&sp, &self.proj.emitters, self.proj.cache_mb)).unwrap_or_default();
        let now = ctx.input(|i| i.time);
        if key != self.seen_key {
            self.seen_key = key.clone();
            self.changed_at = now;
        }
        if key != self.applied_key {
            if self.applied_key.is_empty() || now - self.changed_at > 0.12 {
                self.baker.restart(&sp, &self.proj.emitters, self.proj.cache_mb);
                self.applied_key = key;
                ctx.request_repaint();
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(130));
            }
        }
    }

    // ----------------------------------------------------------------- toolbar

    fn toolbar(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.toggle_value(&mut self.add_mode, RichText::new("➕ Add points").strong())
                .on_hover_text("(A) Click on the floor in the 3D view to place emission points. Esc to stop.");
            ui.label("at height");
            ui.add(DragValue::new(&mut self.new_height).speed(0.05).range(0.0..=30.0).suffix(" m"));
            ui.separator();
            if ui.button("⏮").on_hover_text("Go to start (Home)").clicked() {
                self.playhead = 0.0;
            }
            let label = if self.playing { "⏸ Pause" } else { "⏵ Play" };
            if ui.add(Button::new(RichText::new(label).strong().size(15.0)).min_size(vec2(86.0, 24.0))).on_hover_text("Space").clicked() {
                self.toggle_play();
            }
            ui.checkbox(&mut self.proj.loop_play, "Loop");
            ui.monospace(format!("{:6.2} / {:.2} s", self.playhead, self.proj.params.duration));
            ui.separator();
            if ui.button("🎬 Export video…").clicked() {
                self.export_open = true;
            }
            if ui.button("💾 Save…").on_hover_text("Ctrl+S").clicked() {
                self.save_project(ui.ctx());
            }
            if ui.button("📂 Open…").clicked() {
                self.open_project(ui.ctx());
            }
            if ui.button("New").on_hover_text("Start over with the example scene").clicked() {
                let bg = self.proj.bg.clone();
                let export = self.proj.export.clone();
                self.proj = Project { bg, export, ..Default::default() };
                self.selected = self.proj.emitters.first().map(|e| e.id);
                self.playhead = 0.0;
                self.playing = false;
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let c = self.baker.cache.lock().unwrap();
                let txt = if c.done() {
                    format!("Ready · {} particles · cache {} MB", c.particles, c.bytes >> 20)
                } else {
                    format!(
                        "Simulating {:.0}% · {} particles · {:.1} ms/frame",
                        c.frontier as f32 / c.total.max(1) as f32 * 100.0,
                        c.particles,
                        c.step_ms
                    )
                };
                ui.weak(txt);
            });
        });
    }

    // -------------------------------------------------------------- left panel

    fn left_panel(&mut self, ui: &mut Ui) {
        ui.add_space(4.0);
        ui.heading("Emission points");
        ui.horizontal(|ui| {
            ui.toggle_value(&mut self.add_mode, "➕ Click to place");
            if ui.button("Add at center").clicked() {
                self.add_emitter(Vec3::new(0.0, self.new_height, 0.0));
            }
        });
        ui.add_space(4.0);
        let mut to_select = None;
        let mut to_delete = None;
        for e in self.proj.emitters.iter_mut() {
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(vec2(14.0, 14.0), Sense::hover());
                ui.painter().rect_filled(r, 3.0, c32(e.color));
                ui.checkbox(&mut e.enabled, "").on_hover_text("Enabled");
                let sel = Some(e.id) == self.selected;
                if ui.selectable_label(sel, &e.name).clicked() {
                    to_select = Some(e.id);
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui.small_button("🗑").on_hover_text("Delete").clicked() {
                        to_delete = Some(e.id);
                    }
                });
            });
        }
        if self.proj.emitters.is_empty() {
            ui.weak("No points yet. Turn on “Click to place” and click the floor in the 3D view.");
        }
        if let Some(id) = to_select {
            self.selected = Some(id);
        }
        if let Some(id) = to_delete {
            self.selected = Some(id);
            self.delete_selected();
        }

        ui.separator();
        let dur = self.proj.params.duration;
        let h = self.proj.params.h;
        let Some(i) = self.sel_index() else {
            ui.weak("Select a point to edit it.");
            return;
        };
        let mut dup = false;
        let mut del = false;
        {
            let e = &mut self.proj.emitters[i];
            ui.horizontal(|ui| {
                ui.label("Name");
                ui.text_edit_singleline(&mut e.name);
            });
            ui.add_space(4.0);
            egui::Grid::new("emitter_props").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                ui.label("Color");
                ui.color_edit_button_srgb(&mut e.color);
                ui.end_row();
                ui.label("Transparency");
                ui.add(Slider::new(&mut e.transparency, 0.0..=100.0).suffix(" %"))
                    .on_hover_text("0% = opaque like paint, 100% = clear like water");
                ui.end_row();
                ui.label("Amount");
                ui.add(Slider::new(&mut e.rate, 10.0..=10000.0).logarithmic(true).suffix(" /s"))
                    .on_hover_text("Particles emitted per second. More = thicker stream.");
                ui.end_row();
                ui.label("Start");
                ui.add(DragValue::new(&mut e.start).speed(0.01).range(0.0..=dur).suffix(" s"));
                ui.end_row();
                ui.label("Length");
                ui.add(DragValue::new(&mut e.length).speed(0.01).range(0.02..=dur.max(0.02)).suffix(" s"));
                ui.end_row();
                ui.label("Speed");
                ui.add(Slider::new(&mut e.speed, 0.0..=25.0).suffix(" m/s"));
                ui.end_row();
                ui.label("Spread");
                ui.add(Slider::new(&mut e.spread, 0.0..=120.0).suffix("°")).on_hover_text("Cone angle of the spray");
                ui.end_row();
                ui.label("Nozzle size");
                ui.add(Slider::new(&mut e.radius, 0.01..=1.0).logarithmic(true).suffix(" m"));
                ui.end_row();
                ui.label("Aim (left/right)");
                ui.add(Slider::new(&mut e.yaw, -180.0..=180.0).suffix("°"));
                ui.end_row();
                ui.label("Aim (up/down)");
                ui.add(Slider::new(&mut e.pitch, -90.0..=90.0).suffix("°"));
                ui.end_row();
                ui.label("Position");
                ui.horizontal(|ui| {
                    ui.add(DragValue::new(&mut e.pos[0]).speed(0.02).prefix("x ").max_decimals(2));
                    ui.add(DragValue::new(&mut e.pos[1]).speed(0.02).prefix("y ").max_decimals(2));
                    ui.add(DragValue::new(&mut e.pos[2]).speed(0.02).prefix("z ").max_decimals(2));
                });
                ui.end_row();
            });
            e.length = e.length.min((dur - e.start).max(0.02));
            let stream = sim::effective_radius(e, h);
            ui.add_space(4.0);
            ui.weak(format!("≈ {} particles total · stream ⌀ {:.2} m", (e.rate * e.length).round() as i64, stream * 2.0));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                dup = ui.button("Duplicate").on_hover_text("Ctrl+D").clicked();
                del = ui.button("Delete").on_hover_text("Del").clicked();
            });
        }
        if dup {
            self.duplicate_selected();
        }
        if del {
            self.delete_selected();
        }
        ui.add_space(10.0);
        ui.collapsing("Controls", |ui| {
            ui.weak("3D view: left-drag orbit · right-drag (or Shift+drag) pan · wheel zoom");
            ui.weak("Drag a point's ball to slide it along the floor, the colored arrows to move along an axis, and the tip of its arrow to aim it.");
            ui.weak("Timeline: drag bars to move, drag bar ends to change length (Shift = no snapping). Click the ruler to scrub.");
            ui.weak("Keys: Space play/pause · A add mode · Del delete · Ctrl+D duplicate · Home/End");
        });
    }

    // ------------------------------------------------------------- right panel

    fn right_panel(&mut self, ui: &mut Ui) {
        ui.add_space(4.0);
        ui.heading("Liquid look");
        egui::Grid::new("look").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Smoothness");
            ui.add(Slider::new(&mut self.proj.smooth, 0.0..=3.0)).on_hover_text("How much the droplets melt into one surface");
            ui.end_row();
            ui.label("Body size");
            ui.add(Slider::new(&mut self.proj.droplet_scale, 0.5..=2.5)).on_hover_text("How thick the liquid looks around each particle");
            ui.end_row();
            ui.label("Refraction");
            ui.add(Slider::new(&mut self.proj.refraction, 0.0..=0.15)).on_hover_text("How much the liquid bends what's behind it");
            ui.end_row();
            ui.label("Shine");
            ui.add(Slider::new(&mut self.proj.shine, 0.0..=2.0));
            ui.end_row();
        });

        ui.separator();
        ui.heading("Ground");
        ui.checkbox(&mut self.proj.show_floor, "Show ground");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.proj.bg.floor_mode, FloorMode::Checker, "Checkerboard");
            ui.selectable_value(&mut self.proj.bg.floor_mode, FloorMode::Image, "Image");
        });
        match self.proj.bg.floor_mode {
            FloorMode::Checker => {
                ui.add(Slider::new(&mut self.proj.bg.cell, 0.05..=5.0).logarithmic(true).suffix(" m").text("square size"));
                ui.horizontal(|ui| {
                    ui.label("Colors");
                    ui.color_edit_button_srgb(&mut self.proj.bg.c1);
                    ui.color_edit_button_srgb(&mut self.proj.bg.c2);
                });
            }
            FloorMode::Image => {
                if let Some(p) = &self.proj.bg.floor_image {
                    ui.weak(std::path::Path::new(p).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()).on_hover_text(p);
                }
                ui.add(Slider::new(&mut self.proj.bg.floor_image_m, 0.2..=60.0).logarithmic(true).suffix(" m").text("image width"));
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.proj.bg.floor_tiled, true, "Tiled")
                        .on_hover_text("Repeat the image forever, centered in the middle");
                    ui.selectable_value(&mut self.proj.bg.floor_tiled, false, "Single image")
                        .on_hover_text("One centered image; the ground ends at its edges and liquid pours off");
                });
                if let Some([w, d]) = self.floor_image_size() {
                    ui.weak(format!("Image covers {w:.2} × {d:.2} m, centered"));
                }
            }
        }
        if ui.button("🖼 Choose ground image from PC…").clicked() {
            match pick_image() {
                Some(Ok(p)) => {
                    self.img_error = None;
                    self.proj.bg.floor_image = Some(p);
                    self.proj.bg.floor_mode = FloorMode::Image;
                }
                Some(Err(e)) => self.img_error = Some(e),
                None => {}
            }
        }

        ui.separator();
        ui.heading("Backdrop");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.proj.bg.back_mode, BackMode::Gradient, "Sky");
            ui.selectable_value(&mut self.proj.bg.back_mode, BackMode::Solid, "Solid");
            ui.selectable_value(&mut self.proj.bg.back_mode, BackMode::Image, "Image");
        });
        match self.proj.bg.back_mode {
            BackMode::Gradient => {
                ui.horizontal(|ui| {
                    ui.label("Top");
                    ui.color_edit_button_srgb(&mut self.proj.bg.top);
                    ui.label("Horizon");
                    ui.color_edit_button_srgb(&mut self.proj.bg.horizon);
                });
            }
            BackMode::Solid => {
                ui.horizontal(|ui| {
                    ui.label("Color");
                    ui.color_edit_button_srgb(&mut self.proj.bg.solid);
                });
            }
            BackMode::Image => {
                if let Some(p) = &self.proj.bg.back_image {
                    ui.weak(std::path::Path::new(p).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()).on_hover_text(p);
                }
                egui::ComboBox::from_label("Fit").selected_text(self.proj.bg.fit.name()).show_ui(ui, |ui| {
                    for f in Fit::ALL {
                        ui.selectable_value(&mut self.proj.bg.fit, f, f.name());
                    }
                });
            }
        }
        if ui.button("🖼 Choose backdrop image from PC…").clicked() {
            match pick_image() {
                Some(Ok(p)) => {
                    self.img_error = None;
                    self.proj.bg.back_image = Some(p);
                    self.proj.bg.back_mode = BackMode::Image;
                }
                Some(Err(e)) => self.img_error = Some(e),
                None => {}
            }
        }
        if let Some(e) = &self.img_error {
            ui.colored_label(Color32::LIGHT_RED, e);
        }

        ui.separator();
        ui.heading("Simulation");
        let p = &mut self.proj.params;
        egui::Grid::new("sim_params").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Duration");
            ui.add(DragValue::new(&mut p.duration).speed(0.1).range(0.5..=120.0).suffix(" s"));
            ui.end_row();
            ui.label("Gravity");
            ui.add(Slider::new(&mut p.gravity, 0.0..=30.0).suffix(" m/s²"));
            ui.end_row();
            ui.label("Viscosity");
            ui.add(Slider::new(&mut p.viscosity, 0.0..=1.0)).on_hover_text("0 = water, 1 = honey");
            ui.end_row();
            ui.label("Droplet size");
            ui.add(Slider::new(&mut p.h, 0.05..=0.5).logarithmic(true).suffix(" m"))
                .on_hover_text("Simulation resolution. Smaller = finer detail but needs more particles.");
            ui.end_row();
            ui.label("Pressure");
            ui.add(Slider::new(&mut p.stiffness, 0.05..=2.0)).on_hover_text("How strongly the fluid resists compression");
            ui.end_row();
            ui.label("Splash");
            ui.add(Slider::new(&mut p.near_stiffness, 0.1..=3.0)).on_hover_text("Short-range repulsion between droplets");
            ui.end_row();
            ui.label("Color mixing");
            ui.add(Slider::new(&mut p.color_mix, 0.0..=1.0)).on_hover_text("How fast touching fluids blend their colors");
            ui.end_row();
            ui.label("Lifetime");
            ui.add(DragValue::new(&mut p.lifetime).speed(0.05).range(0.0..=120.0).suffix(" s"))
                .on_hover_text("Particles disappear after this long. 0 = never.");
            ui.end_row();
            ui.label("Max particles");
            ui.add(DragValue::new(&mut p.max_particles).speed(200.0).range(1000..=500_000));
            ui.end_row();
            ui.label("Quality");
            ui.add(Slider::new(&mut p.substeps, 1..=8).suffix(" substeps"));
            ui.end_row();
            ui.label("Floor");
            ui.checkbox(&mut p.floor, "liquid collides with floor");
            ui.end_row();
            ui.label("Floor friction");
            ui.add(Slider::new(&mut p.friction, 0.0..=1.0));
            ui.end_row();
            ui.label("Walls");
            ui.checkbox(&mut p.container, "square container");
            ui.end_row();
            if p.container {
                ui.label("Container half-size");
                ui.add(Slider::new(&mut p.container_size, 0.3..=15.0).suffix(" m"));
                ui.end_row();
            }
        });
        if ui.small_button("Reset simulation settings").clicked() {
            let d = self.proj.params.duration;
            self.proj.params = SimParams { duration: d, ..Default::default() };
        }

        ui.separator();
        ui.heading("Display");
        ui.checkbox(&mut self.proj.show_helpers, "Show points & arrows");
        ui.checkbox(&mut self.proj.show_grid, "Show grid lines");

        ui.separator();
        ui.heading("Storage");
        let dir = paths::data_dir();
        ui.weak("Everything is saved in the program folder:").on_hover_text(dir.display().to_string());
        ui.horizontal(|ui| {
            if ui.button("📂 Open program folder").clicked() {
                let _ = std::process::Command::new("explorer").arg(dir).spawn();
            }
            if ui.button("Save now").on_hover_text("Settings also save automatically every few seconds").clicked() {
                self.autosave();
            }
        });
        ui.collapsing("Performance", |ui| {
            ui.horizontal(|ui| {
                ui.label("Frame cache limit");
                ui.add(DragValue::new(&mut self.proj.cache_mb).speed(16.0).range(128..=32768).suffix(" MB"));
            });
            let c = self.baker.cache.lock().unwrap();
            ui.weak(format!("Using {} MB, keeping every {} frame(s).", c.bytes >> 20, c.stride));
        });
    }

    // --------------------------------------------------------------- viewport

    fn handle_positions(&self, xf: &Xf, e: &Emitter) -> (Option<Pos2>, Option<Pos2>, [Option<[Pos2; 2]>; 3]) {
        let o = e.origin();
        let base = xf.project(o);
        let tip = xf.project(o + e.dir() * AIM_LEN);
        let mut axes = [None; 3];
        if Some(e.id) == self.selected {
            let len = AXIS_PX / xf.px_per_unit(o).max(1e-3);
            for (k, ax) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
                axes[k] = xf.segment(o, o + ax * len);
            }
        }
        (base, tip, axes)
    }

    fn hit_test(&self, xf: &Xf, p: Pos2) -> Option<Hit> {
        if !self.proj.show_helpers {
            return None;
        }
        if let Some(i) = self.sel_index() {
            let e = &self.proj.emitters[i];
            let (_, _, axes) = self.handle_positions(xf, e);
            for (k, a) in axes.iter().enumerate() {
                if let Some([s0, s1]) = a {
                    let start = *s0 + (*s1 - *s0) * 0.3;
                    if dist_to_segment(p, start, *s1) < 7.0 {
                        return Some(Hit::Axis(e.id, k));
                    }
                }
            }
        }
        let mut best: Option<(f32, Hit)> = None;
        for e in &self.proj.emitters {
            let (base, tip, _) = self.handle_positions(xf, e);
            if let Some(t) = tip {
                let d = (t - p).length();
                if d < 9.0 && best.is_none_or(|b| d < b.0) {
                    best = Some((d, Hit::Tip(e.id)));
                }
            }
            if let Some(b) = base {
                let d = (b - p).length();
                if d < 10.0 && best.is_none_or(|bb| d < bb.0) {
                    best = Some((d, Hit::Base(e.id)));
                }
            }
        }
        best.map(|b| b.1)
    }

    fn draw_scene_overlay(&self, painter: &egui::Painter, xf: &Xf) {
        if self.proj.show_grid {
            let grid = Color32::from_rgba_unmultiplied(128, 128, 128, 110);
            let n = 10;
            for i in -n..=n {
                let f = i as f32;
                if let Some(s) = xf.segment(Vec3::new(f, 0.0, -n as f32), Vec3::new(f, 0.0, n as f32)) {
                    painter.line_segment(s, Stroke::new(1.0_f32, grid));
                }
                if let Some(s) = xf.segment(Vec3::new(-n as f32, 0.0, f), Vec3::new(n as f32, 0.0, f)) {
                    painter.line_segment(s, Stroke::new(1.0_f32, grid));
                }
            }
        }
        let p = &self.proj.params;
        if p.container && self.proj.show_helpers {
            let b = p.container_size;
            let hgt = 1.5;
            let c = [(-b, -b), (b, -b), (b, b), (-b, b)];
            let st = Stroke::new(1.5_f32, Color32::from_rgba_unmultiplied(120, 200, 255, 160));
            for k in 0..4 {
                let (x0, z0) = c[k];
                let (x1, z1) = c[(k + 1) % 4];
                for y in [0.0, hgt] {
                    if let Some(s) = xf.segment(Vec3::new(x0, y, z0), Vec3::new(x1, y, z1)) {
                        painter.line_segment(s, st);
                    }
                }
                if let Some(s) = xf.segment(Vec3::new(x0, 0.0, z0), Vec3::new(x0, hgt, z0)) {
                    painter.line_segment(s, st);
                }
            }
        }
    }

    fn draw_emitters(&self, painter: &egui::Painter, xf: &Xf) {
        if !self.proj.show_helpers {
            return;
        }
        let outline = Color32::from_rgb(20, 20, 20);
        for e in &self.proj.emitters {
            let sel = Some(e.id) == self.selected;
            let col = if e.enabled { c32(e.color) } else { Color32::GRAY };
            let o = e.origin();
            if let Some(s) = xf.segment(o, Vec3::new(o.x, 0.0, o.z)) {
                painter.add(Shape::dashed_line(&s, Stroke::new(1.0_f32, Color32::from_rgba_unmultiplied(128, 128, 128, 200)), 4.0, 4.0));
                painter.circle_filled(s[1], 2.5, Color32::from_rgba_unmultiplied(128, 128, 128, 220));
            }
            let (base, tip, axes) = self.handle_positions(xf, e);
            if let Some(s) = xf.segment(o, o + e.dir() * AIM_LEN) {
                painter.line_segment(s, Stroke::new(4.0_f32, outline));
                painter.line_segment(s, Stroke::new(2.0_f32, col));
            }
            if let Some(t) = tip {
                painter.circle(t, if sel { 6.0 } else { 4.5 }, col, Stroke::new(1.5_f32, outline));
            }
            if let Some(b) = base {
                painter.circle(b, if sel { 9.0 } else { 7.0 }, col, Stroke::new(if sel { 2.5 } else { 1.5_f32 }, if sel { Color32::WHITE } else { outline }));
                let galley = painter.layout_no_wrap(e.name.clone(), FontId::proportional(12.0), Color32::WHITE);
                let tp = b + vec2(12.0, -18.0);
                painter.rect_filled(Rect::from_min_size(tp - vec2(3.0, 1.0), galley.size() + vec2(6.0, 2.0)), 3.0, Color32::from_black_alpha(150));
                painter.galley(tp, galley, Color32::WHITE);
            }
            let axis_cols = [Color32::from_rgb(235, 70, 70), Color32::from_rgb(80, 210, 80), Color32::from_rgb(70, 130, 255)];
            for (k, a) in axes.iter().enumerate() {
                if let Some([s0, s1]) = a {
                    painter.line_segment([*s0, *s1], Stroke::new(3.0_f32, axis_cols[k]));
                    painter.circle_filled(*s1, 5.0, axis_cols[k]);
                }
            }
        }
    }

    fn viewport(&mut self, ui: &mut Ui) {
        let rect = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(rect, Sense::click_and_drag());
        self.view_h = rect.height();
        let ppp = ui.ctx().pixels_per_point();
        self.view_size = [(rect.width() * ppp) as u32, (rect.height() * ppp) as u32];
        let painter = ui.painter_at(rect);
        let ctx = ui.ctx().clone();

        // ---- interaction
        let xf = Xf::new(&self.proj.cam, rect);
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.proj.cam.zoom(scroll);
            }
        }
        if resp.drag_started() {
            let press = ui.input(|i| i.pointer.press_origin()).unwrap_or(rect.center());
            let primary = ui.input(|i| i.pointer.button_down(PointerButton::Primary));
            let shift = ui.input(|i| i.modifiers.shift);
            self.drag = if !primary || shift {
                Drag::Pan
            } else {
                match self.hit_test(&xf, press) {
                    Some(Hit::Axis(id, axis)) => {
                        let start = self.index_of(id).map(|i| self.proj.emitters[i].origin()).unwrap_or_default();
                        Drag::Axis { id, axis, start, press }
                    }
                    Some(Hit::Tip(id)) => {
                        self.selected = Some(id);
                        let i = self.index_of(id).unwrap();
                        let e = &self.proj.emitters[i];
                        let tip = e.origin() + e.dir() * AIM_LEN;
                        Drag::Aim { id, plane_p: tip, plane_n: (xf.eye - tip).normalize_or_zero() }
                    }
                    Some(Hit::Base(id)) => {
                        self.selected = Some(id);
                        let i = self.index_of(id).unwrap();
                        let o = self.proj.emitters[i].origin();
                        let (ro, rd) = xf.ray(press);
                        let offset = ray_plane(ro, rd, o, Vec3::Y).map(|h| o - h).unwrap_or_default();
                        Drag::Plane { id, y: o.y, offset }
                    }
                    None => Drag::Orbit,
                }
            };
        }
        if resp.dragged() {
            let d = resp.drag_delta();
            let cur = resp.interact_pointer_pos();
            match self.drag {
                Drag::Orbit => self.proj.cam.orbit(d.x, d.y),
                Drag::Pan => self.proj.cam.pan(d.x, d.y, rect.height()),
                Drag::Axis { id, axis, start, press } => {
                    if let (Some(i), Some(cur)) = (self.index_of(id), cur) {
                        let av = [Vec3::X, Vec3::Y, Vec3::Z][axis];
                        if let (Some(s0), Some(s1)) = (xf.project(start), xf.project(start + av)) {
                            let sv = s1 - s0;
                            if sv.length_sq() > 1.0 {
                                let t = (cur - press).dot(sv) / sv.length_sq();
                                let mut np = start + av * t;
                                if axis == 1 {
                                    np.y = np.y.max(0.0);
                                }
                                self.proj.emitters[i].pos = np.into();
                            }
                        }
                    }
                }
                Drag::Plane { id, y, offset } => {
                    if let (Some(i), Some(cur)) = (self.index_of(id), cur) {
                        let (ro, rd) = xf.ray(cur);
                        if let Some(h) = ray_plane(ro, rd, Vec3::new(0.0, y, 0.0), Vec3::Y) {
                            let np = h + offset;
                            if np.x.abs() < 30.0 && np.z.abs() < 30.0 {
                                self.proj.emitters[i].pos = [np.x, y, np.z];
                            }
                        }
                    }
                }
                Drag::Aim { id, plane_p, plane_n } => {
                    if let (Some(i), Some(cur)) = (self.index_of(id), cur) {
                        let (ro, rd) = xf.ray(cur);
                        if let Some(h) = ray_plane(ro, rd, plane_p, plane_n) {
                            let e = &mut self.proj.emitters[i];
                            let dir = h - e.origin();
                            if dir.length() > 1e-3 {
                                let (yaw, pitch) = sim::yaw_pitch_from(dir);
                                e.yaw = (yaw * 10.0).round() / 10.0;
                                e.pitch = (pitch * 10.0).round() / 10.0;
                            }
                        }
                    }
                }
                Drag::None => {}
            }
        }
        if resp.drag_stopped() {
            self.drag = Drag::None;
        }
        if resp.clicked() {
            if let Some(pos) = resp.interact_pointer_pos() {
                match self.hit_test(&xf, pos) {
                    Some(Hit::Axis(id, _)) | Some(Hit::Tip(id)) | Some(Hit::Base(id)) => self.selected = Some(id),
                    None => {
                        if self.add_mode {
                            let (ro, rd) = xf.ray(pos);
                            if let Some(h) = ray_plane(ro, rd, Vec3::ZERO, Vec3::Y) {
                                if h.x.abs() < 30.0 && h.z.abs() < 30.0 {
                                    self.add_emitter(Vec3::new(h.x, self.new_height, h.z));
                                }
                            }
                        } else {
                            self.selected = None;
                        }
                    }
                }
            }
        }
        if resp.clicked_by(PointerButton::Secondary) {
            self.add_mode = false;
        }
        if let Some(hp) = resp.hover_pos() {
            match self.drag {
                Drag::None => {
                    if self.hit_test(&xf, hp).is_some() {
                        ctx.set_cursor_icon(CursorIcon::Grab);
                    } else if self.add_mode {
                        ctx.set_cursor_icon(CursorIcon::Crosshair);
                    }
                }
                Drag::Orbit | Drag::Pan => ctx.set_cursor_icon(CursorIcon::AllScroll),
                _ => ctx.set_cursor_icon(CursorIcon::Grabbing),
            }
        }

        // ---- GPU scene (floor, backdrop, liquid)
        let xf = Xf::new(&self.proj.cam, rect);
        self.sync_textures();
        {
            let f = (self.playhead as f64 * FPS).floor() as usize;
            let (frame, gen) = {
                let c = self.baker.cache.lock().unwrap();
                (c.get(f), c.gen)
            };
            let key = (gen, Arc::as_ptr(&frame) as usize);
            let params = self.render_params(&self.proj.cam.clone(), rect.width(), rect.height(), 1.0);
            let mut g = self.gl.lock().unwrap();
            if key != self.upload_key {
                g.pending = Some(frame);
                self.upload_key = key;
            }
            g.params = params;
        }
        let shared = self.gl.clone();
        let cb = egui_glow::CallbackFn::new(move |info, painter| gl::paint(&shared, painter.gl(), &info));
        painter.add(egui::PaintCallback { rect, callback: Arc::new(cb) });
        self.draw_scene_overlay(&painter, &xf);
        self.draw_emitters(&painter, &xf);

        // ---- hints
        let mut y = rect.top() + 10.0;
        let mut hint = |text: String, color: Color32| {
            let g = painter.layout_no_wrap(text, FontId::proportional(13.0), color);
            let r = Rect::from_min_size(pos2(rect.left() + 10.0, y), g.size() + vec2(12.0, 6.0));
            painter.rect_filled(r, 4.0, Color32::from_black_alpha(170));
            painter.galley(r.min + vec2(6.0, 3.0), g, color);
            y += r.height() + 6.0;
        };
        if self.add_mode {
            hint("Add mode: click on the floor to place points · Esc or right-click to stop".into(), Color32::from_rgb(255, 220, 120));
        }
        if let Some(err) = &self.gl.lock().unwrap().error {
            hint(format!("GPU error: {err}"), Color32::LIGHT_RED);
        }
        {
            let c = self.baker.cache.lock().unwrap();
            let want = (self.playhead as f64 * FPS) as usize;
            if want > c.frontier {
                hint("Simulating…".into(), Color32::WHITE);
            }
        }
        if let Some((msg, until)) = &self.toast {
            if ctx.input(|i| i.time) < *until {
                hint(msg.clone(), Color32::WHITE);
            }
        }
    }

    // --------------------------------------------------------------- timeline

    fn timeline(&mut self, ui: &mut Ui) {
        let dur = self.proj.params.duration.max(0.1);
        let label_w = 170.0;
        ui.add_space(2.0);
        let full_w = ui.available_width();
        let (ruler, rresp) = ui.allocate_exact_size(vec2(full_w, 26.0), Sense::click_and_drag());
        let x0 = ruler.left() + label_w;
        let tw = (ruler.right() - 16.0 - x0).max(10.0);
        let t2x = move |t: f32| x0 + t / dur * tw;
        let x2t = move |x: f32| ((x - x0) / tw * dur).clamp(0.0, dur);
        let painter = ui.painter().clone();
        let weak = ui.visuals().weak_text_color();
        painter.rect_filled(ruler, 0.0, ui.visuals().extreme_bg_color);
        painter.text(pos2(ruler.left() + 8.0, ruler.center().y), Align2::LEFT_CENTER, "TIMELINE", FontId::proportional(11.0), weak);

        let (front, total) = {
            let c = self.baker.cache.lock().unwrap();
            (c.frontier, c.total.max(1))
        };
        let frac = front as f32 / total as f32;
        painter.rect_filled(
            Rect::from_min_max(pos2(x0, ruler.bottom() - 4.0), pos2(x0 + tw * frac, ruler.bottom() - 1.0)),
            0.0,
            Color32::from_rgb(70, 160, 90),
        );
        let step = [0.1f32, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0].into_iter().find(|s| s / dur * tw >= 55.0).unwrap_or(60.0);
        let mut t = 0.0f32;
        while t <= dur + 1e-4 {
            let x = t2x(t);
            painter.line_segment([pos2(x, ruler.bottom() - 10.0), pos2(x, ruler.bottom() - 4.0)], Stroke::new(1.0_f32, weak));
            let txt = if step < 1.0 { format!("{t:.2}s") } else { format!("{t:.0}s") };
            painter.text(pos2(x + 3.0, ruler.top() + 3.0), Align2::LEFT_TOP, txt, FontId::proportional(10.0), weak);
            t += step;
        }
        if (rresp.is_pointer_button_down_on() || rresp.dragged()) && !rresp.drag_stopped() {
            if let Some(p) = rresp.interact_pointer_pos() {
                if p.x >= x0 - 8.0 {
                    self.playhead = x2t(p.x);
                    self.playing = false;
                }
            }
        }
        let px = t2x(self.playhead);
        painter.add(Shape::convex_polygon(
            vec![pos2(px - 6.0, ruler.top() + 1.0), pos2(px + 6.0, ruler.top() + 1.0), pos2(px, ruler.top() + 11.0)],
            PLAYHEAD,
            Stroke::NONE,
        ));
        painter.line_segment([pos2(px, ruler.top() + 8.0), pos2(px, ruler.bottom())], Stroke::new(1.5_f32, PLAYHEAD));

        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let ids: Vec<u32> = self.proj.emitters.iter().map(|e| e.id).collect();
            if ids.is_empty() {
                ui.add_space(8.0);
                ui.weak("    Add emission points to see them here.");
            }
            let painter = ui.painter().clone();
            let shift = ui.input(|i| i.modifiers.shift);
            for (row_i, id) in ids.iter().copied().enumerate() {
                let (row, row_resp) = ui.allocate_exact_size(vec2(full_w, 30.0), Sense::click_and_drag());
                let Some(idx) = self.index_of(id) else { continue };
                let selected = Some(id) == self.selected;
                if selected {
                    painter.rect_filled(row, 0.0, Color32::from_white_alpha(14));
                } else if row_i % 2 == 1 {
                    painter.rect_filled(row, 0.0, Color32::from_white_alpha(5));
                }
                let (s0, l0) = (self.proj.emitters[idx].start, self.proj.emitters[idx].length);
                let bar_of = |s: f32, l: f32| {
                    let bx0 = t2x(s);
                    let bx1 = t2x((s + l).min(dur)).max(bx0 + 6.0);
                    Rect::from_min_max(pos2(bx0, row.top() + 5.0), pos2(bx1, row.bottom() - 5.0))
                };
                let bar = bar_of(s0, l0);
                let bresp = ui.interact(bar.expand2(vec2(5.0, 0.0)), ui.id().with(("bar", id)), Sense::click_and_drag());
                let edge = |x: f32| {
                    if (x - bar.left()).abs() < 7.0 {
                        1u8
                    } else if (x - bar.right()).abs() < 7.0 {
                        2
                    } else {
                        0
                    }
                };
                if let Some(hp) = bresp.hover_pos() {
                    if self.tl_drag.is_none() {
                        ui.ctx().set_cursor_icon(if edge(hp.x) != 0 { CursorIcon::ResizeHorizontal } else { CursorIcon::Grab });
                    }
                }
                if bresp.drag_started() {
                    let pxs = ui.input(|i| i.pointer.press_origin()).map(|p| p.x).unwrap_or(bar.center().x);
                    self.tl_drag = Some(TlDrag { id, mode: edge(pxs), s0, l0, px: pxs });
                    self.selected = Some(id);
                }
                if bresp.dragged() {
                    if let (Some(d), Some(pp)) = (self.tl_drag, bresp.interact_pointer_pos()) {
                        if d.id == id {
                            ui.ctx().set_cursor_icon(if d.mode == 0 { CursorIcon::Grabbing } else { CursorIcon::ResizeHorizontal });
                            let dt = (pp.x - d.px) / tw * dur;
                            let snap = |v: f32| if shift { v } else { (v / 0.05).round() * 0.05 };
                            let e = &mut self.proj.emitters[idx];
                            match d.mode {
                                0 => e.start = snap(d.s0 + dt).clamp(0.0, (dur - d.l0).max(0.0)),
                                1 => {
                                    let end = d.s0 + d.l0;
                                    let s = snap(d.s0 + dt).clamp(0.0, end - 0.05);
                                    e.start = s;
                                    e.length = end - s;
                                }
                                _ => e.length = snap(d.l0 + dt).clamp(0.05, (dur - d.s0).max(0.05)),
                            }
                        }
                    }
                }
                if bresp.drag_stopped() {
                    self.tl_drag = None;
                }
                if bresp.clicked() {
                    self.selected = Some(id);
                }
                if row_resp.clicked() || row_resp.dragged() {
                    if let Some(pp) = row_resp.interact_pointer_pos() {
                        if pp.x < x0 - 4.0 {
                            if row_resp.clicked() {
                                self.selected = Some(id);
                            }
                        } else {
                            self.playhead = x2t(pp.x);
                            self.playing = false;
                        }
                    }
                }

                let e = &self.proj.emitters[idx];
                let col = if e.enabled { c32(e.color) } else { Color32::from_gray(90) };
                painter.circle_filled(pos2(row.left() + 14.0, row.center().y), 6.0, col);
                let name_col = if e.enabled { ui.visuals().text_color() } else { weak };
                let label_clip = painter.with_clip_rect(Rect::from_min_max(row.min, pos2(x0 - 6.0, row.bottom())));
                label_clip.text(pos2(row.left() + 26.0, row.center().y), Align2::LEFT_CENTER, &e.name, FontId::proportional(13.0), name_col);
                painter.line_segment([pos2(x0, row.center().y), pos2(x0 + tw, row.center().y)], Stroke::new(1.0_f32, Color32::from_white_alpha(12)));
                let bar = bar_of(e.start, e.length);
                painter.rect_filled(bar, 4.0, col);
                let hcol = Color32::from_black_alpha(90);
                painter.rect_filled(Rect::from_min_size(bar.min, vec2(5.0, bar.height())), 2.0, hcol);
                painter.rect_filled(Rect::from_min_size(pos2(bar.right() - 5.0, bar.top()), vec2(5.0, bar.height())), 2.0, hcol);
                if selected {
                    painter.rect_stroke(bar, 4.0, Stroke::new(2.0_f32, Color32::WHITE), egui::StrokeKind::Outside);
                }
                if bar.width() > 90.0 {
                    let luma = 0.299 * e.color[0] as f32 + 0.587 * e.color[1] as f32 + 0.114 * e.color[2] as f32;
                    let tc = if luma > 140.0 || !e.enabled { Color32::BLACK } else { Color32::WHITE };
                    painter.with_clip_rect(bar.shrink2(vec2(6.0, 0.0))).text(
                        pos2(bar.left() + 9.0, bar.center().y),
                        Align2::LEFT_CENTER,
                        format!("{:.2}s – {:.2}s", e.start, e.start + e.length),
                        FontId::proportional(11.0),
                        tc,
                    );
                }
                painter.line_segment([pos2(px, row.top()), pos2(px, row.bottom())], Stroke::new(1.5_f32, PLAYHEAD));
            }
        });
    }

    // ----------------------------------------------------------------- export

    fn start_export(&mut self) {
        let s = self.proj.export.clone();
        let (w, h) = (s.width.max(16) & !1, s.height.max(16) & !1);
        let rp = self.render_params(&self.proj.cam.clone(), w as f32, h as f32, h as f32 / self.view_h.max(1.0));
        let (prog, job) = export::start(&self.sim_params(), &self.proj.emitters, s, rp);
        let mut g = self.gl.lock().unwrap();
        g.export_finished = false;
        g.export = Some(job);
        self.export_prog = Some(prog);
    }

    fn cancel_export(&mut self) {
        if let Some(p) = &self.export_prog {
            p.cancel.store(true, Ordering::SeqCst);
        }
        self.gl.lock().unwrap().export = None;
    }

    fn export_window(&mut self, ctx: &egui::Context) {
        if let Some(p) = &self.export_prog {
            if !p.finished.load(Ordering::SeqCst) {
                ctx.request_repaint(); // keep the GPU export loop running
            }
        }
        if !self.export_open {
            return;
        }
        let mut open = true;
        let mut start = false;
        let mut cancel = false;
        egui::Window::new("Export video")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .default_width(420.0)
            .show(ctx, |ui| {
                if let Some(p) = self.export_prog.clone() {
                    let done = p.done.load(Ordering::SeqCst);
                    let total = p.total.load(Ordering::SeqCst).max(1);
                    let finished = p.finished.load(Ordering::SeqCst);
                    ui.label(p.msg.lock().unwrap().clone());
                    ui.add(egui::ProgressBar::new(done as f32 / total as f32).text(format!("{done} / {total} frames")));
                    ui.horizontal(|ui| {
                        if finished {
                            if let Some(path) = p.out_path.lock().unwrap().clone() {
                                if ui.button("📂 Show in folder").clicked() {
                                    let _ = std::process::Command::new("explorer").arg(format!("/select,{}", path.display())).spawn();
                                }
                                if path.is_file() && ui.button("⏵ Play video").clicked() {
                                    let _ = std::process::Command::new("explorer").arg(&path).spawn();
                                }
                            }
                            if ui.button("New export").clicked() {
                                self.export_prog = None;
                            }
                        } else if ui.button("Cancel").clicked() {
                            cancel = true;
                        }
                    });
                    return;
                }
                let s = &mut self.proj.export;
                egui::Grid::new("export_grid").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
                    ui.label("Format");
                    egui::ComboBox::from_id_salt("fmt").selected_text(s.format.name()).width(280.0).show_ui(ui, |ui| {
                        for f in Format::ALL {
                            ui.add_enabled_ui(!f.needs_ffmpeg() || !s.ffmpeg.is_empty(), |ui| {
                                ui.selectable_value(&mut s.format, f, f.name());
                            });
                        }
                    });
                    ui.end_row();
                    ui.label("Resolution");
                    ui.horizontal(|ui| {
                        ui.add(DragValue::new(&mut s.width).range(64..=7680));
                        ui.label("×");
                        ui.add(DragValue::new(&mut s.height).range(64..=4320));
                    });
                    ui.end_row();
                    ui.label("");
                    ui.horizontal(|ui| {
                        for (w, h, n) in [(1280, 720, "720p"), (1920, 1080, "1080p"), (3840, 2160, "4K"), (1080, 1920, "Vertical")] {
                            if ui.small_button(n).clicked() {
                                s.width = w;
                                s.height = h;
                            }
                        }
                        if ui.small_button("Viewport").clicked() {
                            s.width = self.view_size[0];
                            s.height = self.view_size[1];
                        }
                    });
                    ui.end_row();
                    ui.label("Frame rate");
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut s.fps, 30, "30 fps");
                        ui.radio_value(&mut s.fps, 60, "60 fps");
                    });
                    ui.end_row();
                    ui.label("Background");
                    ui.add_enabled(
                        s.format.supports_alpha(),
                        egui::Checkbox::new(&mut s.transparent, "Transparent (liquid only, no ground/backdrop)"),
                    );
                    ui.end_row();
                    ui.label("Save to");
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut s.folder).desired_width(230.0));
                        if ui.small_button("…").clicked() {
                            if let Some(p) = rfd::FileDialog::new().set_directory(paths::resolve(&s.folder)).pick_folder() {
                                s.folder = paths::relativize(&p);
                            }
                        }
                    });
                    ui.end_row();
                    ui.label("ffmpeg");
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(&mut s.ffmpeg).desired_width(230.0).hint_text("not found"));
                        if ui.small_button("…").clicked() {
                            if let Some(p) = rfd::FileDialog::new().add_filter("ffmpeg", &["exe"]).pick_file() {
                                s.ffmpeg = paths::relativize(&p);
                            }
                        }
                    });
                    ui.end_row();
                });
                ui.add_space(4.0);
                ui.weak(format!(
                    "Renders {:.1} s from the current camera view on the GPU, same look as the viewport.",
                    self.proj.params.duration
                ));
                ui.add_space(4.0);
                let ok = !s.format.needs_ffmpeg() || !s.ffmpeg.is_empty();
                start = ui.add_enabled(ok, Button::new(RichText::new("▶ Start export").strong())).clicked();
            });
        if cancel {
            self.cancel_export();
        }
        if start {
            self.start_export();
        }
        if !open {
            self.export_open = false;
            if self.export_prog.as_ref().is_some_and(|p| p.finished.load(Ordering::SeqCst)) {
                self.export_prog = None;
            }
        }
    }
}

/// Hidden self-test: FLUID_SELFTEST=<png> scrubs to 2.5 s, screenshots, and if
/// FLUID_SELFTEST_EXPORT=<folder> also exports a short MP4, then exits.
fn selftest(app: &mut App, ctx: &egui::Context) {
    let Ok(path) = std::env::var("FLUID_SELFTEST") else { return };
    ctx.request_repaint();
    let t = ctx.input(|i| i.time);
    if t > 120.0 {
        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
    }
    let ready = app.baker.cache.lock().unwrap().frontier >= 150;
    match app.selftest_stage {
        0 => {
            if ready && t > 2.0 {
                app.playhead = 2.5;
                if std::env::var("FLUID_SELFTEST_CLOSE").is_ok() {
                    app.proj.cam.dist = 4.0;
                    app.proj.cam.target = [1.0, 0.4, 0.0];
                    app.proj.show_helpers = false;                }
                app.selftest_stage = 1;
            }
        }
        1 => {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            app.selftest_stage = 2;
        }
        2 => {
            for ev in ctx.input(|i| i.raw.events.clone()) {
                if let egui::Event::Screenshot { image, .. } = ev {
                    let [w, h] = image.size;
                    let bytes: Vec<u8> = image.pixels.iter().flat_map(|c| c.to_array()).collect();
                    let _ = RgbaImage::from_raw(w as u32, h as u32, bytes).map(|i| i.save(&path));
                    if let Ok(dir) = std::env::var("FLUID_SELFTEST_EXPORT") {
                        app.proj.params.duration = 3.0;
                        app.proj.export.width = 960;
                        app.proj.export.height = 540;
                        app.proj.export.fps = 30;
                        app.proj.export.folder = dir;
                        app.start_export();
                        app.selftest_stage = 3;
                    } else {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            }
        }
        _ => {
            if let Some(p) = &app.export_prog {
                if p.finished.load(Ordering::SeqCst) {
                    eprintln!("export: {}", p.msg.lock().unwrap());
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            }
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        selftest(self, ctx);
        self.handle_keys(ctx);
        self.advance_playback(ctx);
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.add_space(4.0);
            self.toolbar(ui);
            ui.add_space(2.0);
        });
        egui::TopBottomPanel::bottom("timeline").resizable(true).default_height(200.0).min_height(110.0).show(ctx, |ui| self.timeline(ui));
        egui::SidePanel::left("left").resizable(true).default_width(300.0).show(ctx, |ui| {
            ScrollArea::vertical().show(ui, |ui| self.left_panel(ui));
        });
        egui::SidePanel::right("right").resizable(true).default_width(290.0).show(ctx, |ui| {
            ScrollArea::vertical().show(ui, |ui| self.right_panel(ui));
        });
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ctx, |ui| self.viewport(ui));
        self.export_window(ctx);
        self.check_dirty(ctx);
        self.periodic_autosave(ctx);
    }

    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        self.autosave();
        self.cancel_export();
        if let (Some(gl), Some(mut r)) = (gl, self.gl.lock().unwrap().renderer.take()) {
            r.destroy(gl);
        }
    }
}

fn main() -> eframe::Result {
    let mut vb = egui::ViewportBuilder::default()
        .with_title("Fluid Studio")
        .with_inner_size([1500.0, 920.0])
        .with_min_inner_size([960.0, 600.0]);
    if std::env::var("FLUID_SELFTEST").is_ok() {
        vb = vb.with_position([-4000.0, 100.0]).with_active(false).with_taskbar(false);
    }
    let options = eframe::NativeOptions { viewport: vb, renderer: eframe::Renderer::Glow, ..Default::default() };
    eframe::run_native("Fluid Studio", options, Box::new(|cc| Ok(Box::new(App::new(cc)))))
}
