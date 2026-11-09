//! Backdrop (behind everything) and ground texture settings.

use image::RgbaImage;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum BackMode {
    Gradient,
    Solid,
    Image,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum FloorMode {
    Checker,
    Image,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum Fit {
    Cover,
    Contain,
    Stretch,
    Center,
    Tile,
}

impl Fit {
    pub const ALL: [Fit; 5] = [Fit::Cover, Fit::Contain, Fit::Stretch, Fit::Center, Fit::Tile];
    pub fn name(self) -> &'static str {
        match self {
            Fit::Cover => "Fill (crop)",
            Fit::Contain => "Fit (letterbox)",
            Fit::Stretch => "Stretch",
            Fit::Center => "Center (1:1)",
            Fit::Tile => "Tile",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BgSettings {
    // backdrop
    pub back_mode: BackMode,
    pub top: [u8; 3],
    pub horizon: [u8; 3],
    pub solid: [u8; 3],
    pub back_image: Option<String>,
    pub fit: Fit,
    // ground
    pub floor_mode: FloorMode,
    pub cell: f32, // checker square size in metres
    pub c1: [u8; 3],
    pub c2: [u8; 3],
    pub floor_image: Option<String>,
    pub floor_image_m: f32, // width of the image on the ground in metres
    pub floor_tiled: bool,
}

impl Default for BgSettings {
    fn default() -> Self {
        Self {
            back_mode: BackMode::Gradient,
            top: [150, 172, 200],
            horizon: [226, 230, 236],
            solid: [40, 44, 52],
            back_image: None,
            fit: Fit::Cover,
            floor_mode: FloorMode::Checker,
            cell: 0.5,
            c1: [255, 255, 255],
            c2: [0, 0, 0],
            floor_image: None,
            floor_image_m: 4.0,
            floor_tiled: true,
        }
    }
}

/// Where an iw x ih image lands in a cw x ch container: [x, y, w, h].
/// For `Tile`, returns the size of one tile placed at the origin.
pub fn layout(fit: Fit, cw: f32, ch: f32, iw: f32, ih: f32, scale: f32) -> [f32; 4] {
    let centered = |s: f32| {
        let (w, h) = (iw * s, ih * s);
        [(cw - w) * 0.5, (ch - h) * 0.5, w, h]
    };
    match fit {
        Fit::Cover => centered((cw / iw).max(ch / ih)),
        Fit::Contain => centered((cw / iw).min(ch / ih)),
        Fit::Stretch => [0.0, 0.0, cw, ch],
        Fit::Center => centered(scale),
        Fit::Tile => [0.0, 0.0, iw * scale, ih * scale],
    }
}

/// 256x256 texture holding 2x2 checker squares (repeats every 2 cells).
pub fn checker_image(c1: [u8; 3], c2: [u8; 3]) -> RgbaImage {
    RgbaImage::from_fn(256, 256, |x, y| {
        let c = if ((x / 128) + (y / 128)) % 2 == 0 { c1 } else { c2 };
        image::Rgba([c[0], c[1], c[2], 255])
    })
}

pub fn average_color(img: &RgbaImage) -> [f32; 3] {
    let step = ((img.width() * img.height()) as usize / 4096).max(1);
    let mut s = [0.0f64; 3];
    let mut n = 0.0;
    for p in img.pixels().step_by(step) {
        for k in 0..3 {
            s[k] += p.0[k] as f64;
        }
        n += 1.0;
    }
    [(s[0] / n / 255.0) as f32, (s[1] / n / 255.0) as f32, (s[2] / n / 255.0) as f32]
}
