//! GPU renderer: textured floor with coloured liquid shadows, backdrop, and
//! screen-space fluid rendering (sphere depth -> bilateral smoothing ->
//! thickness/colour accumulation -> refraction/reflection/absorption shading).
//! Used for both the live viewport and video export.

use crate::sim::{Frame, POS_SCALE};
use eframe::glow::{self, HasContext, PixelPackData, PixelUnpackData};
use glam::{Mat4, Vec3};
use image::RgbaImage;
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::Arc;

const SHADOW_RES: i32 = 1024;
const SHADOW_HALF: f32 = 14.0;
const ABSORB: f32 = 30.0;

// ------------------------------------------------------------------ shaders

const VS_BILLBOARD: &str = r#"#version 330 core
layout(location = 0) in vec2 a_corner;
layout(location = 1) in vec3 a_pos;
layout(location = 2) in vec4 a_col;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_radius;
uniform float u_pos_scale;
out vec2 v_uv;
out vec4 v_col;
out vec3 v_center;
void main() {
    vec4 vc = u_view * vec4(a_pos * u_pos_scale, 1.0);
    v_center = vc.xyz;
    vec4 vp = vc;
    vp.xy += a_corner * u_radius;
    gl_Position = u_proj * vp;
    v_uv = a_corner;
    v_col = a_col;
}
"#;

const FS_DEPTH: &str = r#"#version 330 core
in vec2 v_uv;
in vec3 v_center;
uniform mat4 u_proj;
uniform float u_radius;
out float o_depth;
void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;
    vec3 p = v_center + vec3(v_uv * u_radius, sqrt(1.0 - r2) * u_radius);
    vec4 c = u_proj * vec4(p, 1.0);
    gl_FragDepth = c.z / c.w * 0.5 + 0.5;
    o_depth = 1.0 / max(-p.z, 1e-4); // inverse depth: MAX blending keeps the nearest
}
"#;

const FS_THICK: &str = r#"#version 330 core
in vec2 v_uv;
in vec4 v_col;
uniform float u_radius;
uniform float u_thick_scale;
layout(location = 0) out vec4 o_col;
layout(location = 1) out float o_dens;
void main() {
    float r2 = dot(v_uv, v_uv);
    if (r2 > 1.0) discard;
    float t = 2.0 * sqrt(1.0 - r2) * u_radius * u_thick_scale;
    o_col = vec4(v_col.rgb * t, t);
    o_dens = t * v_col.a * v_col.a;
}
"#;

const VS_FULL: &str = r#"#version 330 core
out vec2 v_uv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    v_uv = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"#;

const FS_BLUR: &str = r#"#version 330 core
in vec2 v_uv;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform float u_rot;
uniform float u_world_r;
uniform float u_proj_scale;
uniform float u_sigma_z;
uniform int u_inv;
out float o;
float fetch(vec2 uv) {
    float r = texture(u_src, uv).r;
    return u_inv == 1 ? (r > 0.0 ? 1.0 / r : 0.0) : r;
}
void main() {
    float d = fetch(v_uv);
    if (d <= 0.0) { o = 0.0; return; }
    // 2D bilateral filter with golden-angle spiral taps (no separable streaks)
    float rpx = clamp(u_world_r * u_proj_scale / d, 1.0, 40.0);
    float sig = max(rpx * 0.5, 0.5);
    float sum = d, wsum = 1.0;
    const int N = 48;
    for (int i = 1; i <= N; i++) {
        float fi = float(i);
        float r = rpx * sqrt(fi / float(N));
        float a = fi * 2.39996 + u_rot;
        float s = fetch(v_uv + vec2(cos(a), sin(a)) * r * u_texel);
        if (s <= 0.0) continue;
        float dz = (s - d) / u_sigma_z;
        float w = exp(-r * r / (2.0 * sig * sig) - dz * dz);
        sum += s * w;
        wsum += w;
    }
    o = sum / wsum;
}
"#;

const FS_BACK: &str = r#"#version 330 core
in vec2 v_uv;
uniform int u_mode;
uniform vec3 u_top;
uniform vec3 u_horizon;
uniform vec3 u_solid;
uniform mat4 u_inv_vp;
uniform sampler2D u_img;
uniform vec4 u_rect;
uniform int u_tile;
out vec4 o;
void main() {
    if (u_mode == 1) { o = vec4(u_solid, 1.0); return; }
    if (u_mode == 2) {
        vec2 q = vec2((v_uv.x - u_rect.x) / u_rect.z, ((1.0 - v_uv.y) - u_rect.y) / u_rect.w);
        if (u_tile == 1) q = fract(q);
        else if (any(lessThan(q, vec2(0.0))) || any(greaterThan(q, vec2(1.0)))) { o = vec4(0.0, 0.0, 0.0, 1.0); return; }
        o = vec4(texture(u_img, q).rgb, 1.0);
        return;
    }
    vec4 a = u_inv_vp * vec4(v_uv * 2.0 - 1.0, -1.0, 1.0);
    vec4 b = u_inv_vp * vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
    vec3 r = normalize(b.xyz / b.w - a.xyz / a.w);
    o = vec4(mix(u_horizon, u_top, pow(clamp(r.y, 0.0, 1.0), 0.5)), 1.0);
}
"#;

const VS_FLOOR: &str = r#"#version 330 core
layout(location = 0) in vec3 a_pos;
uniform mat4 u_vp;
out vec3 v_world;
void main() {
    v_world = a_pos;
    gl_Position = u_vp * vec4(a_pos, 1.0);
}
"#;

const FS_FLOOR: &str = r#"#version 330 core
in vec3 v_world;
uniform sampler2D u_tex;
uniform sampler2D u_scol;
uniform sampler2D u_sdens;
uniform vec2 u_tile;
uniform float u_center;
uniform vec2 u_half;
uniform mat4 u_light_vp;
uniform vec3 u_eye;
uniform vec3 u_fog;
uniform float u_absorb;
uniform float u_texel;
out vec4 o;
void main() {
    if (u_half.x > 0.0 && (abs(v_world.x) > u_half.x || abs(v_world.z) > u_half.y)) discard;
    vec3 c = texture(u_tex, v_world.xz / u_tile + vec2(u_center)).rgb * 0.94;
    vec4 ls = u_light_vp * vec4(v_world, 1.0);
    vec2 suv = ls.xy * 0.5 + 0.5;
    if (all(greaterThan(suv, vec2(0.0))) && all(lessThan(suv, vec2(1.0)))) {
        vec4 sc = vec4(0.0);
        float sd = 0.0;
        for (int i = -2; i <= 2; i++)
        for (int j = -2; j <= 2; j++) {
            vec2 q = suv + vec2(i, j) * u_texel * 1.5;
            sc += texture(u_scol, q);
            sd += texture(u_sdens, q).r;
        }
        sc /= 25.0;
        sd /= 25.0;
        if (sc.a > 1e-5) {
            vec3 fc = sc.rgb / sc.a;
            float T = exp(-sd * u_absorb * 0.5);
            float Tc = exp(-sc.a * 4.0);
            c *= mix(fc * 0.45 + 0.08, mix(fc, vec3(1.0), 0.4), T) * (1.0 - Tc) + vec3(Tc);
        }
    }
    float dist = length(v_world - u_eye);
    c = mix(c, u_fog, smoothstep(25.0, 140.0, dist));
    o = vec4(c, 1.0);
}
"#;

const FS_COMP: &str = r#"#version 330 core
in vec2 v_uv;
uniform sampler2D u_scene;
uniform sampler2D u_depth;
uniform sampler2D u_tcol;
uniform sampler2D u_tdens;
uniform vec2 u_texel;
uniform mat4 u_proj;
uniform mat4 u_inv_view;
uniform vec3 u_top;
uniform vec3 u_horizon;
uniform vec3 u_ground;
uniform vec3 u_light_v;
uniform float u_absorb;
uniform float u_refr;
uniform float u_edge;
uniform float u_shine;
uniform int u_transparent;
out vec4 o;

vec3 vpos(vec2 uv, float d) {
    vec2 n = uv * 2.0 - 1.0;
    return vec3(n.x * d / u_proj[0][0], n.y * d / u_proj[1][1], -d);
}

vec3 env(vec3 r) {
    vec3 c = r.y > 0.0 ? mix(u_horizon, u_top, pow(clamp(r.y, 0.0, 1.0), 0.5))
                       : mix(u_horizon, u_ground, clamp(-r.y * 4.0, 0.0, 1.0));
    float box = smoothstep(0.90, 0.97, dot(r, normalize(vec3(-0.35, 0.85, 0.4))));
    float rim = smoothstep(0.93, 0.99, dot(r, normalize(vec3(0.6, 0.35, -0.7))));
    return c + vec3(1.3) * box * u_shine + vec3(0.6) * rim * u_shine;
}

void main() {
    vec4 scene = texture(u_scene, v_uv);
    float d = texture(u_depth, v_uv).r;
    vec4 tc = texture(u_tcol, v_uv);
    float th = tc.a;
    if (d <= 0.0 || th <= 1e-6) {
        o = u_transparent == 1 ? vec4(0.0) : vec4(scene.rgb, 1.0);
        return;
    }
    float dens = texture(u_tdens, v_uv).r;
    vec3 P = vpos(v_uv, d);
    vec2 dx = vec2(u_texel.x, 0.0), dy = vec2(0.0, u_texel.y);
    float dr = texture(u_depth, v_uv + dx).r, dl = texture(u_depth, v_uv - dx).r;
    float du = texture(u_depth, v_uv + dy).r, dd = texture(u_depth, v_uv - dy).r;
    vec3 ddx = vpos(v_uv + dx, dr) - P, ddx2 = P - vpos(v_uv - dx, dl);
    if (dr <= 0.0 || (dl > 0.0 && abs(ddx2.z) < abs(ddx.z))) ddx = ddx2;
    vec3 ddy = vpos(v_uv + dy, du) - P, ddy2 = P - vpos(v_uv - dy, dd);
    if (du <= 0.0 || (dd > 0.0 && abs(ddy2.z) < abs(ddy.z))) ddy = ddy2;
    vec3 N = cross(ddx, ddy);
    N = (dot(N, N) > 1e-20 && (dr > 0.0 || dl > 0.0) && (du > 0.0 || dd > 0.0)) ? normalize(N) : vec3(0.0, 0.0, 1.0);
    if (N.z < 0.0) N = -N;

    vec3 V = normalize(-P);
    vec3 col = tc.rgb / th;
    float T = exp(-dens * u_absorb);
    vec2 ruv = v_uv - N.xy * u_refr * clamp(th * 6.0, 0.0, 1.0);
    vec3 refr = texture(u_scene, ruv).rgb;
    float ndl = max(dot(N, u_light_v), 0.0);
    vec3 lit = col * (0.30 + 0.80 * ndl);
    vec3 trans = refr * mix(vec3(1.0), col, 0.65);
    vec3 body = mix(lit, trans, T);
    vec3 R = (u_inv_view * vec4(reflect(-V, N), 0.0)).xyz;
    float F = 0.03 + 0.97 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    vec3 H = normalize(u_light_v + V);
    float nh = max(dot(N, H), 0.0);
    float spec = (pow(nh, 400.0) * 3.0 + pow(nh, 60.0) * 0.2) * u_shine;
    vec3 e = env(R);
    // anti-aliased silhouette: fraction of nearby pixels covered by liquid
    float cov = 0.0;
    for (int i = 0; i < 8; i++) {
        float a = float(i) * 0.7854;
        cov += texture(u_depth, v_uv + vec2(cos(a), sin(a)) * u_texel * 1.5).r > 0.0 ? 1.0 : 0.0;
    }
    float edge = smoothstep(0.0, u_edge, th) * smoothstep(0.15, 0.75, cov / 8.0);
    if (u_transparent == 1) {
        vec3 cp = lit * (1.0 - T) * (1.0 - F) + e * F + vec3(spec);
        float a = clamp(1.0 - T * (1.0 - F) + spec, 0.0, 1.0) * edge;
        o = vec4(a > 1e-4 ? clamp(cp * edge / a, 0.0, 1.0) : vec3(0.0), a);
    } else {
        vec3 c = mix(body, e, F) + vec3(spec);
        o = vec4(mix(scene.rgb, c, edge), 1.0);
    }
}
"#;

// --------------------------------------------------------------- parameters

#[derive(Clone, Copy, Default)]
pub struct BackdropU {
    pub mode: i32, // 0 gradient, 1 solid, 2 image
    pub top: Vec3,
    pub horizon: Vec3,
    pub solid: Vec3,
    pub rect: [f32; 4], // image rect in uv (top-left origin)
    pub tile: bool,
}

#[derive(Clone, Copy, Default)]
pub struct RenderParams {
    pub view: Mat4,
    pub proj: Mat4,
    pub eye: Vec3,
    pub target: Vec3,
    pub radius: f32,
    pub h: f32,
    pub smooth: f32,
    pub refraction: f32,
    pub shine: f32,
    pub transparent: bool,
    pub show_floor: bool,
    pub floor_tile: [f32; 2],
    pub floor_center: f32,
    pub floor_half: [f32; 2],
    pub ground: Vec3,
    pub back: BackdropU,
}

pub struct TexSrc {
    pub key: u64,
    pub img: Arc<RgbaImage>,
}

pub struct GpuExport {
    pub rx: Receiver<Arc<Frame>>,
    pub tx: SyncSender<Vec<u8>>,
    pub params: RenderParams,
    pub w: i32,
    pub h: i32,
}

/// State shared between the UI thread and the paint callback.
#[derive(Default)]
pub struct GlShared {
    pub renderer: Option<Renderer>,
    pub error: Option<String>,
    pub pending: Option<Arc<Frame>>,
    pub params: RenderParams,
    pub floor_src: Option<TexSrc>,
    pub back_src: Option<TexSrc>,
    pub export: Option<GpuExport>,
    pub export_finished: bool,
}

// ------------------------------------------------------------------ helpers

unsafe fn compile(gl: &glow::Context, vs: &str, fs: &str) -> Result<glow::Program, String> {
    let prog = gl.create_program()?;
    let mut shaders = Vec::new();
    for (kind, src) in [(glow::VERTEX_SHADER, vs), (glow::FRAGMENT_SHADER, fs)] {
        let s = gl.create_shader(kind)?;
        gl.shader_source(s, src);
        gl.compile_shader(s);
        if !gl.get_shader_compile_status(s) {
            return Err(gl.get_shader_info_log(s));
        }
        gl.attach_shader(prog, s);
        shaders.push(s);
    }
    gl.link_program(prog);
    if !gl.get_program_link_status(prog) {
        return Err(gl.get_program_info_log(prog));
    }
    for s in shaders {
        gl.detach_shader(prog, s);
        gl.delete_shader(s);
    }
    Ok(prog)
}

unsafe fn make_tex(gl: &glow::Context, w: i32, h: i32, ifmt: u32, fmt: u32, ty: u32, filter: u32) -> Result<glow::Texture, String> {
    let t = gl.create_texture()?;
    gl.bind_texture(glow::TEXTURE_2D, Some(t));
    gl.tex_image_2d(glow::TEXTURE_2D, 0, ifmt as i32, w, h, 0, fmt, ty, PixelUnpackData::Slice(None));
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
    gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
    Ok(t)
}

unsafe fn make_fbo(gl: &glow::Context, colors: &[glow::Texture], depth: Option<glow::Renderbuffer>) -> Result<glow::Framebuffer, String> {
    let f = gl.create_framebuffer()?;
    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(f));
    let mut bufs = Vec::new();
    for (i, t) in colors.iter().enumerate() {
        let att = glow::COLOR_ATTACHMENT0 + i as u32;
        gl.framebuffer_texture_2d(glow::FRAMEBUFFER, att, glow::TEXTURE_2D, Some(*t), 0);
        bufs.push(att);
    }
    gl.draw_buffers(&bufs);
    if let Some(d) = depth {
        gl.framebuffer_renderbuffer(glow::FRAMEBUFFER, glow::DEPTH_ATTACHMENT, glow::RENDERBUFFER, Some(d));
    }
    let st = gl.check_framebuffer_status(glow::FRAMEBUFFER);
    if st != glow::FRAMEBUFFER_COMPLETE {
        return Err(format!("framebuffer incomplete: 0x{st:x}"));
    }
    Ok(f)
}

#[derive(Clone, Copy)]
struct Targets {
    w: i32,
    h: i32,
    depth_rb: glow::Renderbuffer,
    scene_tex: glow::Texture,
    scene_fbo: glow::Framebuffer,
    fdepth_tex: glow::Texture,
    fdepth_fbo: glow::Framebuffer,
    blur_tex: [glow::Texture; 2],
    blur_fbo: [glow::Framebuffer; 2],
    tcol: glow::Texture,
    tdens: glow::Texture,
    thick_fbo: glow::Framebuffer,
    out_tex: glow::Texture,
    out_fbo: glow::Framebuffer,
}

impl Targets {
    unsafe fn new(gl: &glow::Context, w: i32, h: i32) -> Result<Self, String> {
        let depth_rb = gl.create_renderbuffer()?;
        gl.bind_renderbuffer(glow::RENDERBUFFER, Some(depth_rb));
        gl.renderbuffer_storage(glow::RENDERBUFFER, glow::DEPTH_COMPONENT24, w, h);
        let scene_tex = make_tex(gl, w, h, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE, glow::LINEAR)?;
        let scene_fbo = make_fbo(gl, &[scene_tex], Some(depth_rb))?;
        let fdepth_tex = make_tex(gl, w, h, glow::R32F, glow::RED, glow::FLOAT, glow::NEAREST)?;
        let fdepth_fbo = make_fbo(gl, &[fdepth_tex], Some(depth_rb))?;
        let b0 = make_tex(gl, w, h, glow::R32F, glow::RED, glow::FLOAT, glow::NEAREST)?;
        let b1 = make_tex(gl, w, h, glow::R32F, glow::RED, glow::FLOAT, glow::NEAREST)?;
        let f0 = make_fbo(gl, &[b0], None)?;
        let f1 = make_fbo(gl, &[b1], None)?;
        let tcol = make_tex(gl, w, h, glow::RGBA16F, glow::RGBA, glow::FLOAT, glow::LINEAR)?;
        let tdens = make_tex(gl, w, h, glow::R16F, glow::RED, glow::FLOAT, glow::LINEAR)?;
        let thick_fbo = make_fbo(gl, &[tcol, tdens], Some(depth_rb))?;
        let out_tex = make_tex(gl, w, h, glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE, glow::NEAREST)?;
        let out_fbo = make_fbo(gl, &[out_tex], None)?;
        Ok(Self {
            w,
            h,
            depth_rb,
            scene_tex,
            scene_fbo,
            fdepth_tex,
            fdepth_fbo,
            blur_tex: [b0, b1],
            blur_fbo: [f0, f1],
            tcol,
            tdens,
            thick_fbo,
            out_tex,
            out_fbo,
        })
    }

    unsafe fn destroy(&self, gl: &glow::Context) {
        gl.delete_renderbuffer(self.depth_rb);
        for t in [self.scene_tex, self.fdepth_tex, self.blur_tex[0], self.blur_tex[1], self.tcol, self.tdens, self.out_tex] {
            gl.delete_texture(t);
        }
        for f in [self.scene_fbo, self.fdepth_fbo, self.blur_fbo[0], self.blur_fbo[1], self.thick_fbo, self.out_fbo] {
            gl.delete_framebuffer(f);
        }
    }
}

struct Particles {
    vao: glow::VertexArray,
    vbo_pos: glow::Buffer,
    vbo_col: glow::Buffer,
    count: i32,
}

impl Particles {
    unsafe fn new(gl: &glow::Context, quad: glow::Buffer) -> Result<Self, String> {
        let vao = gl.create_vertex_array()?;
        gl.bind_vertex_array(Some(vao));
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(quad));
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(0, 2, glow::FLOAT, false, 8, 0);
        let vbo_pos = gl.create_buffer()?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_pos));
        gl.buffer_data_size(glow::ARRAY_BUFFER, 6, glow::DYNAMIC_DRAW);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_f32(1, 3, glow::SHORT, false, 6, 0);
        gl.vertex_attrib_divisor(1, 1);
        let vbo_col = gl.create_buffer()?;
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo_col));
        gl.buffer_data_size(glow::ARRAY_BUFFER, 4, glow::DYNAMIC_DRAW);
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_f32(2, 4, glow::UNSIGNED_BYTE, true, 4, 0);
        gl.vertex_attrib_divisor(2, 1);
        gl.bind_vertex_array(None);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
        Ok(Self { vao, vbo_pos, vbo_col, count: 0 })
    }

    unsafe fn upload(&mut self, gl: &glow::Context, f: &Frame) {
        self.count = f.len() as i32;
        if self.count == 0 {
            return;
        }
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo_pos));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&f.pos), glow::DYNAMIC_DRAW);
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(self.vbo_col));
        gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&f.col), glow::DYNAMIC_DRAW);
        gl.bind_buffer(glow::ARRAY_BUFFER, None);
    }

    unsafe fn destroy(&self, gl: &glow::Context) {
        gl.delete_vertex_array(self.vao);
        gl.delete_buffer(self.vbo_pos);
        gl.delete_buffer(self.vbo_col);
    }
}

pub enum Output {
    Screen { fb: Option<glow::Framebuffer>, viewport: [i32; 4] },
    Offscreen,
}

pub struct Renderer {
    p_depth: glow::Program,
    p_thick: glow::Program,
    p_blur: glow::Program,
    p_back: glow::Program,
    p_floor: glow::Program,
    p_comp: glow::Program,
    quad: glow::Buffer,
    empty_vao: glow::VertexArray,
    floor_vao: glow::VertexArray,
    floor_vbo: glow::Buffer,
    parts: [Particles; 2],
    targets: [Option<Targets>; 2],
    shadow_col: glow::Texture,
    shadow_dens: glow::Texture,
    shadow_fbo: glow::Framebuffer,
    floor_tex: glow::Texture,
    floor_key: u64,
    back_tex: glow::Texture,
    back_key: u64,
    aniso: bool,
    thick_scale: f32,
}

pub const PREVIEW: usize = 0;
pub const EXPORT: usize = 1;

impl Renderer {
    pub fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let p_depth = compile(gl, VS_BILLBOARD, FS_DEPTH)?;
            let p_thick = compile(gl, VS_BILLBOARD, FS_THICK)?;
            let p_blur = compile(gl, VS_FULL, FS_BLUR)?;
            let p_back = compile(gl, VS_FULL, FS_BACK)?;
            let p_floor = compile(gl, VS_FLOOR, FS_FLOOR)?;
            let p_comp = compile(gl, VS_FULL, FS_COMP)?;

            let quad = gl.create_buffer()?;
            let corners: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(quad));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&corners), glow::STATIC_DRAW);

            let empty_vao = gl.create_vertex_array()?;
            let floor_vao = gl.create_vertex_array()?;
            gl.bind_vertex_array(Some(floor_vao));
            let floor_vbo = gl.create_buffer()?;
            let s = 300.0f32;
            let verts: [f32; 12] = [-s, 0.0, -s, s, 0.0, -s, -s, 0.0, s, s, 0.0, s];
            gl.bind_buffer(glow::ARRAY_BUFFER, Some(floor_vbo));
            gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, bytemuck::cast_slice(&verts), glow::STATIC_DRAW);
            gl.enable_vertex_attrib_array(0);
            gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 12, 0);
            gl.bind_vertex_array(None);

            let parts = [Particles::new(gl, quad)?, Particles::new(gl, quad)?];

            let shadow_col = make_tex(gl, SHADOW_RES, SHADOW_RES, glow::RGBA16F, glow::RGBA, glow::FLOAT, glow::LINEAR)?;
            let shadow_dens = make_tex(gl, SHADOW_RES, SHADOW_RES, glow::R16F, glow::RED, glow::FLOAT, glow::LINEAR)?;
            let shadow_fbo = make_fbo(gl, &[shadow_col, shadow_dens], None)?;

            let floor_tex = gl.create_texture()?;
            let back_tex = gl.create_texture()?;
            for t in [floor_tex, back_tex] {
                gl.bind_texture(glow::TEXTURE_2D, Some(t));
                let px = [200u8, 200, 200, 255];
                gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, 1, 1, 0, glow::RGBA, glow::UNSIGNED_BYTE, PixelUnpackData::Slice(Some(&px)));
                gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
            }
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.bind_texture(glow::TEXTURE_2D, None);

            let ext = gl.supported_extensions();
            let aniso = ext.contains("GL_EXT_texture_filter_anisotropic") || ext.contains("GL_ARB_texture_filter_anisotropic");

            Ok(Self {
                p_depth,
                p_thick,
                p_blur,
                p_back,
                p_floor,
                p_comp,
                quad,
                empty_vao,
                floor_vao,
                floor_vbo,
                parts,
                targets: [None, None],
                shadow_col,
                shadow_dens,
                shadow_fbo,
                floor_tex,
                floor_key: 0,
                back_tex,
                back_key: 0,
                aniso,
                thick_scale: 1.0,
            })
        }
    }

    pub fn destroy(&mut self, gl: &glow::Context) {
        unsafe {
            for p in [self.p_depth, self.p_thick, self.p_blur, self.p_back, self.p_floor, self.p_comp] {
                gl.delete_program(p);
            }
            gl.delete_buffer(self.quad);
            gl.delete_buffer(self.floor_vbo);
            gl.delete_vertex_array(self.empty_vao);
            gl.delete_vertex_array(self.floor_vao);
            for p in &self.parts {
                p.destroy(gl);
            }
            for t in self.targets.iter().flatten() {
                t.destroy(gl);
            }
            gl.delete_texture(self.shadow_col);
            gl.delete_texture(self.shadow_dens);
            gl.delete_framebuffer(self.shadow_fbo);
            gl.delete_texture(self.floor_tex);
            gl.delete_texture(self.back_tex);
        }
    }

    unsafe fn upload_image(&self, gl: &glow::Context, tex: glow::Texture, img: &RgbaImage, mip: bool) {
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
        gl.tex_image_2d(
            glow::TEXTURE_2D,
            0,
            glow::RGBA8 as i32,
            img.width() as i32,
            img.height() as i32,
            0,
            glow::RGBA,
            glow::UNSIGNED_BYTE,
            PixelUnpackData::Slice(Some(img.as_raw())),
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::REPEAT as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::REPEAT as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::LINEAR as i32);
        if mip {
            gl.generate_mipmap(glow::TEXTURE_2D);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR_MIPMAP_LINEAR as i32);
            if self.aniso {
                gl.tex_parameter_f32(glow::TEXTURE_2D, 0x84FE, 16.0);
            }
        } else {
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::LINEAR as i32);
        }
        gl.bind_texture(glow::TEXTURE_2D, None);
    }

    pub fn sync_textures(&mut self, gl: &glow::Context, floor: Option<&TexSrc>, back: Option<&TexSrc>) {
        unsafe {
            if let Some(f) = floor {
                if f.key != self.floor_key {
                    self.upload_image(gl, self.floor_tex, &f.img, true);
                    self.floor_key = f.key;
                }
            }
            if let Some(b) = back {
                if b.key != self.back_key {
                    self.upload_image(gl, self.back_tex, &b.img, false);
                    self.back_key = b.key;
                }
            }
        }
    }

    pub fn upload(&mut self, gl: &glow::Context, which: usize, f: &Frame) {
        unsafe { self.parts[which].upload(gl, f) }
    }

    fn light(p: &RenderParams) -> (Vec3, Mat4, Mat4) {
        let l = Vec3::new(-0.35, 1.0, 0.25).normalize();
        let c = Vec3::new(p.target.x, 0.0, p.target.z);
        let view = Mat4::look_at_rh(c + l * 40.0, c, Vec3::Z);
        let proj = Mat4::orthographic_rh_gl(-SHADOW_HALF, SHADOW_HALF, -SHADOW_HALF, SHADOW_HALF, 1.0, 100.0);
        (l, view, proj)
    }

    unsafe fn bind_tex(gl: &glow::Context, prog: glow::Program, unit: u32, name: &str, tex: glow::Texture) {
        gl.active_texture(glow::TEXTURE0 + unit);
        gl.bind_texture(glow::TEXTURE_2D, Some(tex));
        gl.uniform_1_i32(gl.get_uniform_location(prog, name).as_ref(), unit as i32);
    }

    unsafe fn u_mat(gl: &glow::Context, prog: glow::Program, name: &str, m: &Mat4) {
        gl.uniform_matrix_4_f32_slice(gl.get_uniform_location(prog, name).as_ref(), false, &m.to_cols_array());
    }
    unsafe fn u_f(gl: &glow::Context, prog: glow::Program, name: &str, v: f32) {
        gl.uniform_1_f32(gl.get_uniform_location(prog, name).as_ref(), v);
    }
    unsafe fn u_i(gl: &glow::Context, prog: glow::Program, name: &str, v: i32) {
        gl.uniform_1_i32(gl.get_uniform_location(prog, name).as_ref(), v);
    }
    unsafe fn u_v3(gl: &glow::Context, prog: glow::Program, name: &str, v: Vec3) {
        gl.uniform_3_f32(gl.get_uniform_location(prog, name).as_ref(), v.x, v.y, v.z);
    }
    unsafe fn u_v2(gl: &glow::Context, prog: glow::Program, name: &str, x: f32, y: f32) {
        gl.uniform_2_f32(gl.get_uniform_location(prog, name).as_ref(), x, y);
    }

    unsafe fn draw_particles(&self, gl: &glow::Context, prog: glow::Program, which: usize, view: &Mat4, proj: &Mat4, radius: f32) {
        let parts = &self.parts[which];
        if parts.count == 0 {
            return;
        }
        gl.use_program(Some(prog));
        Self::u_mat(gl, prog, "u_view", view);
        Self::u_mat(gl, prog, "u_proj", proj);
        Self::u_f(gl, prog, "u_radius", radius);
        Self::u_f(gl, prog, "u_thick_scale", self.thick_scale);
        Self::u_f(gl, prog, "u_pos_scale", 1.0 / POS_SCALE);
        gl.bind_vertex_array(Some(parts.vao));
        gl.draw_arrays_instanced(glow::TRIANGLE_STRIP, 0, 4, parts.count);
    }

    /// Render one image. `which` selects particle buffers + targets (PREVIEW/EXPORT).
    pub fn render(&mut self, gl: &glow::Context, which: usize, w: i32, h: i32, p: &RenderParams, out: Output) -> Result<(), String> {
        if w < 2 || h < 2 {
            return Ok(());
        }
        unsafe {
            if self.targets[which].is_none_or(|t| t.w != w || t.h != h) {
                if let Some(t) = self.targets[which].take() {
                    t.destroy(gl);
                }
                self.targets[which] = Some(Targets::new(gl, w, h)?);
            }
            let t = self.targets[which].unwrap();
            let (ldir, lview, lproj) = Self::light(p);
            let light_vp = lproj * lview;
            let has_fluid = self.parts[which].count > 0;
            // overlapping spheres over-count depth: scale by real particle volume share
            self.thick_scale = ((0.48 * p.h).powi(3) / (4.18879 * p.radius.powi(3))).clamp(0.02, 1.0);
            let scene_visible = !p.transparent;

            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::CULL_FACE);
            gl.disable(glow::BLEND);
            gl.color_mask(true, true, true, true);
            gl.depth_mask(true);
            gl.depth_func(glow::LESS);
            gl.disable(glow::DEPTH_TEST);

            // 1. shadow thickness from the light
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(self.shadow_fbo));
            gl.viewport(0, 0, SHADOW_RES, SHADOW_RES);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            if has_fluid && p.show_floor && scene_visible {
                gl.enable(glow::BLEND);
                gl.blend_equation(glow::FUNC_ADD);
                gl.blend_func(glow::ONE, glow::ONE);
                self.draw_particles(gl, self.p_thick, which, &lview, &lproj, p.radius);
                gl.disable(glow::BLEND);
            }

            // 2. scene: backdrop + floor
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(t.scene_fbo));
            gl.viewport(0, 0, w, h);
            gl.clear_color(0.0, 0.0, 0.0, 0.0);
            gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT);
            let vp = p.proj * p.view;
            if scene_visible {
                let pr = self.p_back;
                gl.use_program(Some(pr));
                Self::u_i(gl, pr, "u_mode", p.back.mode);
                Self::u_v3(gl, pr, "u_top", p.back.top);
                Self::u_v3(gl, pr, "u_horizon", p.back.horizon);
                Self::u_v3(gl, pr, "u_solid", p.back.solid);
                Self::u_mat(gl, pr, "u_inv_vp", &vp.inverse());
                let r = p.back.rect;
                gl.uniform_4_f32(gl.get_uniform_location(pr, "u_rect").as_ref(), r[0], r[1], r[2], r[3]);
                Self::u_i(gl, pr, "u_tile", p.back.tile as i32);
                Self::bind_tex(gl, pr, 0, "u_img", self.back_tex);
                gl.bind_vertex_array(Some(self.empty_vao));
                gl.draw_arrays(glow::TRIANGLES, 0, 3);

                if p.show_floor {
                    gl.enable(glow::DEPTH_TEST);
                    let pr = self.p_floor;
                    gl.use_program(Some(pr));
                    Self::u_mat(gl, pr, "u_vp", &vp);
                    Self::u_mat(gl, pr, "u_light_vp", &light_vp);
                    Self::u_v2(gl, pr, "u_tile", p.floor_tile[0], p.floor_tile[1]);
                    Self::u_f(gl, pr, "u_center", p.floor_center);
                    Self::u_v2(gl, pr, "u_half", p.floor_half[0], p.floor_half[1]);
                    Self::u_v3(gl, pr, "u_eye", p.eye);
                    Self::u_v3(gl, pr, "u_fog", p.back.horizon);
                    Self::u_f(gl, pr, "u_absorb", ABSORB);
                    Self::u_f(gl, pr, "u_texel", 1.0 / SHADOW_RES as f32);
                    Self::bind_tex(gl, pr, 0, "u_tex", self.floor_tex);
                    Self::bind_tex(gl, pr, 1, "u_scol", self.shadow_col);
                    Self::bind_tex(gl, pr, 2, "u_sdens", self.shadow_dens);
                    gl.bind_vertex_array(Some(self.floor_vao));
                    gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
                    gl.disable(glow::DEPTH_TEST);
                }
            }

            // 3. fluid surface depth (depth-tested against the floor)
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(t.fdepth_fbo));
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.enable(glow::DEPTH_TEST);
            gl.depth_mask(false); // keep the depth buffer = floor only
            gl.enable(glow::BLEND);
            gl.blend_equation(glow::MAX);
            self.draw_particles(gl, self.p_depth, which, &p.view, &p.proj, p.radius);
            gl.blend_equation(glow::FUNC_ADD);
            gl.disable(glow::BLEND);
            gl.depth_mask(true);
            gl.disable(glow::DEPTH_TEST);

            // 4. bilateral smoothing of the depth (first pass also converts inverse -> linear depth)
            let mut src = t.fdepth_tex;
            {
                let pr = self.p_blur;
                gl.use_program(Some(pr));
                gl.bind_vertex_array(Some(self.empty_vao));
                Self::u_f(gl, pr, "u_world_r", p.radius * 1.4 * p.smooth);
                Self::u_f(gl, pr, "u_proj_scale", p.proj.y_axis.y * h as f32 * 0.5);
                Self::u_f(gl, pr, "u_sigma_z", p.radius * 1.2);
                let iters = if p.smooth > 0.0 && has_fluid { 3 } else { 1 };
                Self::u_v2(gl, pr, "u_texel", 1.0 / w as f32, 1.0 / h as f32);
                for i in 0..iters {
                    let k = i % 2;
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(t.blur_fbo[k]));
                    Self::bind_tex(gl, pr, 0, "u_src", src);
                    Self::u_f(gl, pr, "u_rot", i as f32 * 1.3);
                    Self::u_i(gl, pr, "u_inv", (i == 0) as i32);
                    gl.draw_arrays(glow::TRIANGLES, 0, 3);
                    src = t.blur_tex[k];
                }
            }

            // 5. thickness + colour accumulation (only in front of the floor)
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(t.thick_fbo));
            gl.clear(glow::COLOR_BUFFER_BIT);
            if has_fluid {
                gl.enable(glow::DEPTH_TEST);
                gl.depth_mask(false);
                gl.enable(glow::BLEND);
                gl.blend_equation(glow::FUNC_ADD);
                gl.blend_func(glow::ONE, glow::ONE);
                self.draw_particles(gl, self.p_thick, which, &p.view, &p.proj, p.radius);
                gl.disable(glow::BLEND);
                gl.depth_mask(true);
                gl.disable(glow::DEPTH_TEST);
            }

            // 6. composite
            match out {
                Output::Screen { fb, viewport } => {
                    gl.bind_framebuffer(glow::FRAMEBUFFER, fb);
                    gl.viewport(viewport[0], viewport[1], viewport[2], viewport[3]);
                    gl.enable(glow::SCISSOR_TEST);
                }
                Output::Offscreen => {
                    gl.bind_framebuffer(glow::FRAMEBUFFER, Some(t.out_fbo));
                    gl.viewport(0, 0, w, h);
                }
            }
            let pr = self.p_comp;
            gl.use_program(Some(pr));
            Self::bind_tex(gl, pr, 0, "u_scene", t.scene_tex);
            Self::bind_tex(gl, pr, 1, "u_depth", src);
            Self::bind_tex(gl, pr, 2, "u_tcol", t.tcol);
            Self::bind_tex(gl, pr, 3, "u_tdens", t.tdens);
            Self::u_v2(gl, pr, "u_texel", 1.0 / w as f32, 1.0 / h as f32);
            Self::u_mat(gl, pr, "u_proj", &p.proj);
            Self::u_mat(gl, pr, "u_inv_view", &p.view.inverse());
            Self::u_v3(gl, pr, "u_top", p.back.top);
            Self::u_v3(gl, pr, "u_horizon", p.back.horizon);
            Self::u_v3(gl, pr, "u_ground", p.ground);
            Self::u_v3(gl, pr, "u_light_v", p.view.transform_vector3(ldir).normalize());
            Self::u_f(gl, pr, "u_absorb", ABSORB);
            Self::u_f(gl, pr, "u_refr", p.refraction);
            Self::u_f(gl, pr, "u_edge", p.radius * 1.3 * self.thick_scale);
            Self::u_f(gl, pr, "u_shine", p.shine);
            Self::u_i(gl, pr, "u_transparent", p.transparent as i32);
            gl.bind_vertex_array(Some(self.empty_vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            gl.bind_vertex_array(None);
            gl.use_program(None);
            for u in 0..4 {
                gl.active_texture(glow::TEXTURE0 + u);
                gl.bind_texture(glow::TEXTURE_2D, None);
            }
            gl.active_texture(glow::TEXTURE0);
        }
        Ok(())
    }

    /// Read back the last offscreen render (top-down RGBA rows).
    pub fn read_export(&self, gl: &glow::Context) -> Vec<u8> {
        let Some(t) = self.targets[EXPORT] else { return Vec::new() };
        let (w, h) = (t.w as usize, t.h as usize);
        let mut buf = vec![0u8; w * h * 4];
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(t.out_fbo));
            gl.pixel_store_i32(glow::PACK_ALIGNMENT, 1);
            gl.read_pixels(0, 0, t.w, t.h, glow::RGBA, glow::UNSIGNED_BYTE, PixelPackData::Slice(Some(&mut buf)));
        }
        // flip vertically
        let row = w * 4;
        let mut out = vec![0u8; buf.len()];
        for y in 0..h {
            out[y * row..(y + 1) * row].copy_from_slice(&buf[(h - 1 - y) * row..(h - y) * row]);
        }
        out
    }
}

/// Run inside the egui paint callback.
pub fn paint(shared: &std::sync::Mutex<GlShared>, gl: &glow::Context, info: &eframe::egui::PaintCallbackInfo) {
    let mut guard = shared.lock().unwrap();
    let s = &mut *guard;
    if s.renderer.is_none() && s.error.is_none() {
        match Renderer::new(gl) {
            Ok(r) => s.renderer = Some(r),
            Err(e) => s.error = Some(e),
        }
    }
    let Some(r) = s.renderer.as_mut() else { return };
    let fb_id = unsafe { gl.get_parameter_i32(glow::FRAMEBUFFER_BINDING) } as u32;
    let fb = std::num::NonZeroU32::new(fb_id).map(glow::NativeFramebuffer);
    r.sync_textures(gl, s.floor_src.as_ref(), s.back_src.as_ref());
    if let Some(f) = s.pending.take() {
        r.upload(gl, PREVIEW, &f);
    }

    // export frames (time-boxed so the UI stays responsive)
    let mut drop_export = false;
    if let Some(ex) = s.export.as_ref() {
        let t0 = std::time::Instant::now();
        while t0.elapsed().as_millis() < 30 {
            match ex.rx.try_recv() {
                Ok(frame) => {
                    r.upload(gl, EXPORT, &frame);
                    if let Err(e) = r.render(gl, EXPORT, ex.w, ex.h, &ex.params, Output::Offscreen) {
                        s.error = Some(e);
                        drop_export = true;
                        break;
                    }
                    let px = r.read_export(gl);
                    if ex.tx.send(px).is_err() {
                        drop_export = true;
                        break;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    drop_export = true;
                    s.export_finished = true;
                    break;
                }
            }
        }
    }
    if drop_export {
        s.export = None; // closes the writer channel
    }

    let vp = info.viewport_in_pixels();
    let params = s.params;
    if let Err(e) = r.render(
        gl,
        PREVIEW,
        vp.width_px,
        vp.height_px,
        &params,
        Output::Screen { fb, viewport: [vp.left_px, vp.from_bottom_px, vp.width_px, vp.height_px] },
    ) {
        s.error = Some(e);
    }
    unsafe { gl.bind_framebuffer(glow::FRAMEBUFFER, fb) };
}
