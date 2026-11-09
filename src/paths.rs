//! Portable storage: everything lives next to FluidStudio.exe.
//! Paths inside that folder are stored relative so the folder can be moved.
//! Falls back to %APPDATA%\FluidStudio only if the exe folder isn't writable.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const SETTINGS_FILE: &str = "settings.json";

pub fn data_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| {
        if let Some(dir) = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)) {
            let probe = dir.join(".write_test");
            if std::fs::write(&probe, b"ok").is_ok() {
                let _ = std::fs::remove_file(&probe);
                return dir;
            }
        }
        let base = std::env::var("APPDATA").unwrap_or_else(|_| ".".into());
        let dir = Path::new(&base).join("FluidStudio");
        let _ = std::fs::create_dir_all(&dir);
        dir
    })
}

/// Sub-folder of the data dir, created on demand.
pub fn sub(name: &str) -> PathBuf {
    let d = data_dir().join(name);
    let _ = std::fs::create_dir_all(&d);
    d
}

/// Turn a stored path (relative to the program folder, or absolute) into a real path.
pub fn resolve(p: &str) -> PathBuf {
    let pb = PathBuf::from(p);
    if pb.is_absolute() {
        pb
    } else {
        data_dir().join(pb)
    }
}

/// Store paths inside the program folder as relative ones.
pub fn relativize(p: &Path) -> String {
    match p.strip_prefix(data_dir()) {
        Ok(r) => r.to_string_lossy().to_string(),
        Err(_) => p.to_string_lossy().to_string(),
    }
}

/// Copy an image into the program's Images folder (unless it's already inside
/// the program folder) and return the relative path to store.
pub fn import_image(src: &Path) -> Result<String, String> {
    if src.starts_with(data_dir()) {
        return Ok(relativize(src));
    }
    let dir = sub("Images");
    let name = src.file_name().ok_or("bad file name")?.to_string_lossy().to_string();
    let (stem, ext) = match name.rsplit_once('.') {
        Some((s, e)) => (s.to_string(), format!(".{e}")),
        None => (name.clone(), String::new()),
    };
    let src_len = std::fs::metadata(src).map(|m| m.len()).map_err(|e| format!("Can't read image: {e}"))?;
    for i in 0..1000 {
        let cand = dir.join(if i == 0 { name.clone() } else { format!("{stem}_{i}{ext}") });
        match std::fs::metadata(&cand) {
            // same file already imported -> reuse it
            Ok(m) if m.len() == src_len && std::fs::read(&cand).ok() == std::fs::read(src).ok() => return Ok(relativize(&cand)),
            Ok(_) => continue,
            Err(_) => {
                std::fs::copy(src, &cand).map_err(|e| format!("Can't copy image into the program folder: {e}"))?;
                return Ok(relativize(&cand));
            }
        }
    }
    Err("Too many images with the same name".into())
}

/// Old (non-portable) settings location, used once for migration.
pub fn legacy_settings() -> PathBuf {
    let base = std::env::var("APPDATA").unwrap_or_else(|_| ".".into());
    Path::new(&base).join("FluidStudio").join("autosave.json")
}
