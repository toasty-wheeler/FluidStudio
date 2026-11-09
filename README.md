# FluidStudio

A desktop fluid simulation studio written in Rust. Place emission points in a 3D scene, time them on a timeline, press Play, and export the result as a video.

![FluidStudio default interface](docs/screenshot.png)

## Features

- **3D scene editing:** click the floor to place emission points, drag them to move them, and drag the arrow tip to aim.
- **Per-point settings:** amount, length, start time, speed, spread, nozzle size, color and transparency.
- **Timeline:** drag bars to choose when each point fires, and drag the bar ends to change how long it fires.
- **Liquid rendering:** a screen-space fluid surface with refraction, reflections, absorption and colored shadows. Touching fluids blend their colors.
- **Ground and backdrop:** the ground is a checkerboard or any image (tiled, or one centered image the liquid can pour off). The backdrop is a sky gradient, a solid color or an image.
- **Export:** MP4, WebM or MOV (with transparency), or a PNG sequence, using the same GPU renderer as the viewport.
- **Portable:** settings, projects, imported images and exports are all stored in the program's own folder.

## Building

Requires Rust (stable) on Windows with the MSVC toolchain.

```
cargo build --release
```

The program is `target/release/fluid-studio.exe`.

## Video export

Video formats need [ffmpeg](https://ffmpeg.org/). Put `ffmpeg.exe` next to the program, or make sure it is on your `PATH`. You can also pick its location in the export window. PNG sequence export works without ffmpeg.

## Controls

| Action | Input |
| --- | --- |
| Orbit camera | Left-drag |
| Pan | Right-drag or Shift + drag |
| Zoom | Mouse wheel |
| Play / pause | Space |
| Toggle add-point mode | A |
| Delete selected point | Del |
| Duplicate selected point | Ctrl + D |
| Save project | Ctrl + S |
| Free (unsnapped) timeline drag | Hold Shift |
