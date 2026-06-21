# Aerie

A browser-based custom renderer that acts like **Bryce 2** (MetaCreations, 1996). Import
modern assets - glTF meshes, skeletons, PBR textures - and render them with the
dreamy, ray-traced look of the old days.

Aerie runs entirely in the browser on **WebGPU**, pairing a real-time editing
viewport with a custom compute-shader path tracer.

## Highlights

- **Procedural worlds** - fBm heightfield terrain, a volumetric sky with sun and
  clouds, reflective water, and distance haze, all editable live.
- **Path-traced renders** - realistic reflections and refractions, soft shadows,
  depth of field, and a filmic tonemap, accumulated progressively for a clean,
  noise-free final image.
- **Asset import** - glTF/GLB, FBX, OBJ, Collada (`.dae`), 3MF, 3DS, STL, and
  PLY, traced alongside the procedural world via a CPU-built BVH.
- **Full editor** - menu bar, scene list, and inspector, with viewport picking,
  drag-to-move, undo/redo, and `.aerie` scene files.
- **Animation & video** - a keyframe timeline for camera moves, plus turntable
  and cutscene rendering exported to WebM.

## Architecture

There is one scene graph that powers the two renderers:

- **Preview** - a real-time rasterized viewport for editing, built on Three.js.
- **Final render** - a custom WebGPU compute ray/path tracer (WGSL) that does the 
  hard work of looking like Bryce

This mirrors Bryce's own workflow: a fast wireframe preview, then a slow,
beautiful ray-traced render on demand.

## Features

### Procedural world (the "Labs")

Live editing panels drive the world-params:

- **Terrain** - amplitude, frequency, ridges, sea level, on/off.
- **Sky** - zenith/horizon colors, sun size, clouds, and a time-of-day slider
  (0–24h) that sweeps the sun's arc, color, and intensity from blue noon through
  golden hour to a starlit, moonlit night.
- **Water** - color, reflectivity, on/off.
- **Render** - exposure, warmth, haze, and depth of field.

### Rendering

- **Progressive path tracing** with reflections/refractions, soft shadows,
  volumetric sky, and aerial-perspective haze.
- **Depth of field** - thin-lens aperture and focal distance, plus
  click-to-focus in the viewport (the focal plane snaps onto the clicked
  surface). The blur builds up through accumulation.
- **Filmic tonemap** for the final image.
- **High-res render to PNG.**

### Assets & geometry

- **Import formats** - glTF/GLB, FBX, OBJ, Collada (`.dae`), 3MF, 3DS, STL, and
  PLY, via drag-drop or the Scene menu.
- **PBR materials** - base color + texture, metallic/roughness/emissive,
  glossy and normal-mapped surfaces.
- **Multi-model scenes** via a TLAS, with animation pose baking.
- **Ecosystem scatter** - populate a landform with a chosen shape, filtered by
  altitude band and max slope, with random scale and spin. All instances share
  one BLAS, so hundreds stay cheap.
- **Boolean/CSG primitives** and **mesh vertex editing** (weld → select/drag
  points → rebuild with smooth normals).
- **Material library** - a bottom dock of saved materials with rendered preview
  swatches. Drag a swatch onto an object to apply it, right-click to copy, and
  Copy/Paste/Save from the inspector. Presets persist across sessions.

### Editor

- **Menu bar** (Add / Tools / View / Scene / Labs) with a compact status line,
  a scene list on the left, and an inspector on the right.
- **Viewport picking**, drag-to-move, and draggable axis gizmos.
- **Undo/redo** over scene snapshots.
- **`.aerie` scene files** - New Scene resets to vanilla and guards unsaved
  changes.

### Animation & video

- **Turntable render** - spins the camera 360° around the target and exports a
  WebM (seconds / fps / samples / width). Each frame is fully accumulated
  off-screen and stamped via WebCodecs, so clip length is exactly `frames / fps`
  regardless of render time.
- **Cutscene mode** - a timeline dock for authoring camera moves. Pose the
  camera, DoF, and objects, then **Add keyframe** to capture the full state. Set
  per-segment timing and easing, then scrub, play, or render to WebM. The camera
  interpolates tumble-free (shortest-arc yaw); objects lerp position/scale and
  slerp rotation, so only the things you move animate. It's a non-destructive
  overlay - your canonical scene state is what gets saved, and objects snap back
  when you stop. Cutscenes persist in `.aerie`.

## Getting started

Aerie requires a **WebGPU-capable browser** (recent Chrome/Edge, or Firefox
Nightly).

```bash
npm install
npm run dev      # http://localhost:5173
```

The render refines progressively and stops once it converges.

To build for production:

```bash
npm run build
npm run preview
```

## Controls

- **Click** an object to select it; **drag** to move it on the ground (Shift =
  vertical). Drag empty space to **orbit**, wheel to **zoom**.
- **Arrow keys** fly; **Ctrl/Shift + arrows** rotate.
- **Menu bar** (top): **Add** primitives/landforms/lights, **Tools**
  (move/select + gizmo), **View** (snap, bookmarks, Save PNG), **Scene**
  (new/save/load/import), **Labs** (Terrain/Sky/Water/Render), the
  **Preview ⇄ Render** toggle, and **?** for shortcuts. The selected item is
  edited in the right-hand inspector.
- **Material dock** (bottom): drag a saved material onto an object to apply it,
  right-click a swatch to copy it.
- **Cutscene** (menu bar): toggle the timeline dock, **Add keyframe** to capture
  the camera/DoF/objects, scrub or **▶ Play** to preview, **Render WebM** to
  export.

## Project layout

```
src/
  main.ts              entry: device init, input, accumulation loop, status HUD
  ui.ts                menu bar, inspector, Labs, material library dock
  preview.ts           Three.js raster preview (editing viewport)
  scene/
    scene.ts           scene graph + world params; capture/restore for undo
    camera.ts          orbit camera (Three.js math)
    history.ts         undo/redo over scene snapshots
    cutscene.ts        camera/DoF/object keyframe model + interpolation
    serialize.ts       .aerie save/load
  mesh/                model import, BVH, tessellation, landforms, CSG, edit-mesh
  gen/                 procedural terrain recipes + noise (CPU mirror of the GPU)
  webgpu/renderer.ts   WebGPU pipelines, accumulation buffer, uniforms, picking
  shaders/
    raytrace.wgsl      the ray tracer (terrain, sky, water, prims, meshes, DoF)
    present.wgsl       average samples + filmic tonemap → screen
```

## Built with

[Three.js](https://threejs.org/) · [WebGPU](https://www.w3.org/TR/webgpu/) /
WGSL · [Vite](https://vitejs.dev/) · TypeScript
