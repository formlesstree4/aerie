# Aerie

<img width="1920" height="959" alt="aerie-1782789849542" src="https://github.com/user-attachments/assets/f81fb768-8624-4046-8fb4-48c263c52fa3" />

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

## Docker

```bash
docker compose up -d --build     # http://localhost:8080, https://localhost:8443
```

### WebGPU needs a secure context

Browsers only expose `navigator.gpu` on a **secure origin**: HTTPS, or plain
HTTP on `localhost`/`127.0.0.1`. Reaching the container at
`http://192.168.1.50:8080` hides WebGPU entirely and Aerie shows the
"not a secure context" error — no browser setting will fix it. Browsing from
the Docker host itself over `http://localhost:8080` is fine.

For everything else use the HTTPS port. The container generates a self-signed
certificate on first start, so there is nothing to do but tell it what names
you reach the box by:

```bash
AERIE_TLS_HOSTS=aerie.lan,192.168.1.50 docker compose up -d --build
```

Those become the certificate's SANs (`localhost` and `127.0.0.1` are always
included). The cert lands in `./certs/`, which is mounted into the container, so
it survives recreates and only has to be trusted once. Delete `./certs/` and
restart to mint a fresh one — do that if you change `AERIE_TLS_HOSTS`, since an
existing cert is reused as-is.

Already have a real certificate (Let's Encrypt, an internal CA)? Drop it in as
`certs/aerie.crt` + `certs/aerie.key` and the generator steps aside.

### Trusting the self-signed cert

The browser will warn on first visit because nothing vouches for the cert. Either
click through (**Advanced → Proceed**, per browser, and it sticks), or install it
so the warning stops for good:

```bash
docker compose cp aerie:/etc/nginx/certs/aerie.crt ./aerie.crt   # or just use ./certs/aerie.crt
```

- **Windows** — `Import-Certificate -FilePath .\aerie.crt -CertStoreLocation Cert:\CurrentUser\Root`
  (elevated PowerShell), then restart the browser. Chrome, Edge, and Brave use
  this store; Firefox does not.
- **Firefox / Firefox Nightly** — Settings → Privacy & Security → Certificates →
  *View Certificates* → **Servers** → *Import*, or just accept the warning.
- **macOS** — `sudo security add-trusted-cert -d -r trustRoot -k /Library/Keychains/System.keychain aerie.crt`
- **Linux** — copy to `/usr/local/share/ca-certificates/aerie.crt`, then
  `sudo update-ca-certificates` (Chrome also reads its own NSS store via
  `certutil -d sql:$HOME/.pki/nssdb -A -t "P,," -n aerie -i aerie.crt`).

The certificate is valid for 825 days, so re-mint it before then.

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
