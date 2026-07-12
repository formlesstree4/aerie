import {
  WebGLRenderer,
  Scene as ThreeScene,
  PerspectiveCamera,
  DirectionalLight,
  HemisphereLight,
  Mesh,
  PlaneGeometry,
  MeshStandardMaterial,
  Color,
  Fog,
  AnimationMixer,
  AnimationClip,
  Clock,
  Group,
  Object3D,
  MathUtils,
  DoubleSide,
  Float32BufferAttribute,
  Points,
  BufferGeometry,
  BufferAttribute,
  ShaderMaterial,
  AdditiveBlending,
  DynamicDrawUsage,
} from "three";
import type { OrbitCamera } from "./scene/camera";
import { Scene as AppScene, LightType, type BLAS } from "./scene/scene";
import { previewFromBlas } from "./scene/serialize";
import { worldTerrainHeight, worldTerrainColor } from "./gen/terrainField";
import { primitiveLocalGeometry } from "./mesh/tessellate";
import { evaluateEmitter, MAX_PARTICLES, PARTICLE_FLOATS } from "./gen/particles";

/**
 * Real-time rasterized preview (Bryce's fast "nanopreview"). Renders the
 * imported model — textured and animated live — over a ground plane, with the
 * sun synced from the app scene. The ray-traced view is the slow final render.
 */
export class Preview {
  private renderer: WebGLRenderer;
  private scene = new ThreeScene();
  private camera = new PerspectiveCamera(50, 1, 0.1, 4000);
  private sun = new DirectionalLight(0xfff3e0, 3);
  private instances = new Map<
    number,
    { group: Group; inner: Object3D; mixer: AnimationMixer | null; blas: BLAS | null }
  >();
  private posing = new Set<number>(); // instances whose animation is paused for manual posing
  private clock = new Clock();
  // Procedural world surfaces mirrored from the ray tracer so placement in the
  // preview matches the final render. Built lazily / on worldVersion change.
  private ground: Mesh;                 // flat fallback when terrain is disabled
  private terrainMesh: Mesh | null = null;
  private waterMesh: Mesh | null = null;
  private terrainSig = "";               // shape params + patch center the heights were last built for
  private terrainColorSig = "";          // palette/threshold params the vertex colors were last built for
  // SDF primitives rasterized as tessellated meshes, keyed by primitive id.
  private prims = new Map<number, { mesh: Mesh; sig: string }>();
  // Live particle field (emitters). Additive round points, world-sized, evaluated
  // from the same stateless model the ray tracer uses so preview ≈ final render.
  private particles: Points | null = null;
  private particleScratch = new Float32Array(MAX_PARTICLES * PARTICLE_FLOATS);
  private particleTime = 0; // preview-local clock (loops so one-shots replay)
  private static readonly TERRAIN_SIZE = 3000; // world units the patch covers
  private static readonly TERRAIN_SEG = 200;   // grid resolution (segments per side)
  private static readonly TERRAIN_SNAP = 250;  // patch re-centers in steps this large

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly app: AppScene,
  ) {
    this.renderer = new WebGLRenderer({ canvas, antialias: true });
    this.scene.background = new Color(0xc7d6eb);
    this.scene.fog = new Fog(0xc7d6eb, 120, 900);

    this.ground = new Mesh(
      new PlaneGeometry(4000, 4000),
      new MeshStandardMaterial({ color: 0x5b6b45, roughness: 1 }),
    );
    this.ground.rotation.x = -Math.PI / 2;
    this.scene.add(this.ground);

    this.scene.add(new HemisphereLight(0xbcd0ff, 0x52502f, 0.6));
    this.sun.position.set(50, 60, 30);
    this.scene.add(this.sun, this.sun.target);
  }

  addInstance(id: number, group: Group, animations: AnimationClip[], blas?: BLAS | null): void {
    const mixer = animations.length ? new AnimationMixer(group) : null;
    mixer?.clipAction(animations[0]).play();
    this.scene.add(group);
    // Remember the geometry this group was built from, so syncInstances can
    // detect when an undo/redo (or edit/bake) swapped the BLAS and rebuild.
    const inst = this.app.instances.find((i) => i.id === id);
    const resolvedBlas = blas !== undefined ? blas : inst ? this.app.blases[inst.blasIndex] ?? null : null;
    this.instances.set(id, { group, inner: group.children[0], mixer, blas: resolvedBlas });
  }

  /** Skeleton bones of an instance (for posing), or null if it isn't rigged. */
  getBones(id: number): Object3D[] | null {
    const e = this.instances.get(id);
    if (!e) return null;
    let bones: Object3D[] | null = null;
    e.inner.traverse((o) => {
      const sk = o as unknown as { isSkinnedMesh?: boolean; skeleton?: { bones: Object3D[] } };
      if (sk.isSkinnedMesh && sk.skeleton && !bones) bones = sk.skeleton.bones;
    });
    return bones;
  }

  /** The instance's transformed scene-graph root (for world-geometry extraction). */
  instanceRoot(id: number): Object3D | null {
    return this.instances.get(id)?.group ?? null;
  }

  /** Pause/resume an instance's animation so manual bone edits aren't overwritten. */
  setPosing(id: number, posing: boolean): void {
    if (posing) this.posing.add(id);
    else this.posing.delete(id);
  }

  isPosing(id: number): boolean {
    return this.posing.has(id);
  }

  /** Put an instance's geometry in LOCAL space (instance transform removed) and
   *  return its root, so the current animation pose can be re-baked. */
  prepareBake(id: number): Object3D | null {
    const e = this.instances.get(id);
    if (!e) return null;
    e.group.position.set(0, 0, 0);
    e.group.quaternion.identity();
    e.group.scale.setScalar(1);
    e.group.updateMatrixWorld(true);
    return e.inner;
  }

  removeInstance(id: number): void {
    const e = this.instances.get(id);
    if (e) {
      this.scene.remove(e.group);
      this.instances.delete(id);
    }
  }

  /** Drop every instance (e.g. before loading a saved scene). */
  clearInstances(): void {
    for (const id of [...this.instances.keys()]) this.removeInstance(id);
  }

  /** Sync preview group transforms to the app instances; drop dead ones and
   *  rebuild any whose geometry changed (undo/redo, vertex edit, bake). */
  syncInstances(app: AppScene): void {
    const live = new Set(app.instances.map((i) => i.id));
    for (const id of [...this.instances.keys()]) if (!live.has(id)) this.removeInstance(id);
    for (const inst of app.instances) {
      let e = this.instances.get(inst.id);
      const currentBlas = app.blases[inst.blasIndex] ?? null;
      // Missing group (e.g. undo of a delete after its group was reaped) or a
      // swapped BLAS (regenerate / vertex edit / bake undo) → rebuild from the
      // current geometry. Unchanged geometry keeps its reference, so rigged
      // imports are left untouched here.
      if (!e || e.blas !== currentBlas) {
        if (e) this.removeInstance(inst.id);
        this.addInstance(inst.id, previewFromBlas(app, inst.blasIndex), [], currentBlas);
        e = this.instances.get(inst.id)!;
      }
      e.group.position.copy(inst.position);
      e.group.rotation.copy(inst.rotation);
      e.group.scale.setScalar(inst.scale);
    }
  }

  setActive(active: boolean): void {
    this.canvas.style.display = active ? "block" : "none";
    if (active) this.clock.getDelta(); // drop accumulated idle time
  }

  resize(w: number, h: number): void {
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / Math.max(1, h);
    this.camera.updateProjectionMatrix();
  }

  /** Mirror the ray tracer's procedural terrain + water so placement matches the
   *  final render. The terrain patch follows the camera target (snapped to a
   *  coarse grid) and is only re-displaced when its shape params or center
   *  change; water is a flat plane that just re-centers under the camera. */
  private syncWorld(cam: OrbitCamera): void {
    const w = this.app.world;

    if (w.terrainEnabled) {
      this.ground.visible = false;
      if (!this.terrainMesh) {
        const S = Preview.TERRAIN_SIZE, N = Preview.TERRAIN_SEG;
        const geo = new PlaneGeometry(S, S, N, N);
        geo.setAttribute("color", new Float32BufferAttribute(new Float32Array(geo.attributes.position.count * 3), 3));
        this.terrainMesh = new Mesh(
          geo,
          new MeshStandardMaterial({ roughness: 1, side: DoubleSide, vertexColors: true }),
        );
        this.terrainMesh.rotation.x = -Math.PI / 2;
        this.scene.add(this.terrainMesh);
        this.terrainSig = "";
        this.terrainColorSig = "";
      }
      // Center the patch on the camera target, snapped so it only shifts (and
      // re-displaces) in discrete steps rather than every frame.
      const snap = Preview.TERRAIN_SNAP;
      const cx = Math.round(cam.target.x / snap) * snap;
      const cz = Math.round(cam.target.z / snap) * snap;
      this.terrainMesh.position.set(cx, 0, cz);
      const geo = this.terrainMesh.geometry as PlaneGeometry;
      const pos = geo.attributes.position;

      // Heights + normals: rebuilt only when the shape or patch center changes
      // (sky/exposure/etc. also bump worldVersion but must not force a rebuild).
      const shapeSig = [
        cx, cz,
        w.terrainAmp, w.terrainFreq, w.terrainRidge, w.terrainOffset,
        w.terrainOctaves, w.terrainWarp, w.terrainWarpFreq, w.terrainSeed,
        w.terrainBasis, w.terrainFractal, w.terrainBasis2, w.terrainFractal2,
        w.terrainFreq2, w.terrainOctaves2, w.terrainWeight2,
        w.terrainTerraceSteps, w.terrainTerraceSharp,
      ].join(",");
      let reshaped = false;
      if (shapeSig !== this.terrainSig) {
        for (let i = 0; i < pos.count; i++) {
          // Mesh rotated -90° about X: local (x, y) → world (x, -y); local Z → world Y.
          // Add the patch center to sample the field at true world coordinates.
          pos.setZ(i, worldTerrainHeight(cx + pos.getX(i), cz - pos.getY(i), w));
        }
        pos.needsUpdate = true;
        geo.computeVertexNormals();
        this.terrainSig = shapeSig;
        reshaped = true;
      }

      // Vertex colors: re-run when the shape changed (normals moved) or when a
      // palette/threshold param changed. Cheaper than a re-displace on its own.
      const colorSig = [
        w.terrainAmp, w.terrainOffset, w.terrainSnowLine, w.terrainSlopeRock,
        ...w.terrainLow, ...w.terrainRock, ...w.terrainHigh,
      ].join(",");
      if (reshaped || colorSig !== this.terrainColorSig) {
        const nor = geo.attributes.normal;
        const col = geo.getAttribute("color") as Float32BufferAttribute;
        const rgb: [number, number, number] = [0, 0, 0];
        for (let i = 0; i < pos.count; i++) {
          // World-up normal component is the mesh's local +Z (rotation maps local Z → world Y).
          worldTerrainColor(cx + pos.getX(i), pos.getZ(i), cz - pos.getY(i), nor.getZ(i), w, rgb);
          col.setXYZ(i, rgb[0], rgb[1], rgb[2]);
        }
        col.needsUpdate = true;
        this.terrainColorSig = colorSig;
      }
      this.terrainMesh.visible = true;
    } else {
      this.ground.visible = true;
      if (this.terrainMesh) this.terrainMesh.visible = false;
    }

    if (w.waterEnabled) {
      if (!this.waterMesh) {
        this.waterMesh = new Mesh(
          new PlaneGeometry(8000, 8000),
          new MeshStandardMaterial({
            transparent: true, opacity: 0.6, roughness: 0.15, metalness: 0.2, side: DoubleSide,
          }),
        );
        this.waterMesh.rotation.x = -Math.PI / 2; // WATER_Y = 0
        this.scene.add(this.waterMesh);
      }
      this.waterMesh.position.set(cam.target.x, 0, cam.target.z); // keep it under the view
      (this.waterMesh.material as MeshStandardMaterial).color.setRGB(
        w.waterColor[0], w.waterColor[1], w.waterColor[2],
      );
      this.waterMesh.visible = true;
    } else if (this.waterMesh) {
      this.waterMesh.visible = false;
    }
  }

  /** Rasterize the scene's SDF primitives as tessellated meshes. Geometry is
   *  rebuilt only when a primitive's shape (type/size) changes; transform, color
   *  and solid/carve state refresh cheaply. Diffs against the live prim set so
   *  added/removed primitives appear/disappear on their own. */
  private syncPrims(): void {
    const live = new Set(this.app.prims.map((p) => p.id));
    for (const [id, e] of [...this.prims]) {
      if (!live.has(id)) {
        this.scene.remove(e.mesh);
        e.mesh.geometry.dispose();
        this.prims.delete(id);
      }
    }
    for (const prim of this.app.prims) {
      const sig = [prim.type, prim.a, prim.b, prim.c].join(",");
      let e = this.prims.get(prim.id);
      if (!e || e.sig !== sig) {
        const geom = primitiveLocalGeometry(prim);
        if (e) {
          e.mesh.geometry.dispose();
          e.mesh.geometry = geom;
          e.sig = sig;
        } else {
          const mesh = new Mesh(geom, new MeshStandardMaterial());
          this.scene.add(mesh);
          e = { mesh, sig };
          this.prims.set(prim.id, e);
        }
      }
      e.mesh.position.copy(prim.position);
      e.mesh.rotation.copy(prim.rotation);
      const mat = e.mesh.material as MeshStandardMaterial;
      mat.color.setRGB(prim.color[0], prim.color[1], prim.color[2]);
      mat.metalness = prim.reflectivity;
      mat.roughness = Math.max(0.04, 1 - prim.reflectivity);
      // Carve (subtractive) prims aren't solid in the final render — show them as
      // translucent ghosts so the boolean volume is visible but distinct.
      mat.transparent = prim.subtractive;
      mat.opacity = prim.subtractive ? 0.3 : 1;
    }
  }

  /** Build the additive round-points object the emitters draw into (lazy). */
  private ensureParticles(): Points {
    if (this.particles) return this.particles;
    const g = new BufferGeometry();
    const mk = (n: number) => new BufferAttribute(new Float32Array(MAX_PARTICLES * n), n).setUsage(DynamicDrawUsage);
    g.setAttribute("position", mk(3));
    g.setAttribute("psize", mk(1));
    g.setAttribute("pcolor", mk(3));
    // World-sized round sprites, additively blended, soft radial falloff. Colour
    // already carries opacity baked in, so additive blending reads like real fire.
    const mat = new ShaderMaterial({
      uniforms: { uScale: { value: 300 } },
      vertexShader: /* glsl */ `
        attribute float psize;
        attribute vec3 pcolor;
        varying vec3 vcol;
        uniform float uScale;
        void main() {
          vcol = pcolor;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_Position = projectionMatrix * mv;
          gl_PointSize = clamp(psize * uScale / max(-mv.z, 0.001), 1.0, 400.0);
        }`,
      fragmentShader: /* glsl */ `
        varying vec3 vcol;
        void main() {
          vec2 d = gl_PointCoord - 0.5;
          float r2 = dot(d, d);
          if (r2 > 0.25) { discard; }
          float a = 1.0 - r2 * 4.0;
          gl_FragColor = vec4(vcol * a * a, 1.0);
        }`,
      transparent: true,
      blending: AdditiveBlending,
      depthWrite: false,
    });
    this.particles = new Points(g, mat);
    this.particles.frustumCulled = false; // particles roam past the emitter's bounds
    this.scene.add(this.particles);
    return this.particles;
  }

  /** Evaluate every emitter at the looping preview clock and fill the points. */
  private syncParticles(dt: number): void {
    if (this.app.emitters.length === 0) {
      if (this.particles) this.particles.visible = false;
      return;
    }
    const pts = this.ensureParticles();
    pts.visible = true;

    // Loop the preview clock over a window so one-shot bursts replay while editing.
    this.particleTime += dt;
    let win = 3;
    for (const e of this.app.emitters) win = Math.max(win, e.loop ? e.lifetime : e.burstTime + e.lifetime);
    const t = this.particleTime % Math.min(win, 8);

    const buf = this.particleScratch;
    let count = 0;
    for (const e of this.app.emitters) {
      if (count >= MAX_PARTICLES) break;
      count += evaluateEmitter(e, t, buf, count * PARTICLE_FLOATS);
    }

    const g = pts.geometry;
    const pos = g.getAttribute("position") as BufferAttribute;
    const psize = g.getAttribute("psize") as BufferAttribute;
    const pcol = g.getAttribute("pcolor") as BufferAttribute;
    const posA = pos.array as Float32Array, sizeA = psize.array as Float32Array, colA = pcol.array as Float32Array;
    let n = 0;
    for (let i = 0; i < count; i++) {
      const o = i * PARTICLE_FLOATS;
      const op = buf[o + 7];
      if (op <= 0.003) continue; // dead/recycled slot
      posA[n * 3] = buf[o]; posA[n * 3 + 1] = buf[o + 1]; posA[n * 3 + 2] = buf[o + 2];
      sizeA[n] = buf[o + 3];
      colA[n * 3] = buf[o + 4] * op; colA[n * 3 + 1] = buf[o + 5] * op; colA[n * 3 + 2] = buf[o + 6] * op;
      n++;
    }
    g.setDrawRange(0, n);
    pos.needsUpdate = true; psize.needsUpdate = true; pcol.needsUpdate = true;

    // World radius → pixel size: viewport height / (2·tan(fov/2)) / depth (in vs).
    const h = Math.max(1, this.renderer.domElement.height);
    const fov = MathUtils.degToRad(this.camera.fov);
    (pts.material as ShaderMaterial).uniforms.uScale.value = h / (2 * Math.tan(fov / 2));
  }

  render(cam: OrbitCamera): void {
    // Terrain/water depend on the camera (patch follows the target), so refresh
    // every frame; the heavy re-displacement is guarded inside syncWorld.
    this.syncWorld(cam);
    this.syncPrims();

    const dt = this.clock.getDelta();
    for (const [id, e] of this.instances) {
      if (!this.posing.has(id)) e.mixer?.update(dt);
    }

    this.camera.position.copy(cam.position);
    this.camera.up.copy(cam.up);
    this.camera.lookAt(cam.target);
    this.camera.fov = MathUtils.radToDeg(cam.fovY);
    this.camera.updateProjectionMatrix();
    this.syncParticles(dt);

    // Sync the sun from the first directional light (illumination = -travel).
    const dir = this.app.lights.find((l) => l.type === LightType.Directional);
    if (dir) {
      this.sun.position.copy(dir.direction).multiplyScalar(100);
      this.sun.color.setRGB(dir.color[0], dir.color[1], dir.color[2]);
      this.sun.intensity = dir.intensity;
    } else {
      this.sun.intensity = 0; // no sun in the scene → genuinely dark
    }

    this.renderer.render(this.scene, this.camera);
  }
}
