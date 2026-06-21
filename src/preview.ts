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
} from "three";
import type { OrbitCamera } from "./scene/camera";
import { Scene as AppScene, LightType, type BLAS } from "./scene/scene";
import { previewFromBlas } from "./scene/serialize";

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

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly app: AppScene,
  ) {
    this.renderer = new WebGLRenderer({ canvas, antialias: true });
    this.scene.background = new Color(0xc7d6eb);
    this.scene.fog = new Fog(0xc7d6eb, 120, 900);

    const ground = new Mesh(
      new PlaneGeometry(4000, 4000),
      new MeshStandardMaterial({ color: 0x5b6b45, roughness: 1 }),
    );
    ground.rotation.x = -Math.PI / 2;
    this.scene.add(ground);

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

  render(cam: OrbitCamera): void {
    const dt = this.clock.getDelta();
    for (const [id, e] of this.instances) {
      if (!this.posing.has(id)) e.mixer?.update(dt);
    }

    this.camera.position.copy(cam.position);
    this.camera.up.copy(cam.up);
    this.camera.lookAt(cam.target);
    this.camera.fov = MathUtils.radToDeg(cam.fovY);
    this.camera.updateProjectionMatrix();

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
