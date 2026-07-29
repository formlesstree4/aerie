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
  AnimationUtils,
  AdditiveAnimationBlendMode,
  NormalAnimationBlendMode,
  type AnimationAction,
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
import type { BonePose } from "./scene/cutscene";
import { makeSpringBone, stepSpringBone, resetSpringBone, DEFAULT_SPRING, type SpringBoneState, type SpringParams } from "./mesh/spring";
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
    {
      group: Group; inner: Object3D; mixer: AnimationMixer | null; blas: BLAS | null;
      // Clip mixer: each imported clip has a lazily-created action, a blend weight,
      // and a normal/additive mode, so several can play at once and blend (Phase 8).
      // `clipIndex` is the "primary" clip for the scrub/snapshot workflow (Phase 5).
      animations: AnimationClip[];
      actions: (AnimationAction | null)[];
      weights: number[];
      additive: boolean[];
      additiveClips: (AnimationClip | null)[];
      clipIndex: number;
    }
  >();
  private posing = new Set<number>(); // instances whose animation is paused for manual posing
  // Cutscene-driven skeleton poses. Applied after the mixer each frame so keyed
  // joint animation wins over any imported clip; bones cached to skip traversal.
  private poseOverride = new Map<number, { bones: Object3D[]; pose: BonePose[] }>();
  // Spring-bone dynamics per instance (bone index → sim state) + per-instance params.
  // Stepped after the pose is set, so secondary motion layers on the base pose.
  private springs = new Map<number, Map<number, SpringBoneState>>();
  private springParamsById = new Map<number, SpringParams>();
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
    const actions: (AnimationAction | null)[] = animations.map(() => null);
    const weights = animations.map((_, i) => (i === 0 ? 1 : 0)); // clip 0 solo by default
    const additive = animations.map(() => false);
    const additiveClips: (AnimationClip | null)[] = animations.map(() => null);
    if (mixer && animations.length) { const a = mixer.clipAction(animations[0]); a.play(); a.setEffectiveWeight(1); actions[0] = a; }
    this.scene.add(group);
    // Remember the geometry this group was built from, so syncInstances can
    // detect when an undo/redo (or edit/bake) swapped the BLAS and rebuild.
    const inst = this.app.instances.find((i) => i.id === id);
    const resolvedBlas = blas !== undefined ? blas : inst ? this.app.blases[inst.blasIndex] ?? null : null;
    this.instances.set(id, { group, inner: group.children[0], mixer, blas: resolvedBlas, animations, actions, weights, additive, additiveClips, clipIndex: 0 });
  }

  // ---- clip mixer (Phase 5 scrub/snapshot + Phase 8 weighted blending) ----
  private clipEntry = (id: number) => this.instances.get(id);

  /** Additive version of a clip (cached) — its motion is applied relative to the
   *  first frame, so it layers on top of whatever else is playing. */
  private additiveClip(e: NonNullable<ReturnType<Preview["clipEntry"]>>, i: number): AnimationClip {
    let c = e.additiveClips[i];
    if (!c) { c = e.animations[i].clone(); AnimationUtils.makeClipAdditive(c); e.additiveClips[i] = c; }
    return c;
  }

  /** The action for clip `i`, created (playing, at its stored weight/mode) if needed. */
  private ensureAction(e: NonNullable<ReturnType<Preview["clipEntry"]>>, i: number): AnimationAction | null {
    if (!e.mixer) return null;
    let a = e.actions[i];
    if (!a) {
      a = e.mixer.clipAction(e.additive[i] ? this.additiveClip(e, i) : e.animations[i]);
      a.blendMode = e.additive[i] ? AdditiveAnimationBlendMode : NormalAnimationBlendMode;
      a.play();
      a.enabled = e.weights[i] > 0;
      a.setEffectiveWeight(e.weights[i]);
      e.actions[i] = a;
    }
    return a;
  }

  /** Append imported animation clips to a rig's clip list (creating the mixer if
   *  the model had none). Clips bind to bones by name, so they drive the rig when
   *  the animation's skeleton matches this model's. Returns how many were added. */
  addClips(id: number, clips: AnimationClip[]): number {
    const e = this.clipEntry(id);
    if (!e || clips.length === 0) return 0;
    if (!e.mixer) e.mixer = new AnimationMixer(e.group);
    for (const c of clips) {
      e.animations.push(c);
      e.actions.push(null);
      e.weights.push(0);
      e.additive.push(false);
      e.additiveClips.push(null);
    }
    return clips.length;
  }

  /** Names of an instance's imported animation clips. */
  clipNames(id: number): string[] {
    return this.clipEntry(id)?.animations.map((a, i) => a.name || `clip ${i + 1}`) ?? [];
  }

  activeClipIndex(id: number): number {
    return this.clipEntry(id)?.clipIndex ?? -1;
  }

  activeClipDuration(id: number): number {
    const e = this.clipEntry(id);
    return e?.animations[e.clipIndex]?.duration ?? 0;
  }

  /** Current scrub time of the primary clip (seconds). */
  clipTime(id: number): number {
    const e = this.clipEntry(id);
    return e?.actions[e.clipIndex]?.time ?? 0;
  }

  /** Blend weight of clip `i` (0 = off, 1 = full). */
  clipWeight(id: number, i: number): number {
    return this.clipEntry(id)?.weights[i] ?? 0;
  }

  /** Whether clip `i` layers additively over the others. */
  clipAdditive(id: number, i: number): boolean {
    return this.clipEntry(id)?.additive[i] ?? false;
  }

  /** Set clip `i`'s blend weight; the mixer normalises simultaneous normal-mode
   *  clips and layers additive ones. Playing clips animate when not posing. */
  setClipWeight(id: number, i: number, w: number): void {
    const e = this.clipEntry(id);
    if (!e || !e.mixer || i < 0 || i >= e.animations.length) return;
    e.weights[i] = Math.max(0, Math.min(1, w));
    const a = this.ensureAction(e, i);
    if (a) { a.enabled = e.weights[i] > 0; a.setEffectiveWeight(e.weights[i]); if (e.weights[i] > 0 && !a.isRunning()) a.play(); }
  }

  /** Toggle clip `i` between normal and additive blending (rebuilds its action). */
  setClipAdditive(id: number, i: number, on: boolean): void {
    const e = this.clipEntry(id);
    if (!e || !e.mixer || e.additive[i] === on) return;
    e.additive[i] = on;
    e.actions[i]?.stop();
    e.actions[i] = null;
    this.ensureAction(e, i); // recreated with the new blend mode + current weight
  }

  /** Crossfade the primary clip to clip `to` over `seconds`, updating weights. */
  crossFade(id: number, to: number, seconds: number): void {
    const e = this.clipEntry(id);
    if (!e || !e.mixer || to < 0 || to >= e.animations.length) return;
    const from = e.clipIndex;
    if (from === to) return;
    const fa = this.ensureAction(e, from);
    const ta = this.ensureAction(e, to);
    if (!fa || !ta) return;
    ta.enabled = true; ta.setEffectiveWeight(1); ta.play();
    fa.crossFadeTo(ta, Math.max(0.01, seconds), false);
    e.weights[from] = 0; e.weights[to] = 1; // end-state bookkeeping for the UI
    e.clipIndex = to;
  }

  /** Switch the primary clip and solo it (weight 1, others 0). While posing,
   *  sample its first frame so the rig shows it without the mixer advancing. */
  setActiveClip(id: number, index: number): void {
    const e = this.clipEntry(id);
    if (!e || !e.mixer || index < 0 || index >= e.animations.length) return;
    e.clipIndex = index;
    for (let j = 0; j < e.animations.length; j++) {
      e.weights[j] = j === index ? 1 : 0;
      const a = this.ensureAction(e, j);
      if (a) { a.enabled = e.weights[j] > 0; a.setEffectiveWeight(e.weights[j]); }
    }
    const pa = e.actions[index];
    if (pa) { pa.reset(); pa.play(); }
    if (this.posing.has(id) && pa) { pa.paused = true; pa.time = 0; e.mixer.update(0); }
  }

  /** Sample the primary clip at `time` onto the skeleton (scrub to a frame). */
  scrubClip(id: number, time: number): void {
    const e = this.clipEntry(id);
    if (!e || !e.mixer) return;
    const a = e.actions[e.clipIndex];
    if (!a) return;
    a.paused = true;
    a.time = Math.max(0, Math.min(time, e.animations[e.clipIndex]?.duration ?? 0));
    e.mixer.update(0); // apply the sampled pose now
  }

  /** Commit the current pose as the rig's working pose by stopping every clip, so
   *  the mixer no longer drives the bones. */
  snapshotClipPose(id: number): void {
    const e = this.clipEntry(id);
    if (!e) return;
    for (const a of e.actions) a?.stop();
    e.actions = e.animations.map(() => null);
    e.weights = e.animations.map(() => 0);
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

  /**
   * Total skin weight each bone carries, across every skinned mesh of the
   * instance (indexed like `getBones`).
   *
   * Dense character rigs carry hundreds of bones that deform nothing you can
   * see — twist and roll helpers, cloth and jiggle chains, correctives, IK
   * targets, attachment points. Dragging one moves the skeleton overlay and
   * leaves the model sitting there, which is indistinguishable from "posing is
   * broken". This is how the poser tells the two apart.
   */
  boneInfluence(id: number): Float32Array | null {
    const e = this.instances.get(id);
    const bones = this.getBones(id);
    if (!e || !bones) return null;
    const out = new Float32Array(bones.length);
    const index = new Map<Object3D, number>(bones.map((b, i) => [b, i]));
    e.inner.traverse((o) => {
      const sk = o as unknown as {
        isSkinnedMesh?: boolean; skeleton?: { bones: Object3D[] };
        geometry?: { attributes: Record<string, { count: number; getComponent(i: number, k: number): number } | undefined> };
      };
      if (!sk.isSkinnedMesh || !sk.skeleton || !sk.geometry) return;
      const si = sk.geometry.attributes.skinIndex;
      const sw = sk.geometry.attributes.skinWeight;
      if (!si || !sw) return;
      const meshBones = sk.skeleton.bones;
      for (let v = 0; v < si.count; v++) {
        for (let k = 0; k < 4; k++) {
          const w = sw.getComponent(v, k);
          if (w <= 0) continue;
          const gi = index.get(meshBones[si.getComponent(v, k)]);
          if (gi !== undefined) out[gi] += w;
        }
      }
    });
    return out;
  }

  /** What the rig looks like to the poser — skinned meshes, whether they share a
   *  skeleton, and how many of its bones actually deform geometry. Logged when
   *  posing starts, so "I dragged a joint and nothing moved" is answerable. */
  rigReport(id: number): string {
    const e = this.instances.get(id);
    if (!e) return "no preview instance";
    const meshes: { name: string; bones: number; skel: object; verts: number; visible: boolean }[] = [];
    e.inner.traverse((o) => {
      const sk = o as unknown as {
        isSkinnedMesh?: boolean; skeleton?: { bones: Object3D[] }; name: string; visible: boolean;
        geometry?: { attributes?: { position?: { count: number } } };
      };
      if (sk.isSkinnedMesh && sk.skeleton) {
        meshes.push({
          name: sk.name || "(unnamed)", bones: sk.skeleton.bones.length, skel: sk.skeleton,
          verts: sk.geometry?.attributes?.position?.count ?? 0, visible: sk.visible,
        });
      }
    });
    const skeletons = new Set(meshes.map((m) => m.skel));
    const posed = this.getBones(id);
    const drivenBy = (m: (typeof meshes)[number]) =>
      posed && (m.skel as { bones: Object3D[] }).bones.some((b) => posed.includes(b));
    const inf = this.boneInfluence(id);
    const deforming = inf ? inf.reduce((n, w) => n + (w >= 1 ? 1 : 0), 0) : -1;
    return [
      `skinned meshes: ${meshes.length} · distinct skeletons: ${skeletons.size}`,
      `posed skeleton: ${posed ? `${posed.length} bones` : "none"}` +
        (deforming >= 0 ? ` · ${deforming} carry skin weight, ${(posed?.length ?? 0) - deforming} are helpers` : ""),
      ...meshes.map((m, i) =>
        `  [${i}] ${m.name} — ${m.bones} bones, ${m.verts} verts` +
        `${m.visible ? "" : ", HIDDEN"}${drivenBy(m) ? "  ← follows the posed bones" : "  ← NOT driven by the posed bones"}`),
    ].join("\n");
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

  /** Drive an instance's skeleton from a cutscene keyframe pose (null clears it).
   *  Applied every frame after the mixer, so it overrides clip playback. */
  setPoseOverride(id: number, pose: BonePose[] | null): void {
    if (!pose) { this.poseOverride.delete(id); return; }
    const bones = this.getBones(id);
    if (bones) this.poseOverride.set(id, { bones, pose });
  }

  clearPoseOverrides(): void {
    this.poseOverride.clear();
  }

  // ---- spring bones (Phase 9: secondary motion) ----
  /** Mark/unmark bone `i` as a spring. Springs need a child bone to aim at, so
   *  leaf joints are ignored. Disabling restores the bone to its rest pose. */
  setSpringBone(id: number, i: number, on: boolean): void {
    let m = this.springs.get(id);
    if (!on) {
      const s = m?.get(i);
      if (s) { resetSpringBone(s); m!.delete(i); }
      return;
    }
    const bones = this.getBones(id);
    const bone = bones?.[i];
    if (!bones || !bone) return;
    const child = bones.find((x) => x.parent === bone); // first child bone → the aim target
    if (!child) return; // leaf: nothing to swing
    if (!m) { m = new Map(); this.springs.set(id, m); }
    if (!m.has(i)) m.set(i, makeSpringBone(bone, child.position.clone()));
  }

  isSpringBone(id: number, i: number): boolean {
    return this.springs.get(id)?.has(i) ?? false;
  }

  hasSprings(id: number): boolean {
    return (this.springs.get(id)?.size ?? 0) > 0;
  }

  springParams(id: number): SpringParams {
    return this.springParamsById.get(id) ?? { ...DEFAULT_SPRING };
  }

  setSpringParam(id: number, key: keyof SpringParams, v: number): void {
    const p = { ...this.springParams(id), [key]: v };
    this.springParamsById.set(id, p);
  }

  clearSprings(id: number): void {
    const m = this.springs.get(id);
    if (m) for (const s of m.values()) resetSpringBone(s);
    this.springs.delete(id);
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

  /** Like `prepareBake`, but first stamps a cutscene keyframe pose onto the
   *  skeleton — used to re-bake each frame of an offline render so the ray tracer
   *  sees the posed joints (it draws from the BLAS, not the live skeleton). */
  prepareBakePosed(id: number, pose: BonePose[]): Object3D | null {
    const bones = this.getBones(id);
    if (bones) {
      for (const bp of pose) {
        const b = bones[bp.i];
        if (b) b.quaternion.set(bp.q[0], bp.q[1], bp.q[2], bp.q[3]);
      }
    }
    return this.prepareBake(id); // zeroes the group + refreshes world matrices
  }

  /** Offline-render prep with spring dynamics. Stamps the keyframe pose, places
   *  the instance at its world transform so the springs feel it moving through the
   *  shot, steps the sim by `dt`, then hands back the local-space root to bake.
   *  `place` is the instance's world transform (quaternion as xyzw). */
  prepareBakePosedDynamic(
    id: number,
    pose: BonePose[] | null,
    place: { px: number; py: number; pz: number; qx: number; qy: number; qz: number; qw: number; scale: number },
    dt: number,
  ): Object3D | null {
    const e = this.instances.get(id);
    if (!e) return null;
    const bones = this.getBones(id);
    if (bones && pose) {
      for (const bp of pose) { const b = bones[bp.i]; if (b) b.quaternion.set(bp.q[0], bp.q[1], bp.q[2], bp.q[3]); }
    }
    const m = this.springs.get(id);
    if (m && m.size) {
      // Place in the world, step the springs (they read world-space head motion),
      // then fall through to the local-space bake below.
      e.group.position.set(place.px, place.py, place.pz);
      e.group.quaternion.set(place.qx, place.qy, place.qz, place.qw);
      e.group.scale.setScalar(place.scale);
      e.group.updateMatrixWorld(true);
      const p = this.springParams(id);
      for (const s of [...m.values()].sort((a, b) => a.depth - b.depth)) stepSpringBone(s, p, dt);
    }
    return this.prepareBake(id); // zero the group + refresh for local-space bake
  }

  /** Re-arm every spring to rest — call before an offline render so its secondary
   *  motion starts settled and accumulates over the shot. */
  resetAllSprings(): void {
    for (const m of this.springs.values()) for (const s of m.values()) resetSpringBone(s);
  }

  removeInstance(id: number): void {
    const e = this.instances.get(id);
    if (e) {
      this.scene.remove(e.group);
      this.instances.delete(id);
    }
    this.springs.delete(id);
    this.springParamsById.delete(id);
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

  /**
   * Advance every rig by one frame: clip mixers, then cutscene pose overrides,
   * then spring dynamics. Returns the delta it consumed.
   *
   * Split out of `render` because the ray-traced view needs the skeletons to move
   * without drawing the raster canvas — it bakes the stepped pose into geometry
   * instead. Call it exactly once per frame from whichever view is live, or the
   * clock gets consumed twice and animation runs at double speed.
   */
  stepAnimation(): number {
    const dt = this.clock.getDelta();
    for (const [id, e] of this.instances) {
      if (!this.posing.has(id)) e.mixer?.update(dt);
    }
    // Cutscene poses override the mixer: write keyed local rotations last.
    for (const { bones, pose } of this.poseOverride.values()) {
      for (const bp of pose) {
        const b = bones[bp.i];
        if (b) b.quaternion.set(bp.q[0], bp.q[1], bp.q[2], bp.q[3]);
      }
    }
    // Spring dynamics layer on the base pose: step shallower bones first so a
    // chain's parents move before their children each frame.
    for (const [id, m] of this.springs) {
      if (m.size === 0) continue;
      const p = this.springParams(id);
      const list = [...m.values()].sort((a, b) => a.depth - b.depth);
      for (const s of list) stepSpringBone(s, p, dt);
    }
    return dt;
  }

  /**
   * Point an instance at the BLAS it now corresponds to, WITHOUT rebuilding it.
   *
   * `syncInstances` rebuilds any instance whose BLAS changed under it, which is
   * right for an edit or a bake but fatal for the render view's per-frame pose
   * bake: it would tear down the rigged group and put static geometry in its
   * place, leaving a skeleton that moves nothing. Re-baking a pose doesn't change
   * what the preview should be showing, so we just re-point the record.
   */
  setInstanceBlas(id: number, blas: BLAS | null): void {
    const e = this.instances.get(id);
    if (e) e.blas = blas;
  }

  /** Is a clip actually driving this rig — a mixer with some weight above zero? */
  hasPlayingClip(id: number): boolean {
    if (this.posing.has(id)) return false; // manual posing owns the skeleton
    const e = this.instances.get(id);
    return !!e?.mixer && e.weights.some((w) => w > 0);
  }

  /** Is anything actually moving this rig — a clip playing above zero weight, a
   *  cutscene pose driving it, or spring bones still settling? */
  isAnimating(id: number): boolean {
    if (this.posing.has(id)) return false; // manual posing owns the skeleton
    if (this.poseOverride.has(id)) return true;
    if (this.hasSprings(id)) return true;
    return this.hasPlayingClip(id);
  }

  /**
   * Put every playing clip at an ABSOLUTE time, for offline rendering.
   *
   * The live views advance clips by whatever the wall clock delivered since the
   * last frame, which is exactly wrong for an export: frames are rendered as fast
   * (or as slowly) as the machine manages, so real elapsed time bears no relation
   * to the frame being written. Sampling each frame at its own timeline position
   * instead makes the exported motion deterministic and correctly paced — and it
   * is what makes a clip play across a WebM rather than sitting on frame one.
   *
   * `setTime` rewinds to zero and re-runs the mixer, so looping clips wrap the way
   * they would have on playback.
   */
  sampleAnimationAt(timeSec: number): void {
    for (const [id, e] of this.instances) {
      if (this.posing.has(id) || !e.mixer) continue;
      e.mixer.setTime(Math.max(0, timeSec));
    }
  }

  render(cam: OrbitCamera): void {
    // Terrain/water depend on the camera (patch follows the target), so refresh
    // every frame; the heavy re-displacement is guarded inside syncWorld.
    this.syncWorld(cam);
    this.syncPrims();

    const dt = this.stepAnimation();

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
