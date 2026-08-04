import { Renderer } from "./webgpu/renderer";
import { OrbitCamera } from "./scene/camera";
import { defaultScene, Primitive, MeshInstance, Light, Emitter, LightType, PrimType, PrimPattern, PRIM_LABELS, type Selectable, type SceneState, type BLAS } from "./scene/scene";
import { SCENE_PRESETS, type ScenePreset, type CameraHint } from "./scene/presets";
import { buildUI, type ScatterOptions, type TurntableOptions } from "./ui";
import { Preview } from "./preview";
import { importModel, importAnimations, bakePose, worldMesh } from "./mesh/modelImport";
import { solveTwoBoneIK, solveCCD, autoPole } from "./mesh/ik";
import { ikChain as poseIkChain, boneSubtree } from "./mesh/poseChain";
import "./mesh/loaders"; // registers glTF / FBX / STL / PLY loaders
import { convertPrimitive, primitiveGeometry } from "./mesh/tessellate";
import { landscapeModel, defaultLandform, LANDSCAPE_HALF, type LandscapeType } from "./mesh/landscape";
import { evaluateHeight } from "./gen/evaluate";
import { csgBuildModel, trisToGeometry, type BoolInput, type MatDesc } from "./mesh/boolean";
import { serializeScene, deserializeScene, previewFromBlas, b64ToU8 } from "./scene/serialize";
import { buildEditMesh, rebuildBlas, type EditMesh } from "./mesh/editmesh";
import { History } from "./scene/history";
import { evalCutscene, cutsceneDuration, keyTime, poseVariesAcrossKeys, type CamKey, type CamState, type Ease, type ObjXform, type BonePose } from "./scene/cutscene";
import { evaluateEmitter, MAX_PARTICLES, PARTICLE_FLOATS } from "./gen/particles";
import { Muxer, ArrayBufferTarget } from "webm-muxer";
import { Vector3, Euler, Matrix3, Matrix4, Quaternion, type Object3D } from "three";

const canvas = document.getElementById("view") as HTMLCanvasElement;
const previewCanvas = document.getElementById("preview") as HTMLCanvasElement;
const overlay = document.getElementById("overlay") as HTMLCanvasElement;
const octx = overlay.getContext("2d")!;
const hud = document.getElementById("hud") as HTMLDivElement;
const errBox = document.getElementById("err") as HTMLDivElement;

function fail(msg: string): never {
  errBox.style.display = "grid";
  errBox.textContent = msg;
  throw new Error(msg);
}

async function main() {
  if (!navigator.gpu) {
    // navigator.gpu is only exposed in a secure context: HTTPS, or plain HTTP
    // on localhost/127.0.0.1. Serving the container over http://<lan-ip>:8080
    // hides the API no matter how well the browser supports WebGPU, so call
    // that case out instead of blaming the browser.
    if (!window.isSecureContext) {
      fail(
        `WebGPU is hidden because this page is not a secure context (${location.origin}).\n\n` +
          "Browsers only expose navigator.gpu over HTTPS, or over plain HTTP on " +
          "localhost / 127.0.0.1.\n\n" +
          "Fixes: open the app at http://localhost:8080, tunnel the port to your " +
          "machine, or serve it over HTTPS.",
      );
    }
    fail(
      "WebGPU is not available in this browser.\n\n" +
        "Use a recent Chrome/Edge (or Firefox Nightly with WebGPU enabled), " +
        "and make sure hardware acceleration is on.",
    );
  }

  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) fail("No suitable GPU adapter found.");

  // The compute tracer binds 11 storage buffers (8 scene pools + TLAS nodes +
  // TLAS order + the particle field); the spec only guarantees 8, so raise the
  // limit to what the adapter actually supports.
  const NEEDED_STORAGE_BUFFERS = 11;
  const maxStorage = adapter.limits.maxStorageBuffersPerShaderStage;
  if (maxStorage < NEEDED_STORAGE_BUFFERS) {
    fail(
      `This GPU allows only ${maxStorage} storage buffers per shader stage; ` +
        `the renderer needs ${NEEDED_STORAGE_BUFFERS}.`,
    );
  }
  const device = await adapter.requestDevice({
    // Request the adapter's actual max (≥ what we need) so headroom is available
    // and adding a buffer later doesn't silently break pipeline creation again.
    requiredLimits: { maxStorageBuffersPerShaderStage: maxStorage },
  });
  device.lost.then((info) => fail(`GPU device lost: ${info.message}`));

  const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "opaque" });

  const renderer = new Renderer(device, context, format);
  const cam = new OrbitCamera();
  const scene = defaultScene();
  const preview = new Preview(previewCanvas, scene);
  // scene.version at the last save / load / new — used to detect unsaved changes.
  let savedVersion = scene.version;

  let mode: "render" | "preview" = "render";
  const applyMode = () => preview.setActive(mode === "preview");
  let showCarverGhosts = false; // draw negative (carve) prims as wireframe cages
  let focusPick = false; // armed by the Render lab: next viewport click sets DoF focus

  // Active viewport tool: "move" = click-select + drag-move (default); "select"
  // = left-drag box-selects objects (or vertices, in edit mode). Toggled by the
  // 1/2 hotkeys and the Tools menu. Alt+left-drag always orbits.
  let tool: "move" | "select" = "move";
  // Gizmo transform mode (translate/rotate/scale), toggled by W/E/R + Tools menu.
  // Declared here (before buildUI) because the menu bar reads it during setup.
  let gizmoMode: "translate" | "rotate" | "scale" = "translate";
  // Grid snapping for gizmo transforms (translate → grid, rotate → 15°, scale → 0.25).
  let snapEnabled = false;
  let snapGrid = 1;

  // Camera bookmarks (4 slots): saved orbit poses to jump between.
  interface CamPose { target: [number, number, number]; yaw: number; pitch: number; distance: number; }
  const camBookmarks: (CamPose | null)[] = [null, null, null, null];

  // Undo/redo history (assigned once the UI exists, since it owns selection).
  let history: History;

  // Vertex-editing state (declared early: read by UI hooks during setup).
  let editInstance: MeshInstance | null = null;
  let editMesh: EditMesh | null = null;

  // Skeleton-posing state.
  let poseInst: MeshInstance | null = null;
  let poseBones: Object3D[] = [];
  let selectedBone = -1;
  let poseRoot = -1; // topmost bone (whole-armature translation)
  const poseOrig: [number, number, number][] = []; // original bone euler angles (for reset)
  const poseRootOrig = new Vector3(); // original root position (for reset)
  // Scratch quaternions for drag-to-rotate.
  const qParent = new Quaternion();
  const qWorld = new Quaternion();
  const qDeltaA = new Quaternion();
  const qDeltaB = new Quaternion();
  // IK posing: when enabled, dragging a joint solves a chain of ancestors so the
  // joint follows the cursor and its whole subtree rides along. `ikChain` picks
  // the chain from the skeleton's shape (see below) — a 3-bone limb takes the
  // exact two-bone solver, anything else CCD. Joints with no usable chain fall
  // back to FK.
  let ikEnabled = true;
  let hoverBone = -1;         // joint under the cursor in pose mode (drag preview)
  let hoverExtend = false;    // Shift held → the preview shows the extended chain
  // Multi-joint selection. `selectedBone` stays the active joint (the one the FK
  // rings and the panel act on); `boneSel` is the wider set a drag moves together.
  // Ctrl-click toggles membership, the Box-Select tool rubber-bands over joints.
  const boneSel = new Set<number>();
  // One entry per selected joint during a group drag: where it started and the
  // chain that carries it there.
  type MultiGrab = { bone: Object3D; start: Vector3; chain: Object3D[]; pole: Vector3 | null; depth: number };
  let multiGrab: MultiGrab[] = [];
  const multiHit0 = new Vector3(); // cursor position on the drag plane at grab time
  let ikRoot: Object3D | null = null;
  let ikMid: Object3D | null = null;
  let ikTip: Object3D | null = null;
  let ikChainArr: Object3D[] | null = null; // full chain (root→tip) for the CCD path
  const ikPole = new Vector3();       // world-space bend hint, captured at drag start
  const ikPlanePoint = new Vector3(); // camera-facing drag plane through the effector
  // FK rotation-ring gizmo: drag a ring to rotate the selected bone about one of
  // its local axes, applied as a quaternion (gimbal-free — no Euler). Shown in FK
  // mode only; IK mode uses limb dragging instead.
  let gzAxis = -1;                       // 0/1/2 local axis being dragged, or -1
  const gzStartQuat = new Quaternion();  // bone local rotation at drag start
  const gzAxisWorld = new Vector3();     // dragged axis in world space (drag-plane normal)
  const gzRefU = new Vector3();          // in-plane reference direction (angle origin)
  const gzCenter = new Vector3();        // joint world position (drag-plane point)
  let gzPrevAngle = 0;                   // last in-plane angle (for unwrapped accumulation)
  let gzAccum = 0;                       // accumulated signed rotation this drag
  const gzUnit = [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)];
  const gzN = new Vector3(), gzTmpU = new Vector3(), gzTmpV = new Vector3();
  const gzHit = new Vector3(), gzQ = new Quaternion(), gzBoneWQ = new Quaternion();
  // Other selected joints turning with the gizmo, and their pre-drag rotations.
  let gzGroup: { bone: Object3D; start: Quaternion }[] = [];

  // Pose ergonomics: a pose-local undo/redo stack (the scene History snapshots
  // SceneState, not the live skeleton), a copy/paste clipboard, and a session
  // pose library. Poses are per-bone local quaternions, indexed by bone.
  type PoseSnap = { i: number; q: [number, number, number, number] }[];
  let poseUndo: PoseSnap[] = [];
  let poseRedo: PoseSnap[] = [];
  let poseDragSnap: PoseSnap | null = null; // pre-drag pose, pushed on release if it changed
  let poseClipboard: PoseSnap | null = null;
  const poseLibrary: { name: string; pose: PoseSnap }[] = [];
  const POSE_UNDO_LIMIT = 60;

  function capturePose(): PoseSnap {
    return poseBones.map((b, i) => ({
      i, q: [b.quaternion.x, b.quaternion.y, b.quaternion.z, b.quaternion.w] as [number, number, number, number],
    }));
  }
  function applyPose(p: PoseSnap): void {
    for (const bp of p) {
      const b = poseBones[bp.i];
      if (b) b.quaternion.set(bp.q[0], bp.q[1], bp.q[2], bp.q[3]);
    }
  }
  function posesDiffer(a: PoseSnap, b: PoseSnap): boolean {
    if (a.length !== b.length) return true;
    for (let k = 0; k < a.length; k++) {
      const p = a[k].q, q = b[k].q;
      if (Math.abs(p[0] - q[0]) + Math.abs(p[1] - q[1]) + Math.abs(p[2] - q[2]) + Math.abs(p[3] - q[3]) > 1e-6) return true;
    }
    return false;
  }
  /** Record `snap` (a pre-edit pose) as the state an undo returns to. */
  function pushPoseUndo(snap: PoseSnap): void {
    poseUndo.push(snap);
    if (poseUndo.length > POSE_UNDO_LIMIT) poseUndo.shift();
    poseRedo.length = 0;
  }
  function poseUndoStep(): void {
    if (!poseInst || poseUndo.length === 0) return;
    poseRedo.push(capturePose());
    applyPose(poseUndo.pop()!);
    ui.refresh();
  }
  function poseRedoStep(): void {
    if (!poseInst || poseRedo.length === 0) return;
    poseUndo.push(capturePose());
    applyPose(poseRedo.pop()!);
    ui.refresh();
  }

  // Discrete pose ops (each is one undo step). Paste/library only apply to a rig
  // with the same bone count, so an index-based pose can't scramble another rig.
  function copyPose(): void { if (poseInst) poseClipboard = capturePose(); }
  function pastePose(): void {
    if (!poseInst || !poseClipboard || poseClipboard.length !== poseBones.length) return;
    pushPoseUndo(capturePose());
    applyPose(poseClipboard);
    ui.refresh();
  }
  function resetAllJoints(): void {
    if (!poseInst) return;
    pushPoseUndo(capturePose());
    for (let i = 0; i < poseBones.length; i++) {
      const o = poseOrig[i];
      if (o) poseBones[i].rotation.set(o[0], o[1], o[2]);
    }
    ui.refresh();
  }
  function savePose(): void {
    if (poseInst) poseLibrary.push({ name: `Pose ${poseLibrary.length + 1}`, pose: capturePose() });
  }
  function applyLibraryPose(idx: number): void {
    const entry = poseLibrary[idx];
    if (!poseInst || !entry || entry.pose.length !== poseBones.length) return;
    pushPoseUndo(capturePose());
    applyPose(entry.pose);
    ui.refresh();
  }

  // ---- which joints are worth showing ----
  // A dense character rig carries hundreds of bones that deform nothing visible:
  // twist and roll helpers, cloth and jiggle chains, correctives, attachment
  // points. Drawn as a dot each, they bury the joints you actually pose and make
  // it easy to grab one that moves the skeleton and not the model. So the
  // overlay only offers bones that carry real skin weight — plus whatever is
  // needed to keep those connected to the root.
  let poseDeforms: boolean[] = [];   // per bone: does moving it deform the mesh?
  let poseShowAll = false;           // override, for rigs where the filter guesses wrong
  const DEFORM_MIN = 1.0;            // total skin weight: below this it's a helper

  /** A bone is offered if it deforms geometry, or if it's on the path from the
   *  root to one that does (so chains and overlay lines stay connected). */
  function buildDeformSet(inst: MeshInstance): void {
    const inf = preview.boneInfluence(inst.id);
    poseDeforms = new Array(poseBones.length).fill(true);
    if (!inf) return;
    const index = new Map<Object3D, number>(poseBones.map((b, i) => [b, i]));
    const keep = poseBones.map((_, i) => inf[i] >= DEFORM_MIN);
    for (let i = 0; i < poseBones.length; i++) {
      if (!keep[i]) continue;
      for (let p = poseBones[i].parent; p; p = p.parent) { // link it back to the root
        const pi = index.get(p);
        if (pi === undefined || keep[pi]) break;
        keep[pi] = true;
      }
    }
    if (keep.some(Boolean)) poseDeforms = keep; // never filter everything away
  }

  /** Bones the overlay draws and the cursor can grab. */
  const boneOffered = (i: number) => poseShowAll || poseDeforms[i] !== false;

  function togglePose(inst: MeshInstance): void {
    if (poseInst === inst) {
      preview.setPosing(inst.id, false);
      poseInst = null;
      poseBones = [];
      selectedBone = -1;
      boneSel.clear();
      poseRoot = -1;
      hoverBone = -1;
      canvas.style.cursor = "";
      poseUndo = []; poseRedo = []; poseDragSnap = null;
      return;
    }
    const bones = preview.getBones(inst.id);
    if (!bones || bones.length === 0) return; // not rigged
    preview.clearPoseOverrides(); // manual posing takes the skeleton off the timeline
    poseUndo = []; poseRedo = []; poseDragSnap = null;
    poseInst = inst;
    poseBones = bones;
    selectedBone = 0;
    boneSel.clear();
    poseOrig.length = 0;
    for (const b of bones) poseOrig.push([b.rotation.x, b.rotation.y, b.rotation.z]);
    // Root = first bone whose parent isn't itself a bone.
    const boneSet = new Set(bones);
    poseRoot = bones.findIndex((b) => !b.parent || !boneSet.has(b.parent));
    if (poseRoot >= 0) poseRootOrig.copy(bones[poseRoot].position);
    buildDeformSet(inst);
    console.log(`[pose] ${preview.rigReport(inst.id)}`);
    preview.setPosing(inst.id, true);
    mode = "preview"; // the raster view is the responsive place to pose (the ray
                      // view keeps up too, but re-bakes the rig on every drag frame)
    applyMode();
  }

  let importing = false;
  async function doImport(file: File) {
    if (importing) return;
    importing = true;
    hud.textContent = `Importing ${file.name}…`;
    try {
      const res = await importModel(file);
      const inst = scene.addModel(res.model, cam.target.clone());
      // Keep the original bytes so save/load can re-import (rig + textured preview).
      scene.blasFile[inst.blasIndex] = { name: file.name, bytes: new Uint8Array(await file.arrayBuffer()) };
      preview.addInstance(inst.id, res.group, res.animations);
      ui.select(inst);
      mode = "preview";
      applyMode();
      history.commit();
    } catch (e) {
      hud.textContent = `Import failed: ${e instanceof Error ? e.message : e}`;
    } finally {
      importing = false;
    }
  }

  // ---- animation in the ray-traced view ----
  // The raster preview shows skinning live, because three deforms the mesh on the
  // GPU every frame. The ray tracer traces a BVH built from static triangles, so a
  // moving rig has to be re-baked into that geometry frame by frame. That's real
  // work (skin + rebuild the BLAS) and every change restarts the sample
  // accumulation, so it's deliberately limited to rigs that are actually moving:
  // a still scene re-converges to a clean image on its own.
  let animateInRender = true;
  const lastBakedPose = new Map<number, Float32Array>(); // per instance: bone quats at last bake

  /** Current bone rotations, flat, for the cheap "has this rig moved?" test. */
  function poseSignature(id: number): Float32Array | null {
    const bones = preview.getBones(id);
    if (!bones) return null;
    const out = new Float32Array(bones.length * 4);
    for (let i = 0; i < bones.length; i++) {
      const q = bones[i].quaternion;
      out[i * 4] = q.x; out[i * 4 + 1] = q.y; out[i * 4 + 2] = q.z; out[i * 4 + 3] = q.w;
    }
    return out;
  }
  function poseChanged(a: Float32Array, b: Float32Array | undefined): boolean {
    if (!b || a.length !== b.length) return true;
    for (let i = 0; i < a.length; i++) if (Math.abs(a[i] - b[i]) > 1e-5) return true;
    return false;
  }

  /**
   * Step every animating rig and bake the ones that moved into their BLAS.
   * Returns true if geometry changed (so the caller re-uploads and restarts the
   * accumulation). No-ops when nothing is playing, which is what lets a paused
   * scene still accumulate its full sample count.
   */
  function stepRenderAnimation(): boolean {
    preview.syncInstances(scene);
    // Always step — it also drains the shared clock, so returning to the raster
    // view doesn't jump forward by however long the ray trace was on screen.
    preview.stepAnimation(); // mixers + cutscene overrides + springs, without drawing
    if (!animateInRender) return false;
    // Manual posing counts as motion too: dragging a joint re-bakes each frame,
    // and the moment you let go the dirty check goes quiet and the image refines.
    const live = scene.instances.filter((inst) => preview.isAnimating(inst.id) || preview.isPosing(inst.id));

    let baked = false;
    for (const inst of live) {
      const sig = poseSignature(inst.id);
      if (!sig || !poseChanged(sig, lastBakedPose.get(inst.id))) continue;
      const inner = preview.prepareBake(inst.id);
      if (!inner) continue;
      // replaceBlas clears blasFile — the link save/load re-imports the rig from.
      // A transient render-view bake must not cost the scene its rig, so put it back.
      const file = scene.blasFile[inst.blasIndex];
      scene.replaceBlas(inst.blasIndex, bakePose(inner, scene.blasMatBase[inst.blasIndex]));
      scene.blasFile[inst.blasIndex] = file;
      // Keep the preview pointed at the new BLAS, or syncInstances would tear the
      // rigged group down and rebuild it as static geometry — a skeleton driving
      // nothing. The rig itself hasn't changed; only its baked snapshot has.
      preview.setInstanceBlas(inst.id, scene.blases[inst.blasIndex]);
      lastBakedPose.set(inst.id, sig);
      baked = true;
    }
    if (baked) {
      preview.syncInstances(scene); // restore the group transforms prepareBake zeroed
      renderer.uploadMeshPools(scene);
      renderer.resetAccumulation(); // the geometry moved — start the image again
    }
    return baked;
  }

  // ---- offline (WebM) rig baking ----
  // Same idea as the live render view, but driven by the frame's timeline position
  // rather than the wall clock, and reversible: an export must never leave the
  // scene holding baked geometry, so every BLAS it swaps is recorded for restore.
  type BlasSnap = { blas: BLAS; file: (typeof scene.blasFile)[number] };
  const bakeQuat = new Quaternion();

  /** Bake one rigged instance's current pose (clip- and/or keyframe-driven, plus
   *  spring dynamics stepped by `shutter`) into its BLAS for an offline frame. */
  function bakeRigFrame(
    obj: MeshInstance,
    pose: BonePose[] | null,
    shutter: number,
    restore: Map<number, BlasSnap>,
  ): boolean {
    if (!restore.has(obj.blasIndex)) {
      restore.set(obj.blasIndex, { blas: scene.blases[obj.blasIndex], file: scene.blasFile[obj.blasIndex] });
    }
    bakeQuat.setFromEuler(obj.rotation);
    const place = {
      px: obj.position.x, py: obj.position.y, pz: obj.position.z,
      qx: bakeQuat.x, qy: bakeQuat.y, qz: bakeQuat.z, qw: bakeQuat.w, scale: obj.scale,
    };
    const inner = preview.prepareBakePosedDynamic(obj.id, pose, place, shutter);
    if (!inner) return false;
    scene.replaceBlas(obj.blasIndex, bakePose(inner, scene.blasMatBase[obj.blasIndex]));
    preview.setInstanceBlas(obj.id, scene.blases[obj.blasIndex]); // don't let syncInstances rebuild the rig
    return true;
  }

  /** Put every BLAS an export swapped back, so the scene is exactly as it was. */
  function restoreBakedBlas(restore: Map<number, BlasSnap>): void {
    if (!restore.size) return;
    for (const [bi, snap] of restore) {
      scene.replaceBlas(bi, snap.blas);
      scene.blasFile[bi] = snap.file; // restore the re-import link replaceBlas cleared
    }
    for (const inst of scene.instances) preview.setInstanceBlas(inst.id, scene.blases[inst.blasIndex] ?? null);
    preview.syncInstances(scene); // restore preview group transforms zeroed by the bake
    restore.clear();
    lastBakedPose.clear(); // rest geometry is back — the live view must re-bake
  }

  function bakeSelected() {
    const sel = ui.getSelected();
    if (!(sel instanceof MeshInstance)) return;
    const inner = preview.prepareBake(sel.id);
    if (!inner) return;
    scene.replaceBlas(sel.blasIndex, bakePose(inner, scene.blasMatBase[sel.blasIndex]));
    preview.syncInstances(scene); // restore the preview group's transform
    if (mode === "preview") {
      mode = "render"; // show the baked result
      applyMode();
    }
    history.commit();
  }

  async function fileToLayer(file: File, size: number): Promise<Uint8Array<ArrayBuffer>> {
    const bmp = await createImageBitmap(file);
    const cnv = document.createElement("canvas");
    cnv.width = cnv.height = size;
    const ctx2d = cnv.getContext("2d")!;
    ctx2d.drawImage(bmp, 0, 0, size, size);
    return new Uint8Array(ctx2d.getImageData(0, 0, size, size).data);
  }

  function onPrimImage(prim: Primitive) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.addEventListener("change", async () => {
      const f = input.files?.[0];
      if (!f) return;
      try {
        prim.imageLayer = scene.addPrimImage(await fileToLayer(f, scene.primImageSize));
        prim.pattern = PrimPattern.Image;
        scene.touch();
        history.commit();
        ui.refresh();
      } catch (e) {
        hud.textContent = `Image load failed: ${e instanceof Error ? e.message : e}`;
      }
    });
    input.click();
  }

  // Convert an SDF primitive into an editable mesh and drop straight into the
  // point editor (the SDF prim is replaced by the new instance).
  function editPrim(prim: Primitive) {
    const { model, group } = convertPrimitive(prim);
    const inst = scene.addModel(model, prim.position.clone());
    inst.rotation.copy(prim.rotation);
    preview.addInstance(inst.id, group, []);
    scene.remove(prim);
    scene.touchInstances();
    history.commit(); // converting a prim → mesh is one undo step
    ui.select(inst);
    toggleEdit(inst); // enter vertex editing (forces ray-traced view)
    ui.refresh();
  }

  // Upload a base-color texture for one material of an imported model. The
  // model's UVs already exist, so the image maps correctly (fixes white imports
  // whose textures were external/missing — e.g. many FBX exports).
  function onMeshTexture(blasIndex: number, mi: number) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.addEventListener("change", async () => {
      const f = input.files?.[0];
      if (!f) return;
      try {
        const layer = await fileToLayer(f, scene.meshTexSize);
        scene.setMaterial(blasIndex, mi, { texLayer: scene.addMeshTexture(layer) });
        history.commit();
        ui.refresh();
      } catch (e) {
        hud.textContent = `Texture load failed: ${e instanceof Error ? e.message : e}`;
      }
    });
    input.click();
  }

  // Drop triangles whose (local) material id is hidden, from a 32-float tri array.
  function dropHiddenTris(tris: Float32Array, hidden: Set<number>): Float32Array {
    const STRIDE = 32;
    const tcount = tris.length / STRIDE;
    const kept: number[] = [];
    for (let t = 0; t < tcount; t++) if (!hidden.has(tris[t * STRIDE + 30])) kept.push(t);
    const out = new Float32Array(kept.length * STRIDE);
    kept.forEach((t, i) => out.set(tris.subarray(t * STRIDE, (t + 1) * STRIDE), i * STRIDE));
    return out;
  }

  // Bake a boolean group (any mix of primitives + models) into one multi-material
  // CSG mesh (textures preserved), then remove the inputs.
  function bakeBoolean(groupId: number) {
    if (groupId <= 0) return;
    const inputs: BoolInput[] = [];
    const consumed: (Primitive | MeshInstance)[] = [];

    for (const p of scene.prims) {
      if (p.group !== groupId) continue;
      consumed.push(p);
      const desc: MatDesc = {
        name: p.name,
        color: [...p.color],
        metalness: p.reflectivity,
        roughness: Math.max(0.04, 1 - p.reflectivity),
        tex: null,
        normal: null,
      };
      inputs.push({ geometry: primitiveGeometry(p), descs: [desc], subtractive: p.subtractive });
    }
    preview.syncInstances(scene);
    for (const inst of scene.instances) {
      if (inst.group !== groupId) continue;
      const root = preview.instanceRoot(inst.id);
      if (!root) continue;
      consumed.push(inst);
      const { tris, matCount } = worldMesh(root);
      const descs: MatDesc[] = [];
      const hidden = new Set<number>();
      for (let k = 0; k < matCount; k++) {
        descs.push(scene.materialDescriptor(inst.blasIndex, k));
        if (scene.getMaterial(inst.blasIndex, k).hidden) hidden.add(k);
      }
      // Drop hidden-material triangles (e.g. toon-outline shells) before CSG.
      const geomTris = hidden.size ? dropHiddenTris(tris, hidden) : tris;
      inputs.push({ geometry: trisToGeometry(geomTris), descs, subtractive: inst.subtractive });
    }

    if (!inputs.some((i) => !i.subtractive)) {
      hud.textContent = `Boolean group ${groupId}: needs at least one Solid object.`;
      return;
    }
    let built: ReturnType<typeof csgBuildModel>;
    try {
      built = csgBuildModel(inputs, `Boolean ${groupId}`);
    } catch (e) {
      hud.textContent = `Boolean failed: ${e instanceof Error ? e.message : e}`;
      return;
    }
    if (!built) return;

    const newInst = scene.addModel(built.model, built.center);
    preview.addInstance(newInst.id, built.group, []);

    for (const it of consumed) {
      if (it instanceof MeshInstance) preview.removeInstance(it.id);
      scene.remove(it);
    }
    scene.touchInstances();
    ui.select(newInst);
    mode = "render";
    applyMode();
    history.commit();
    ui.refresh();
  }

  // Generate a procedural landscape as a placeable model. The landform spec
  // (recipe + seed) rides along on the instance so it can be re-authored.
  function addLandscape(type: LandscapeType) {
    const spec = defaultLandform(type);
    const { model, group } = landscapeModel(spec.recipe, spec.seed);
    const spawn = cam.target.clone();
    spawn.y = 0;
    const inst = scene.addModel(model, spawn);
    inst.landform = spec;
    preview.addInstance(inst.id, group, []);
    ui.select(inst);
    history.commit();
    ui.refresh();
  }

  // Scatter many instances of one baked shape across a landform's surface,
  // filtered by altitude band + max slope, with random scale/spin. All clones
  // share a single BLAS (GPU-instancing style), so hundreds stay cheap.
  function scatterEcosystem(landform: MeshInstance, o: ScatterOptions) {
    const lf = landform.landform;
    if (!lf) return;
    const { recipe, seed } = lf;
    const amp = recipe.amplitude;
    const H = LANDSCAPE_HALF;
    const model = landform.modelMatrix(new Matrix4());
    const nrot = new Quaternion().setFromEuler(landform.rotation); // uniform scale → rotates normals
    const e = 0.01; // normalized finite-difference step
    const up = new Vector3(0, 1, 0);
    const hAt = (u: number, v: number) => evaluateHeight(recipe, u, v, seed) * amp;

    const placements: { pos: Vector3; rot: Euler; scale: number }[] = [];
    const maxTries = o.count * 30;
    for (let tries = 0; tries < maxTries && placements.length < o.count; tries++) {
      const u = Math.random() * 2 - 1;
      const v = Math.random() * 2 - 1;
      const t = evaluateHeight(recipe, u, v, seed); // 0..1 normalized height
      if (t < o.altMin || t > o.altMax) continue;
      // Surface normal from local finite differences; uniform scale preserves it.
      const nLocal = new Vector3(hAt(u - e, v) - hAt(u + e, v), 2 * (e * H), hAt(u, v - e) - hAt(u, v + e)).normalize();
      if (1 - nLocal.y > o.maxSlope) continue;

      const pos = new Vector3(u * H, t * amp, v * H).applyMatrix4(model);
      const spin = Math.random() * Math.PI * 2;
      const rot = new Euler();
      if (o.align) {
        const nWorld = nLocal.clone().applyQuaternion(nrot).normalize();
        const q = new Quaternion().setFromUnitVectors(up, nWorld);
        q.multiply(new Quaternion().setFromAxisAngle(up, spin));
        rot.setFromQuaternion(q);
      } else {
        rot.y = spin;
      }
      const scale = o.scaleMin + Math.random() * Math.max(0, o.scaleMax - o.scaleMin);
      placements.push({ pos, rot, scale });
    }

    if (placements.length === 0) {
      hud.textContent = "Scatter: no spots match those slope/altitude limits.";
      return;
    }

    // Bake the chosen shape into one shared BLAS (first instance), then clone.
    const template = new Primitive(o.shape);
    template.color = [o.color[0], o.color[1], o.color[2]];
    const { model: shapeModel } = convertPrimitive(template);
    const label = PRIM_LABELS[o.shape];
    const first = scene.addModel(shapeModel, placements[0].pos.clone());
    first.rotation.copy(placements[0].rot);
    first.scale = placements[0].scale;
    first.name = `${label} scatter 1`;
    const made: MeshInstance[] = [first];
    for (let i = 1; i < placements.length; i++) {
      const inst = scene.addDuplicate(first) as MeshInstance;
      inst.position.copy(placements[i].pos);
      inst.rotation.copy(placements[i].rot);
      inst.scale = placements[i].scale;
      inst.name = `${label} scatter ${i + 1}`;
      made.push(inst);
    }
    scene.touchInstances();
    preview.syncInstances(scene);
    history.commit();
    ui.setSelection(made);
    ui.refresh();
    hud.textContent = `Scattered ${made.length} ${label}(s) on ${landform.name}.`;
  }

  // Re-bake a placed landform in place from its (edited) recipe + seed.
  function regenerateLandform(inst: MeshInstance) {
    if (!inst.landform) return;
    const { model, group } = landscapeModel(inst.landform.recipe, inst.landform.seed);
    // Rebase the freshly-baked tris' material ids into this instance's pool slot.
    const matBase = scene.blasMatBase[inst.blasIndex] ?? 0;
    const tris = model.blas.tris;
    for (let i = 0; i < tris.length; i += 32) tris[i + 30] += matBase;
    scene.replaceBlas(inst.blasIndex, model.blas);
    preview.removeInstance(inst.id);
    preview.addInstance(inst.id, group, []);
    preview.syncInstances(scene);
    history.commit();
    ui.refresh();
  }

  // Re-roll a landform's seed, then re-bake.
  function rerollLandform(inst: MeshInstance) {
    if (!inst.landform) return;
    inst.landform.seed = Math.floor(Math.random() * 1000);
    regenerateLandform(inst);
  }

  // Spawn a glowing planet: a point light with a visible emissive body.
  function addPlanet() {
    const spawn = cam.target.clone();
    spawn.y += 45;
    const l = scene.addLight(LightType.Point, spawn);
    l.bodyRadius = 10;
    l.color = [1.0, 0.84, 0.62];
    l.intensity = 2200;
    l.range = 260;
    l.name = `Planet ${l.id}`;
    scene.touch();
    ui.select(l);
    history.commit();
    ui.refresh();
  }

  // ---- new / save / load scene ----

  // A small three-way modal ("Save" / "Don't save" / "Cancel"). Resolves to the
  // chosen action; backdrop click or Esc counts as Cancel.
  function askSaveChanges(): Promise<"save" | "discard" | "cancel"> {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.className = "modal-backdrop";
      const box = document.createElement("div");
      box.className = "modal";
      const title = document.createElement("div");
      title.className = "modal-title";
      title.textContent = "Unsaved changes";
      const msg = document.createElement("div");
      msg.className = "modal-msg";
      msg.textContent = "Save the current scene before starting a new one?";
      const btns = document.createElement("div");
      btns.className = "modal-btns";

      const close = (val: "save" | "discard" | "cancel") => {
        document.removeEventListener("keydown", onKey, true);
        backdrop.remove();
        resolve(val);
      };
      const onKey = (e: KeyboardEvent) => {
        if (e.key === "Escape") { e.stopPropagation(); close("cancel"); }
      };
      const mk = (label: string, val: "save" | "discard" | "cancel", primary = false) => {
        const b = document.createElement("button");
        b.className = "btn" + (primary ? " primary" : "");
        b.textContent = label;
        b.addEventListener("click", () => close(val));
        return b;
      };
      btns.append(mk("Save", "save", true), mk("Don't save", "discard"), mk("Cancel", "cancel"));
      box.append(title, msg, btns);
      backdrop.append(box);
      backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close("cancel"); });
      document.addEventListener("keydown", onKey, true);
      document.body.append(backdrop);
    });
  }

  // Swap in a fresh scene state (new / gallery preset), dropping models, edit/pose
  // modes, cutscene, and undo history, and reselecting a sensible default.
  function applySceneState(state: SceneState, label: string) {
    scene.restoreState(state);
    editInstance = null; editMesh = null;
    poseInst = null; poseBones = []; selectedBone = -1; boneSel.clear();
    selectedVerts.clear();
    preview.clearInstances();
    preview.syncInstances(scene);
    cutsceneKeys = []; cutsceneMode = false; cutsceneSel = -1; cutsceneTime = 0; cutscenePlaying = false; cutsceneHome = null;
    cutsceneKeysVersion++;
    mode = "render";
    applyMode();
    savedVersion = scene.version;
    history.reset();
    ui.select(scene.prims[0] ?? scene.lights[0] ?? null);
    ui.refresh();
    hud.textContent = label;
  }

  // Reset the live scene back to the vanilla default.
  function resetScene() {
    applySceneState(defaultScene().captureState(), "New scene");
  }

  // New Scene: offer to save first if there are unsaved changes, then reset.
  async function newScene() {
    if (scene.version !== savedVersion) {
      const choice = await askSaveChanges();
      if (choice === "cancel") return;
      if (choice === "save") saveScene();
    }
    resetScene();
  }

  // Point a camera at a framing hint (falls back to a generic origin view).
  function applyCameraHint(c: OrbitCamera, h?: CameraHint): void {
    const f = h ?? { target: [0, 6, 0] as [number, number, number], distance: 80, yaw: -0.6, pitch: 0.24 };
    c.target.set(f.target[0], f.target[1], f.target[2]);
    c.distance = f.distance; c.yaw = f.yaw; c.pitch = f.pitch;
    c.update();
  }
  const cameraHintFromCam = (): CameraHint => ({
    target: [cam.target.x, cam.target.y, cam.target.z],
    distance: cam.distance, yaw: cam.yaw, pitch: cam.pitch,
  });

  const THUMB_W = 256, THUMB_H = 160, THUMB_SAMPLES = 80;

  // Render whatever scene is currently uploaded to the renderer to a PNG data URL
  // at the given framing. Freezes the live loop so it doesn't fight the GPU.
  async function renderThumbnail(hint?: CameraHint): Promise<string> {
    const tcam = new OrbitCamera();
    applyCameraHint(tcam, hint);
    const wasOffline = offlineRendering;
    offlineRendering = true;
    try {
      const px = await renderer.renderToPixels(tcam, THUMB_W, THUMB_H, THUMB_SAMPLES);
      const cnv = document.createElement("canvas");
      cnv.width = THUMB_W; cnv.height = THUMB_H;
      cnv.getContext("2d")!.putImageData(new ImageData(px, THUMB_W, THUMB_H), 0, 0);
      return cnv.toDataURL("image/png");
    } finally {
      offlineRendering = wasOffline;
    }
  }

  // Cached built-in thumbnails (preset id → PNG data URL), ray-traced once on the
  // first gallery open by borrowing the renderer to draw each preset small.
  const thumbCache = new Map<string, string>();
  let thumbsGenerated = false;

  async function ensureThumbnails(): Promise<void> {
    if (thumbsGenerated) return;
    hud.textContent = "Rendering gallery previews…";
    try {
      for (const preset of SCENE_PRESETS) {
        if (thumbCache.has(preset.id)) continue;
        const s = preset.build();
        renderer.uploadScene(s);
        renderer.uploadInstances(s);
        renderer.uploadWorld(s);
        renderer.uploadPrimTextures(s);
        thumbCache.set(preset.id, await renderThumbnail(preset.camera));
      }
      thumbsGenerated = true;
    } finally {
      // Restore the live scene into the renderer so the viewport is correct on
      // cancel (the version-gated frame loop won't re-upload an unchanged scene).
      renderer.uploadScene(scene);
      renderer.uploadInstances(scene);
      renderer.uploadWorld(scene);
      renderer.uploadPrimTextures(scene);
      hud.textContent = "";
    }
  }

  // ---- user-saved gallery scenes (persisted in localStorage) ----
  interface CustomScene {
    id: string;
    name: string;
    description: string;
    thumbnail: string;   // PNG data URL
    camera: CameraHint;  // framing captured at save time
    scene: string;       // serialized .aerie JSON
    created: number;
  }
  const GALLERY_KEY = "aerie.gallery.v1";

  function loadCustomScenes(): CustomScene[] {
    try {
      const raw = localStorage.getItem(GALLERY_KEY);
      const list = raw ? JSON.parse(raw) : [];
      return Array.isArray(list) ? list : [];
    } catch {
      return [];
    }
  }
  function persistCustomScenes(list: CustomScene[]): boolean {
    try {
      localStorage.setItem(GALLERY_KEY, JSON.stringify(list));
      return true;
    } catch {
      return false; // quota exceeded (e.g. large embedded models) or disabled
    }
  }

  // Save the current scene as a gallery entry: capture framing, render a
  // thumbnail, serialize, and persist locally.
  async function saveToGallery() {
    const meta = await promptSceneMeta();
    if (!meta) return;
    hud.textContent = "Saving to gallery…";
    // Make sure the renderer holds the live scene before we thumbnail it.
    renderer.uploadScene(scene);
    renderer.uploadInstances(scene);
    renderer.uploadWorld(scene);
    renderer.uploadPrimTextures(scene);
    const camera = cameraHintFromCam();
    const thumbnail = await renderThumbnail(camera);
    const entry: CustomScene = {
      id: `custom-${Date.now()}`,
      name: meta.name,
      description: meta.description,
      thumbnail,
      camera,
      scene: serializeCurrentScene(),
      created: Date.now(),
    };
    const list = loadCustomScenes();
    list.push(entry);
    hud.textContent = persistCustomScenes(list)
      ? `Saved “${entry.name}” to gallery.`
      : "Couldn't save — local storage is full. Try Export instead (models make scenes large).";
  }

  // Download a gallery entry as a portable .aeriescene file (self-contained:
  // thumbnail + framing + scene).
  function exportCustomScene(entry: CustomScene): void {
    const blob = new Blob([JSON.stringify(entry)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${entry.name.replace(/[^\w -]+/g, "_") || "scene"}.aeriescene`;
    a.click();
    URL.revokeObjectURL(url);
  }

  // Import a .aeriescene file into the local gallery (fresh id to avoid clashes).
  async function importToGallery(file: File): Promise<void> {
    try {
      const entry = JSON.parse(await file.text()) as CustomScene;
      if (typeof entry.scene !== "string" || typeof entry.name !== "string") {
        throw new Error("not a gallery scene file");
      }
      entry.id = `custom-${Date.now()}`;
      const list = loadCustomScenes();
      list.push(entry);
      hud.textContent = persistCustomScenes(list)
        ? `Imported “${entry.name}” to gallery.`
        : "Import failed — local storage is full.";
    } catch (e) {
      hud.textContent = `Import failed: ${e instanceof Error ? e.message : e}`;
    }
  }

  // Load a saved gallery scene (full deserialize path) and frame it.
  async function loadCustomScene(entry: CustomScene): Promise<void> {
    await loadSceneFromText(entry.scene, `Loaded “${entry.name}”`);
    applyCameraHint(cam, entry.camera);
  }

  // A small modal that collects a name + description for a new gallery entry.
  function promptSceneMeta(): Promise<{ name: string; description: string } | null> {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.className = "modal-backdrop";
      const box = document.createElement("div");
      box.className = "modal";
      const title = document.createElement("div");
      title.className = "modal-title";
      title.textContent = "Save to gallery";
      const nameIn = document.createElement("input");
      nameIn.className = "modal-input";
      nameIn.placeholder = "Scene name";
      nameIn.value = "My Scene";
      const descIn = document.createElement("input");
      descIn.className = "modal-input";
      descIn.placeholder = "Short description (optional)";
      const btns = document.createElement("div");
      btns.className = "modal-btns";
      const close = (val: { name: string; description: string } | null) => {
        document.removeEventListener("keydown", onKey, true);
        backdrop.remove();
        resolve(val);
      };
      const commit = () => {
        const name = nameIn.value.trim();
        if (!name) { nameIn.focus(); return; }
        close({ name, description: descIn.value.trim() });
      };
      const onKey = (e: KeyboardEvent) => {
        if (e.key === "Escape") { e.stopPropagation(); close(null); }
        if (e.key === "Enter") { e.stopPropagation(); commit(); }
      };
      const cancel = document.createElement("button");
      cancel.className = "btn";
      cancel.textContent = "Cancel";
      cancel.addEventListener("click", () => close(null));
      const save = document.createElement("button");
      save.className = "btn primary";
      save.textContent = "Save";
      save.addEventListener("click", commit);
      btns.append(cancel, save);
      box.append(title, nameIn, descIn, btns);
      backdrop.append(box);
      backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(null); });
      document.addEventListener("keydown", onKey, true);
      document.body.append(backdrop);
      nameIn.focus();
      nameIn.select();
    });
  }

  type GalleryChoice = { builtin: ScenePreset } | { custom: CustomScene };

  // A modal grid of built-in starter scenes plus the user's saved scenes.
  // Custom cards carry Export/Delete actions; resolves to the picked scene.
  function pickPreset(): Promise<GalleryChoice | null> {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.className = "modal-backdrop";
      const box = document.createElement("div");
      box.className = "modal gallery";
      const title = document.createElement("div");
      title.className = "modal-title";
      title.textContent = "Gallery — start from a scene";

      const close = (val: GalleryChoice | null) => {
        document.removeEventListener("keydown", onKey, true);
        backdrop.remove();
        resolve(val);
      };
      const onKey = (e: KeyboardEvent) => {
        if (e.key === "Escape") { e.stopPropagation(); close(null); }
      };

      const makeCard = (o: {
        thumb?: string; name: string; desc: string; onClick: () => void;
        onExport?: () => void; onDelete?: () => void;
      }): HTMLElement => {
        // A div (not <button>) so it can legally contain the action buttons.
        const card = document.createElement("div");
        card.className = "gallery-card";
        card.tabIndex = 0;
        if (o.thumb) {
          const img = document.createElement("img");
          img.className = "gallery-card-thumb";
          img.src = o.thumb; img.alt = o.name;
          card.append(img);
        }
        const name = document.createElement("div");
        name.className = "gallery-card-name";
        name.textContent = o.name;
        const desc = document.createElement("div");
        desc.className = "gallery-card-desc";
        desc.textContent = o.desc;
        card.append(name, desc);
        if (o.onExport || o.onDelete) {
          const actions = document.createElement("div");
          actions.className = "gallery-card-actions";
          if (o.onExport) {
            const ex = document.createElement("button");
            ex.className = "gallery-card-act";
            ex.textContent = "Export";
            ex.addEventListener("click", (e) => { e.stopPropagation(); o.onExport!(); });
            actions.append(ex);
          }
          if (o.onDelete) {
            const del = document.createElement("button");
            del.className = "gallery-card-act danger";
            del.textContent = "Delete";
            del.addEventListener("click", (e) => { e.stopPropagation(); o.onDelete!(); });
            actions.append(del);
          }
          card.append(actions);
        }
        card.addEventListener("click", o.onClick);
        return card;
      };

      const body = document.createElement("div");
      const renderCards = () => {
        body.textContent = "";
        const grid = document.createElement("div");
        grid.className = "gallery-grid";
        for (const p of SCENE_PRESETS) {
          grid.append(makeCard({
            thumb: thumbCache.get(p.id), name: p.name, desc: p.description,
            onClick: () => close({ builtin: p }),
          }));
        }
        body.append(grid);

        const custom = loadCustomScenes();
        if (custom.length) {
          const sub = document.createElement("div");
          sub.className = "gallery-section";
          sub.textContent = "Your scenes";
          body.append(sub);
          const cgrid = document.createElement("div");
          cgrid.className = "gallery-grid";
          for (const entry of custom) {
            cgrid.append(makeCard({
              thumb: entry.thumbnail, name: entry.name, desc: entry.description,
              onClick: () => close({ custom: entry }),
              onExport: () => exportCustomScene(entry),
              onDelete: () => { persistCustomScenes(loadCustomScenes().filter((c) => c.id !== entry.id)); renderCards(); },
            }));
          }
          body.append(cgrid);
        }
      };
      renderCards();

      const btns = document.createElement("div");
      btns.className = "modal-btns";
      const cancel = document.createElement("button");
      cancel.className = "btn";
      cancel.textContent = "Cancel";
      cancel.addEventListener("click", () => close(null));
      btns.append(cancel);

      box.append(title, body, btns);
      backdrop.append(box);
      backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(null); });
      document.addEventListener("keydown", onKey, true);
      document.body.append(backdrop);
    });
  }

  // Gallery: offer to save unsaved work, pick a scene, then load it (built-in
  // presets rebuild synchronously; custom scenes go through the full loader).
  async function openGallery() {
    if (scene.version !== savedVersion) {
      const choice = await askSaveChanges();
      if (choice === "cancel") return;
      if (choice === "save") saveScene();
    }
    await ensureThumbnails(); // one-time: ray-trace a preview of each built-in
    const pick = await pickPreset();
    if (!pick) return;
    if ("builtin" in pick) {
      applySceneState(pick.builtin.build().captureState(), `Loaded “${pick.builtin.name}”`);
      applyCameraHint(cam, pick.builtin.camera);
    } else {
      await loadCustomScene(pick.custom);
    }
  }

  // Serialize the live scene to an .aerie JSON string (models, poses, cutscene).
  function serializeCurrentScene(): string {
    // Serialize the canonical (home) object/DoF state, not a mid-cutscene pose.
    if (cutsceneMode) { restoreHome(); cutsceneTime = 0; cutscenePlaying = false; ui.refresh(); }
    // Object ids regenerate on load, so persist cutscene object refs as positional
    // slots over [prims…, instances…] (which save/load preserves in order).
    const slotOf = new Map<number, number>();
    let slot = 0;
    for (const p of scene.prims) slotOf.set(p.id, slot++);
    for (const m of scene.instances) slotOf.set(m.id, slot++);
    const cutsceneOut: CamKey[] = cutsceneKeys.map((k) => ({
      ...k,
      objects: k.objects
        ?.map((o) => ({ ...o, id: slotOf.get(o.id) ?? -1 }))
        .filter((o) => o.id >= 0),
    }));
    return serializeScene(scene, {
      fileFor: (bi) => scene.blasFile[bi],
      poseFor: (id) => {
        const bones = preview.getBones(id);
        if (!bones) return null;
        return {
          bones: bones.map((b) => ({
            p: [b.position.x, b.position.y, b.position.z] as [number, number, number],
            q: [b.quaternion.x, b.quaternion.y, b.quaternion.z, b.quaternion.w] as [number, number, number, number],
          })),
          paused: preview.isPosing(id),
        };
      },
      cutscene: cutsceneOut,
    });
  }

  function saveScene() {
    const json = serializeCurrentScene();
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `aerie-${Date.now()}.aerie`;
    a.click();
    URL.revokeObjectURL(url);
    savedVersion = scene.version; // scene now matches what's on disk
  }

  async function loadScene(file: File) {
    await loadSceneFromText(await file.text(), `Loaded ${file.name}`);
  }

  async function loadSceneFromText(text: string, label: string) {
    try {
      const data = deserializeScene(scene, text);
      // Exit any per-object modes that reference now-gone objects.
      editInstance = null; editMesh = null;
      poseInst = null; poseBones = []; selectedBone = -1; boneSel.clear();
      preview.clearInstances();

      const savedInstances = data.mesh?.instances ?? [];
      for (let i = 0; i < scene.instances.length; i++) {
        const inst = scene.instances[i];
        const sd = savedInstances[i];
        let rebuilt = false;
        if (sd?.file) {
          try {
            const bytes = b64ToU8(sd.file.bytes);
            const res = await importModel(new File([bytes], sd.file.name));
            preview.addInstance(inst.id, res.group, res.animations);
            scene.blasFile[inst.blasIndex] = { name: sd.file.name, bytes };
            if (sd.pose) {
              const bones = preview.getBones(inst.id);
              if (bones) {
                for (let k = 0; k < bones.length && k < sd.pose.bones.length; k++) {
                  const bp = sd.pose.bones[k];
                  bones[k].position.set(bp.p[0], bp.p[1], bp.p[2]);
                  bones[k].quaternion.set(bp.q[0], bp.q[1], bp.q[2], bp.q[3]);
                }
                if (sd.pose.paused) preview.setPosing(inst.id, true);
              }
            }
            rebuilt = true;
          } catch {
            /* re-import failed → fall back to a baked preview below */
          }
        }
        if (!rebuilt) preview.addInstance(inst.id, previewFromBlas(scene, inst.blasIndex), []);
      }

      // Map serialized object slots back to the freshly-created objects' ids.
      const slotToId: number[] = [];
      for (const p of scene.prims) slotToId.push(p.id);
      for (const m of scene.instances) slotToId.push(m.id);
      cutsceneKeys = (Array.isArray(data.cutscene) ? (data.cutscene as CamKey[]) : []).map((k) => ({
        ...k,
        // Backfill atmosphere for scenes saved before it was animatable, so
        // older cutscenes keep their (fixed) look rather than reading NaN.
        timeOfDay: k.timeOfDay ?? scene.world.timeOfDay,
        exposure: k.exposure ?? scene.world.exposure,
        haze: k.haze ?? scene.world.hazeDensity,
        objects: (k.objects as ObjXform[] | undefined)
          ?.map((o) => ({ ...o, id: slotToId[o.id] ?? -1 }))
          .filter((o) => o.id >= 0),
      }));
      cutsceneSel = cutsceneKeys.length ? 0 : -1;
      cutsceneTime = 0; cutscenePlaying = false; cutsceneMode = false; cutsceneHome = null;
      if (cutsceneKeys.length) cutsceneKeys[0].duration = 0;
      cutsceneKeysVersion++;

      ui.select(scene.prims[0] ?? scene.instances[0] ?? scene.lights[0] ?? null);
      history.commit();
      savedVersion = scene.version; // freshly loaded scene matches its file
      ui.refresh();
      hud.textContent = label;
    } catch (e) {
      hud.textContent = `Load failed: ${e instanceof Error ? e.message : e}`;
    }
  }

  let renderingImage = false;
  async function onRenderImage() {
    if (renderingImage) return;
    renderingImage = true;
    const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    const W = 1920;
    const H = Math.round(W / aspect);
    const prevHud = hud.textContent;
    hud.textContent = `Rendering ${W}×${H} PNG…`;
    try {
      cam.update();
      const blob = await renderer.renderToImage(cam, W, H, 240);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `aerie-${Date.now()}.png`;
      a.click();
      URL.revokeObjectURL(url);
      hud.textContent = prevHud;
    } catch (e) {
      hud.textContent = `Render failed: ${e instanceof Error ? e.message : e}`;
    } finally {
      renderingImage = false;
    }
  }

  // Resolve the scene object under a screen point (lights tested first, then a
  // GPU pick for prims/meshes). Used by drag-to-apply from the material dock.
  async function pickObjectAt(clientX: number, clientY: number): Promise<Primitive | MeshInstance | Light | null> {
    const light = pickLight(clientX, clientY);
    if (light) return light;
    cam.update();
    const dpr = renderer.width / Math.max(1, canvas.clientWidth);
    const r = await renderer.pick(cam, clientX * dpr, clientY * dpr);
    if (r.kind === 4 && r.index >= 0 && r.index < scene.prims.length) return scene.prims[r.index];
    if (r.kind === 5 && r.index >= 0 && r.index < scene.instances.length) return scene.instances[r.index];
    return null;
  }

  // While true, the live rAF loop idles so an off-screen export (turntable /
  // cutscene) owns the GPU and uniforms.
  let offlineRendering = false;

  // A non-dismissable, full-screen modal shown for the duration of an off-screen
  // WebM export. The backdrop covers (and so blocks pointer input to) every panel
  // and dock, and we also swallow keyboard shortcuts — the export owns the camera
  // and uniforms, so the scene must not be editable while it runs. There is no
  // close affordance; it is torn down only when the render finishes or fails.
  function showRenderModal(label: string): {
    update: (done: number, total: number, sub: string, etaMs: number) => void;
    cancelled: () => boolean;
    close: () => void;
  } {
    // Human-readable "time left" from a millisecond estimate. Empty until we
    // have a real estimate (etaMs <= 0), so the first frame reads "estimating…".
    const fmtETA = (ms: number): string => {
      if (!isFinite(ms) || ms <= 0) return "";
      const s = Math.ceil(ms / 1000);
      const m = Math.floor(s / 60);
      return m > 0 ? `${m}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`;
    };

    const backdrop = document.createElement("div");
    backdrop.className = "modal-backdrop";
    backdrop.style.zIndex = "200"; // above menubar/docks/panels
    const box = document.createElement("div");
    box.className = "modal";
    const title = document.createElement("div");
    title.className = "modal-title";
    title.textContent = `Rendering ${label}…`;
    const msg = document.createElement("div");
    msg.className = "modal-msg";
    msg.textContent = "Preparing…";
    const bar = document.createElement("div");
    bar.style.cssText =
      "height:8px;border-radius:5px;border:1px solid var(--line);" +
      "background:rgba(120,170,255,0.12);overflow:hidden;";
    const fill = document.createElement("div");
    fill.style.cssText =
      "height:100%;width:0%;background:rgba(140,200,255,0.8);transition:width 0.12s linear;";
    bar.append(fill);
    const hint = document.createElement("div");
    hint.className = "modal-msg";
    hint.style.cssText = "margin:10px 0 0;opacity:0.55;";
    hint.textContent = "Editing is disabled until the render finishes.";

    let cancelled = false;
    const btns = document.createElement("div");
    btns.className = "modal-btns";
    btns.style.marginTop = "14px";
    const cancelBtn = document.createElement("button");
    cancelBtn.className = "btn";
    cancelBtn.textContent = "Cancel render";
    cancelBtn.addEventListener("click", () => {
      cancelled = true;
      cancelBtn.disabled = true;
      cancelBtn.textContent = "Cancelling…";
      title.textContent = `Cancelling ${label}…`;
      // The current frame must finish tracing before the loop can bail; note that.
      hint.textContent = "Finishing the current frame, then stopping…";
    });
    btns.append(cancelBtn);
    box.append(title, msg, bar, hint, btns);
    backdrop.append(box);

    // Swallow every key while the export runs so shortcuts can't fly the camera
    // or mutate the scene under the render. Pointer input is already blocked by
    // the backdrop; clicks on it do nothing (no dismiss).
    const swallowKey = (e: KeyboardEvent) => { e.stopPropagation(); };
    document.addEventListener("keydown", swallowKey, true);
    document.addEventListener("keyup", swallowKey, true);
    backdrop.addEventListener("click", (e) => e.stopPropagation());
    document.body.append(backdrop);

    return {
      update: (done, total, sub, etaMs) => {
        const pct = total > 0 ? Math.round((done / total) * 100) : 0;
        fill.style.width = `${pct}%`;
        const eta = fmtETA(etaMs);
        const tail = cancelled ? "" : eta ? ` · ~${eta} left` : done > 0 ? " · estimating…" : "";
        msg.textContent = `${sub} · ${done}/${total} · ${pct}%${tail}`;
      },
      cancelled: () => cancelled,
      close: () => {
        document.removeEventListener("keydown", swallowKey, true);
        document.removeEventListener("keyup", swallowKey, true);
        backdrop.remove();
      },
    };
  }

  // Asked after a cancelled export: keep the frames that made it, or throw them
  // away? Resolves true = save the partial clip. Backdrop click or Esc = discard.
  function askPartialSave(label: string, done: number, total: number): Promise<boolean> {
    return new Promise((resolve) => {
      const backdrop = document.createElement("div");
      backdrop.className = "modal-backdrop";
      backdrop.style.zIndex = "200";
      const box = document.createElement("div");
      box.className = "modal";
      const title = document.createElement("div");
      title.className = "modal-title";
      title.textContent = `${label} cancelled`;
      const msg = document.createElement("div");
      msg.className = "modal-msg";
      msg.textContent = `${done} of ${total} frames rendered. Save the partial video?`;
      const btns = document.createElement("div");
      btns.className = "modal-btns";
      const close = (val: boolean) => {
        document.removeEventListener("keydown", onKey, true);
        backdrop.remove();
        resolve(val);
      };
      const onKey = (e: KeyboardEvent) => {
        if (e.key === "Escape") { e.stopPropagation(); close(false); }
      };
      const mk = (lbl: string, val: boolean, primary = false) => {
        const b = document.createElement("button");
        b.className = "btn" + (primary ? " primary" : "");
        b.textContent = lbl;
        b.addEventListener("click", () => close(val));
        return b;
      };
      btns.append(mk("Save partial", true, true), mk("Discard", false));
      box.append(title, msg, btns);
      backdrop.append(box);
      backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(false); });
      document.addEventListener("keydown", onKey, true);
      document.body.append(backdrop);
    });
  }

  // Encode an off-screen render to WebM with WebCodecs. Each frame is stamped
  // with an explicit presentation timestamp (i / fps), so the clip's duration is
  // exactly frames/fps regardless of how long each frame takes to ray-trace —
  // unlike MediaRecorder, which captures in wall-clock real time.
  async function recordWebM(
    label: string,
    name: string,
    W: number, H: number, frames: number, fps: number,
    renderFrame: (i: number) => Promise<Uint8ClampedArray<ArrayBuffer>>,
  ): Promise<void> {
    if (offlineRendering) return;
    if (typeof VideoEncoder === "undefined") {
      hud.textContent = `${label}: video export needs WebCodecs (use a recent Chrome/Edge).`;
      return;
    }
    const candidates = [
      { mux: "V_VP9", enc: "vp09.00.41.08" },
      { mux: "V_VP9", enc: "vp09.00.31.08" },
      { mux: "V_VP8", enc: "vp8" },
    ];
    let codec: { mux: string; enc: string } | null = null;
    for (const c of candidates) {
      const ok = await VideoEncoder.isConfigSupported({ codec: c.enc, width: W, height: H, bitrate: 12_000_000, framerate: fps });
      if (ok.supported) { codec = c; break; }
    }
    if (!codec) { hud.textContent = `${label}: no supported video codec at ${W}×${H}.`; return; }

    const muxer = new Muxer({
      target: new ArrayBufferTarget(),
      video: { codec: codec.mux, width: W, height: H, frameRate: fps },
    });
    const encoder = new VideoEncoder({
      output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
      error: (e) => { hud.textContent = `${label} encode error: ${e.message}`; },
    });
    encoder.configure({ codec: codec.enc, width: W, height: H, bitrate: 12_000_000, framerate: fps });

    const cnv = document.createElement("canvas");
    cnv.width = W; cnv.height = H;
    const ctx = cnv.getContext("2d")!;
    const usPerFrame = 1_000_000 / fps;

    offlineRendering = true;
    const modal = showRenderModal(label);
    modal.update(0, frames, `${W}×${H}`, 0);
    const startMs = performance.now();
    let done = 0;
    // Flush whatever frames the encoder has, mux them, and trigger the download.
    const saveClip = async (partial: boolean) => {
      await encoder.flush();
      muxer.finalize();
      const blob = new Blob([muxer.target.buffer], { type: "video/webm" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      URL.revokeObjectURL(url);
      hud.textContent = partial
        ? `${label} saved (partial) · ${done} frames @ ${fps}fps`
        : `${label} saved · ${frames} frames @ ${fps}fps (${(frames / fps).toFixed(1)}s)`;
    };
    try {
      for (let i = 0; i < frames; i++) {
        if (modal.cancelled()) break; // abort before starting the next frame
        const px = await renderFrame(i);
        ctx.putImageData(new ImageData(px, W, H), 0, 0);
        const vf = new VideoFrame(cnv, { timestamp: Math.round(i * usPerFrame), duration: Math.round(usPerFrame) });
        encoder.encode(vf, { keyFrame: i % fps === 0 }); // a keyframe each second for seeking
        vf.close();
        done = i + 1;
        // ETA from the running average frame time × frames still to go.
        const eta = (performance.now() - startMs) / done * (frames - done);
        hud.textContent = `${label} ${done}/${frames} · ${W}×${H}`;
        modal.update(done, frames, `${W}×${H}`, eta);
        if (encoder.encodeQueueSize > 6) await new Promise((r) => setTimeout(r, 0)); // light backpressure
      }
      if (modal.cancelled()) {
        // Take down the blocking overlay, then ask whether to keep the frames
        // rendered so far. Nothing to offer if the abort landed before frame 1.
        modal.close();
        if (done > 0 && await askPartialSave(label, done, frames)) {
          await saveClip(true);
        } else {
          hud.textContent = `${label} cancelled · ${done}/${frames} frames`;
        }
      } else {
        await saveClip(false);
      }
    } catch (e) {
      hud.textContent = `${label} failed: ${e instanceof Error ? e.message : e}`;
    } finally {
      try { encoder.close(); } catch { /* already closed */ }
      renderer.resetAccumulation();
      offlineRendering = false;
      modal.close();
    }
  }

  // 360° turntable around the camera target → WebM.
  async function renderTurntable(o: TurntableOptions) {
    if (offlineRendering) return;
    if (mode !== "render") { mode = "render"; applyMode(); ui.refresh(); }
    const W = Math.max(16, Math.round(o.width));
    const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    const H = Math.max(16, Math.round(W / aspect));
    const frames = Math.max(1, Math.round(o.seconds * o.fps));
    const startYaw = cam.yaw;
    const shutter = 1 / o.fps;
    // Rigs keep playing while the camera orbits: sample each clip at this frame's
    // time and bake it, exactly as the cutscene export does.
    const bakeRestore = new Map<number, BlasSnap>();
    preview.resetAllSprings();
    await recordWebM("Turntable", `aerie-turntable-${Date.now()}.webm`, W, H, frames, o.fps, (i) => {
      const tSec = i / o.fps;
      cam.yaw = startYaw + (i / frames) * Math.PI * 2;
      cam.update();
      preview.sampleAnimationAt(tSec);
      let rebaked = false;
      for (const inst of scene.instances) {
        if (!preview.hasPlayingClip(inst.id) && !preview.hasSprings(inst.id)) continue;
        if (bakeRigFrame(inst, null, shutter, bakeRestore)) rebaked = true;
      }
      if (rebaked) renderer.uploadMeshPools(scene); // re-uploads geometry (+ instances)
      // Animate emitters over the spin so fire/smoke live; motion-blur across 1/fps.
      if (scene.emitters.length) uploadSceneParticles(tSec, shutter);
      return renderer.renderToPixels(cam, W, H, o.samples);
    });
    restoreBakedBlas(bakeRestore);
    cam.yaw = startYaw;
    cam.update();
    uploadedParticleTime = NaN; // force a re-freeze on the next live frame
  }

  // ---- cutscene (camera + DoF keyframe timeline) ----
  let cutsceneKeys: CamKey[] = [];
  let cutsceneMode = false;    // dock visible / editing
  let cutsceneSel = -1;        // selected keyframe index
  let cutsceneTime = 0;        // playhead seconds
  let cutscenePlaying = false;
  // Bumped whenever the keys change, to invalidate the keyed-skeleton cache.
  let cutsceneKeysVersion = 0;
  let lastCutsceneSync = 0;    // throttle UI refresh during playback

  const normalizeCutscene = () => { if (cutsceneKeys.length) cutsceneKeys[0].duration = 0; };
  // Canonical object transforms + DoF captured on cutscene entry; restored on
  // exit/save so the cutscene's per-frame mutation never disturbs the saved scene.
  type HomeXform = { id: number; pos: [number, number, number]; rot: [number, number, number]; scale: number };
  // Full atmosphere snapshot: the cutscene animates timeOfDay (which re-derives
  // sun + sky), so we save the derived state verbatim and restore it as-is —
  // preserving a custom sky rather than re-deriving it from a time value.
  type AtmoSnap = {
    timeOfDay: number; exposure: number; haze: number;
    zenith: [number, number, number]; horizon: [number, number, number]; stars: boolean;
    sunDir: [number, number, number] | null; sunColor: [number, number, number] | null; sunIntensity: number;
  };
  let cutsceneHome: { xforms: HomeXform[]; aperture: number; focusDistance: number; atmo: AtmoSnap } | null = null;

  function captureAtmo(): AtmoSnap {
    const w = scene.world;
    const sun = scene.lights.find((l) => l.type === LightType.Directional);
    return {
      timeOfDay: w.timeOfDay, exposure: w.exposure, haze: w.hazeDensity,
      zenith: [...w.zenith], horizon: [...w.horizon], stars: w.starsEnabled,
      sunDir: sun ? [sun.direction.x, sun.direction.y, sun.direction.z] : null,
      sunColor: sun ? [...sun.color] : null,
      sunIntensity: sun ? sun.intensity : 0,
    };
  }

  function restoreAtmo(a: AtmoSnap): void {
    const w = scene.world;
    w.timeOfDay = a.timeOfDay; w.exposure = a.exposure; w.hazeDensity = a.haze;
    w.zenith = [...a.zenith]; w.horizon = [...a.horizon]; w.starsEnabled = a.stars;
    const sun = scene.lights.find((l) => l.type === LightType.Directional);
    if (sun && a.sunDir && a.sunColor) {
      sun.direction.set(a.sunDir[0], a.sunDir[1], a.sunDir[2]);
      sun.color = [...a.sunColor];
      sun.intensity = a.sunIntensity;
    }
    scene.touchWorld();
    scene.touch();
  }
  const csQuat = new Quaternion();

  // Snapshot every animatable object's transform (prims: pos+rot; meshes: +scale).
  function captureObjects(): ObjXform[] {
    const out: ObjXform[] = [];
    for (const p of scene.prims) {
      csQuat.setFromEuler(p.rotation);
      out.push({ id: p.id, pos: [p.position.x, p.position.y, p.position.z], quat: [csQuat.x, csQuat.y, csQuat.z, csQuat.w], scale: 1 });
    }
    for (const m of scene.instances) {
      csQuat.setFromEuler(m.rotation);
      // Rigged instances also record their skeleton, so the shot animates at the
      // joints. Bone order is the instance's skeleton order (stable per rig).
      //
      // But NOT when a clip owns the skeleton: a keyed pose is stamped over clip
      // playback, so snapshotting one here would silently pin the rig to whatever
      // frame it happened to be on — the model would travel along the shot while
      // standing perfectly still. Keys carry a pose when YOU posed the rig.
      const bones = preview.hasPlayingClip(m.id) ? null : preview.getBones(m.id);
      const pose = bones?.map((b, i) => ({
        i, q: [b.quaternion.x, b.quaternion.y, b.quaternion.z, b.quaternion.w] as [number, number, number, number],
      }));
      out.push({ id: m.id, pos: [m.position.x, m.position.y, m.position.z], quat: [csQuat.x, csQuat.y, csQuat.z, csQuat.w], scale: m.scale, pose });
    }
    for (const e of scene.emitters) {
      out.push({ id: e.id, pos: [e.position.x, e.position.y, e.position.z], quat: [0, 0, 0, 1], scale: 1 });
    }
    return out;
  }

  function captureKey(): CamKey {
    return {
      target: [cam.target.x, cam.target.y, cam.target.z],
      distance: cam.distance, yaw: cam.yaw, pitch: cam.pitch,
      aperture: scene.world.aperture, focusDistance: scene.world.focusDistance,
      timeOfDay: scene.world.timeOfDay, exposure: scene.world.exposure, haze: scene.world.hazeDensity,
      duration: cutsceneKeys.length === 0 ? 0 : 2,
      ease: "smooth",
      bezier: [0.42, 0, 0.58, 1], // ease in-out by default (curve editor changes it)
      objects: captureObjects(),
    };
  }

  const objectsById = () => {
    const m = new Map<number, Primitive | MeshInstance | Emitter>();
    for (const p of scene.prims) m.set(p.id, p);
    for (const inst of scene.instances) m.set(inst.id, inst);
    for (const e of scene.emitters) m.set(e.id, e);
    return m;
  };

  function captureHome() {
    const xforms: HomeXform[] = [];
    for (const p of scene.prims) xforms.push({ id: p.id, pos: [p.position.x, p.position.y, p.position.z], rot: [p.rotation.x, p.rotation.y, p.rotation.z], scale: 1 });
    for (const m of scene.instances) xforms.push({ id: m.id, pos: [m.position.x, m.position.y, m.position.z], rot: [m.rotation.x, m.rotation.y, m.rotation.z], scale: m.scale });
    for (const e of scene.emitters) xforms.push({ id: e.id, pos: [e.position.x, e.position.y, e.position.z], rot: [0, 0, 0], scale: 1 });
    cutsceneHome = { xforms, aperture: scene.world.aperture, focusDistance: scene.world.focusDistance, atmo: captureAtmo() };
  }

  // Put objects + DoF + atmosphere back to their canonical (pre-cutscene) state.
  function restoreHome() {
    if (!cutsceneHome) return;
    preview.clearPoseOverrides(); // stop driving skeletons from the timeline
    const byId = objectsById();
    for (const h of cutsceneHome.xforms) {
      const obj = byId.get(h.id);
      if (!obj) continue;
      obj.position.set(h.pos[0], h.pos[1], h.pos[2]);
      if (obj instanceof Emitter) continue; // emitters have no rotation/scale
      obj.rotation.set(h.rot[0], h.rot[1], h.rot[2]);
      if (obj instanceof MeshInstance) obj.scale = h.scale;
    }
    scene.world.aperture = cutsceneHome.aperture;
    scene.world.focusDistance = cutsceneHome.focusDistance;
    restoreAtmo(cutsceneHome.atmo);
    scene.touchInstances();
    scene.touchWorld();
  }

  // Pose the camera + DoF + atmosphere + tracked objects at timeline time `t`.
  // Returns the evaluated state so callers (e.g. the offline bake) can reuse it.
  // Does the TIMELINE actually animate this rig's skeleton — i.e. do its keyed
  // poses differ from one key to the next?
  //
  // Cutscenes captured before poses were made conditional (and any captured while
  // a clip was paused) carry an identical skeleton snapshot in every key. Stamping
  // that over a playing clip pins the rig mid-stride for the whole shot. If the
  // poses never change, the timeline isn't posing anything, so a clip may drive.
  // A rig you genuinely keyed pose-by-pose still wins, as it should.
  const skeletonKeyedCache = new Map<number, boolean>();
  let skeletonKeyedStamp = -1;
  function timelineDrivesSkeleton(id: number): boolean {
    if (skeletonKeyedStamp !== cutsceneKeysVersion) {
      skeletonKeyedCache.clear();
      skeletonKeyedStamp = cutsceneKeysVersion;
    }
    const hit = skeletonKeyedCache.get(id);
    if (hit !== undefined) return hit;
    const varies = poseVariesAcrossKeys(cutsceneKeys, id); // cached: it walks every bone of every key
    skeletonKeyedCache.set(id, varies);
    return varies;
  }

  /** The keyed pose to apply for an instance, or null to let its clip play. */
  function effectiveKeyedPose(id: number, pose: BonePose[] | undefined): BonePose[] | null {
    if (!pose) return null;
    if (preview.hasPlayingClip(id) && !timelineDrivesSkeleton(id)) return null;
    return pose;
  }

  function applyCutsceneAt(t: number): CamState | null {
    const s = evalCutscene(cutsceneKeys, t);
    if (!s) return null;
    cam.target.set(s.target[0], s.target[1], s.target[2]);
    cam.distance = s.distance; cam.yaw = s.yaw; cam.pitch = s.pitch;
    cam.update();
    scene.world.aperture = s.aperture;
    scene.world.focusDistance = s.focusDistance;
    // Drive the sky/sun from the interpolated time of day, then layer the
    // explicitly-keyed exposure + haze on top (applyTimeOfDay also sets exposure).
    scene.applyTimeOfDay(s.timeOfDay);
    scene.world.exposure = s.exposure;
    scene.world.hazeDensity = s.haze;
    if (s.objects && s.objects.length) {
      const byId = objectsById();
      preview.clearPoseOverrides(); // rebuilt below for whatever is posed this frame
      for (const o of s.objects) {
        const obj = byId.get(o.id);
        if (!obj) continue;
        obj.position.set(o.pos[0], o.pos[1], o.pos[2]);
        if (obj instanceof Emitter) continue; // emitters have no rotation/scale
        csQuat.set(o.quat[0], o.quat[1], o.quat[2], o.quat[3]);
        obj.rotation.setFromQuaternion(csQuat);
        if (obj instanceof MeshInstance) {
          obj.scale = o.scale;
          const keyed = effectiveKeyedPose(o.id, o.pose);
          if (keyed) preview.setPoseOverride(o.id, keyed); // drive the skeleton
        }
      }
      scene.touchInstances();
    }
    scene.touchWorld();
    return s;
  }

  async function renderCutscene(o: TurntableOptions) {
    if (offlineRendering) return;
    if (cutsceneKeys.length < 2) { hud.textContent = "Cutscene: add at least two keyframes first."; return; }
    if (mode !== "render") { mode = "render"; applyMode(); ui.refresh(); }
    const total = cutsceneDuration(cutsceneKeys);
    const W = Math.max(16, Math.round(o.width));
    const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
    const H = Math.max(16, Math.round(W / aspect));
    const frames = Math.max(2, Math.round(total * o.fps));
    const shutter = 1 / o.fps;

    // The ray tracer renders from the BLAS, not the live skeleton, so posed joints
    // are baked into fresh geometry each frame. We snapshot the original BLAS (and
    // its re-import link, which replaceBlas clears) to restore after the export, so
    // exporting a video never mutates the scene. A dirty-check skips re-baking a
    // rig whose pose is unchanged since its last bake (static segments).
    type BlasSnap = { blas: BLAS; file: (typeof scene.blasFile)[number] };
    const bakeRestore = new Map<number, BlasSnap>();
    const lastPose = new Map<number, BonePose[]>();

    const samePose = (a: BonePose[], b: BonePose[] | undefined): boolean => {
      if (!b || a.length !== b.length) return false;
      for (let k = 0; k < a.length; k++) {
        if (a[k].i !== b[k].i) return false;
        const p = a[k].q, q = b[k].q;
        if (Math.abs(p[0] - q[0]) + Math.abs(p[1] - q[1]) + Math.abs(p[2] - q[2]) + Math.abs(p[3] - q[3]) > 1e-6) return false;
      }
      return true;
    };

    // Re-bake every rigged instance driven by a keyframe pose, a playing clip or
    // spring bones at this frame; returns whether the mesh pools changed (and so
    // need re-uploading).
    const bakePosedFrame = (s: CamState): boolean => {
      if (!s.objects) return false;
      let any = false;
      const byId = objectsById();
      for (const o of s.objects) {
        const obj = byId.get(o.id);
        if (!(obj instanceof MeshInstance)) continue;
        // Clips and springs both evolve every frame, so neither can be dirty-checked
        // against the keyframe pose — they always re-bake.
        const dynamic = preview.hasSprings(o.id) || preview.hasPlayingClip(o.id);
        // A keyed pose only overrides clip playback when the timeline genuinely
        // animates the joints; otherwise the sampled clip is what gets baked.
        const keyed = effectiveKeyedPose(o.id, o.pose);
        if (!keyed && !dynamic) continue;
        if (!dynamic && samePose(keyed!, lastPose.get(o.id))) continue; // static pose → keep last bake
        if (bakeRigFrame(obj, keyed, shutter, bakeRestore)) {
          if (keyed) lastPose.set(o.id, keyed);
          any = true;
        }
      }
      return any;
    };

    preview.resetAllSprings(); // secondary motion starts settled, then accumulates over the shot
    await recordWebM("Cutscene", `aerie-cutscene-${Date.now()}.webm`, W, H, frames, o.fps, (i) => {
      const tSec = (total * i) / (frames - 1);
      const s = applyCutsceneAt(tSec);
      // Put the clips at THIS frame's time before baking. Keyframed joint poses are
      // stamped over the sampled clip inside the bake, so keys still beat playback.
      preview.sampleAnimationAt(tSec);
      // The live loop is paused during export, so push the per-frame state into
      // the renderer ourselves: prims (scene), mesh instance matrices, and DoF.
      const rebaked = s ? bakePosedFrame(s) : false;
      renderer.uploadScene(scene);
      if (rebaked) renderer.uploadMeshPools(scene); // re-uploads geometry (+ instances)
      else renderer.uploadInstances(scene);
      renderer.uploadWorld(scene);
      // Emitters evaluated at this frame's cutscene time, motion-blurred over 1/fps.
      if (scene.emitters.length) uploadSceneParticles(tSec, shutter);
      return renderer.renderToPixels(cam, W, H, o.samples);
    });

    // Put the original (un-posed) geometry and rig links back so the ray view isn't
    // left frozen in the final frame's pose; the live loop re-uploads on the version bump.
    restoreBakedBlas(bakeRestore);
    applyCutsceneAt(cutsceneTime); // return camera + DoF + objects to the playhead
    uploadedParticleTime = NaN; // force a re-freeze on the next live frame
  }

  const cutscene = {
    active: () => cutsceneMode,
    toggle: () => {
      cutsceneMode = !cutsceneMode;
      if (cutsceneMode) {
        captureHome(); // remember canonical object/DoF state to restore on exit
      } else {
        cutscenePlaying = false;
        restoreHome(); // non-destructive: scene returns to its canonical state
        cutsceneHome = null; cutsceneTime = 0;
      }
      ui.refresh();
    },
    count: () => cutsceneKeys.length,
    selected: () => cutsceneSel,
    time: () => cutsceneTime,
    duration: () => cutsceneDuration(cutsceneKeys),
    playing: () => cutscenePlaying,
    keyInfo: (i: number) => ({
      duration: cutsceneKeys[i].duration,
      ease: cutsceneKeys[i].ease,
      bezier: (cutsceneKeys[i].bezier ?? (cutsceneKeys[i].ease === "smooth" ? [0.42, 0, 0.58, 1] : [0, 0, 1, 1])).slice() as [number, number, number, number],
      time: keyTime(cutsceneKeys, i),
    }),
    keyAtmo: (i: number) => ({ timeOfDay: cutsceneKeys[i].timeOfDay, exposure: cutsceneKeys[i].exposure, haze: cutsceneKeys[i].haze }),
    // Edit an atmosphere field on a keyframe and preview it live at the playhead.
    setKeyAtmo: (i: number, field: "timeOfDay" | "exposure" | "haze", v: number) => {
      cutsceneKeys[i][field] = v;
      applyCutsceneAt(cutsceneTime);
      markDirty();
    },
    add: () => { cutsceneKeys.push(captureKey()); cutsceneKeysVersion++; normalizeCutscene(); cutsceneSel = cutsceneKeys.length - 1; cutsceneTime = keyTime(cutsceneKeys, cutsceneSel); markDirty(); ui.refresh(); },
    remove: (i: number) => { cutsceneKeys.splice(i, 1); cutsceneKeysVersion++; normalizeCutscene(); cutsceneSel = Math.min(cutsceneSel, cutsceneKeys.length - 1); markDirty(); ui.refresh(); },
    recapture: (i: number) => { const k = captureKey(); k.duration = cutsceneKeys[i].duration; k.ease = cutsceneKeys[i].ease; k.bezier = cutsceneKeys[i].bezier; cutsceneKeys[i] = k; cutsceneKeysVersion++; normalizeCutscene(); markDirty(); ui.refresh(); },
    select: (i: number) => { cutsceneSel = i; cutsceneTime = keyTime(cutsceneKeys, i); applyCutsceneAt(cutsceneTime); ui.refresh(); },
    setDuration: (i: number, s: number) => { if (i > 0) { cutsceneKeys[i].duration = Math.max(0, s); markDirty(); } },
    setEase: (i: number, e: Ease) => { cutsceneKeys[i].ease = e; markDirty(); ui.refresh(); },
    // Curve editor: set the segment's Bézier handles. No ui.refresh so dragging
    // the handles stays smooth; the panel re-syncs on preset click / drag release.
    setBezier: (i: number, b: [number, number, number, number]) => { cutsceneKeys[i].bezier = b; applyCutsceneAt(cutsceneTime); markDirty(); },
    scrub: (t: number) => { cutscenePlaying = false; cutsceneTime = t; applyCutsceneAt(t); },
    playPause: () => {
      if (cutsceneKeys.length < 2) return;
      if (!cutscenePlaying && cutsceneTime >= cutsceneDuration(cutsceneKeys)) cutsceneTime = 0;
      cutscenePlaying = !cutscenePlaying;
      ui.refresh();
    },
    render: (o: TurntableOptions) => renderCutscene(o),
  };

  // Cutscene edits mark the scene dirty (it serializes into .aerie). Bump only
  // when currently clean — avoids re-uploading the scene on every slider tick.
  function markDirty() { if (scene.version === savedVersion) scene.touch(); }

  const ui = buildUI(scene, () => cam.target.clone(), {
    onImport: doImport,
    onToggleMode: () => {
      mode = mode === "render" ? "preview" : "render";
      applyMode();
    },
    isPreview: () => mode === "preview",
    onBakePose: bakeSelected,
    onPrimImage,
    onRenderImage,
    onToggleEdit: toggleEdit,
    onEditPrim: editPrim,
    onMeshTexture,
    onBakeBoolean: bakeBoolean,
    onAddLandscape: addLandscape,
    onScatterEcosystem: scatterEcosystem,
    onRegenerateLandform: regenerateLandform,
    onRerollLandform: rerollLandform,
    onAddPlanet: addPlanet,
    onNewScene: newScene,
    onOpenGallery: openGallery,
    onSaveToGallery: saveToGallery,
    onImportToGallery: importToGallery,
    onSaveScene: saveScene,
    onLoadScene: loadScene,
    isEditing: () => editInstance !== null,
    isGhostCarvers: () => showCarverGhosts,
    onToggleGhostCarvers: () => {
      showCarverGhosts = !showCarverGhosts;
    },
    isFocusPick: () => focusPick,
    onToggleFocusPick: () => {
      focusPick = !focusPick;
      hud.textContent = focusPick ? "Click a point in the view to focus on…" : "";
    },
    onPickAt: pickObjectAt,
    onStatus: (msg) => { hud.textContent = msg; },
    onRenderTurntable: renderTurntable,
    cutscene,
    boneNames: (inst) => preview.getBones(inst.id)?.map((b, i) => b.name || `bone ${i}`) ?? [],
    isPosing: () => poseInst !== null,
    onTogglePose: togglePose,
    isIK: () => ikEnabled,
    onToggleIK: () => { ikEnabled = !ikEnabled; },
    animateInRender: () => animateInRender,
    onToggleAnimateInRender: () => { animateInRender = !animateInRender; renderer.resetAccumulation(); },
    selectedBoneCount: () => boneSel.size,
    onClearBoneSelection: () => { boneSel.clear(); if (selectedBone >= 0) boneSel.add(selectedBone); },
    hiddenBoneCount: () => (poseShowAll ? 0 : poseDeforms.reduce((n, d) => n + (d ? 0 : 1), 0)),
    showAllBones: () => poseShowAll,
    onToggleShowAllBones: () => { poseShowAll = !poseShowAll; },
    isSpringBone: (i) => (poseInst ? preview.isSpringBone(poseInst.id, i) : false),
    toggleSpringBone: (i) => { if (poseInst) preview.setSpringBone(poseInst.id, i, !preview.isSpringBone(poseInst.id, i)); },
    hasSprings: () => (poseInst ? preview.hasSprings(poseInst.id) : false),
    springParams: () => (poseInst ? preview.springParams(poseInst.id) : { stiffness: 0.2, damping: 0.2, gravity: 0.5 }),
    setSpringParam: (key, v) => { if (poseInst) preview.setSpringParam(poseInst.id, key, v); },
    clearSprings: () => { if (poseInst) preview.clearSprings(poseInst.id); },
    clipNames: (inst) => preview.clipNames(inst.id),
    onImportAnimation: async (inst, file) => {
      hud.textContent = `Loading animation ${file.name}…`;
      try {
        const clips = await importAnimations(file);
        const n = preview.addClips(inst.id, clips);
        hud.textContent = n
          ? `Added ${n} clip${n > 1 ? "s" : ""}. Open the Pose tab's clip list to scrub or blend them.`
          : "No animation clips found in that file.";
        ui.refresh();
      } catch (e) {
        hud.textContent = `Animation import failed: ${e instanceof Error ? e.message : e}`;
      }
    },
    clipWeight: (inst, i) => preview.clipWeight(inst.id, i),
    setClipWeight: (inst, i, w) => preview.setClipWeight(inst.id, i, w),
    clipAdditive: (inst, i) => preview.clipAdditive(inst.id, i),
    toggleClipAdditive: (inst, i) => preview.setClipAdditive(inst.id, i, !preview.clipAdditive(inst.id, i)),
    poseClipIndex: () => (poseInst ? preview.activeClipIndex(poseInst.id) : -1),
    poseClipDuration: () => (poseInst ? preview.activeClipDuration(poseInst.id) : 0),
    poseClipTime: () => (poseInst ? preview.clipTime(poseInst.id) : 0),
    onSelectClip: (i) => { if (poseInst) preview.setActiveClip(poseInst.id, i); },
    onScrubClip: (t) => { if (poseInst) preview.scrubClip(poseInst.id, t); },
    onSnapshotClip: () => { if (poseInst) preview.snapshotClipPose(poseInst.id); },
    onPoseUndo: poseUndoStep,
    onPoseRedo: poseRedoStep,
    canPoseUndo: () => poseUndo.length > 0,
    canPoseRedo: () => poseRedo.length > 0,
    onCopyPose: copyPose,
    onPastePose: pastePose,
    canPastePose: () => !!poseInst && !!poseClipboard && poseClipboard.length === poseBones.length,
    onResetAllJoints: resetAllJoints,
    onSavePose: savePose,
    poseLibraryNames: () => poseLibrary.map((p) => p.name),
    onApplyLibraryPose: applyLibraryPose,
    poseSelectedBone: () => selectedBone,
    onSelectBone: (i) => { selectedBone = i; boneSel.clear(); },
    boneRotation: (i) => {
      const b = poseBones[i];
      return b ? [b.rotation.x, b.rotation.y, b.rotation.z] : [0, 0, 0];
    },
    setBoneRotation: (i, axis, v) => {
      const b = poseBones[i];
      if (!b) return;
      if (axis === 0) b.rotation.x = v;
      else if (axis === 1) b.rotation.y = v;
      else b.rotation.z = v;
    },
    resetBone: (i) => {
      const b = poseBones[i];
      const o = poseOrig[i];
      if (b && o) b.rotation.set(o[0], o[1], o[2]);
    },
    poseRootIndex: () => poseRoot,
    rootPosition: () => {
      const b = poseBones[poseRoot];
      return b ? [b.position.x, b.position.y, b.position.z] : [0, 0, 0];
    },
    setRootPosition: (axis, v) => {
      const b = poseBones[poseRoot];
      if (!b) return;
      if (axis === 0) b.position.x = v;
      else if (axis === 1) b.position.y = v;
      else b.position.z = v;
    },
    resetRoot: () => {
      const b = poseBones[poseRoot];
      if (b) b.position.copy(poseRootOrig);
    },
    onDuplicate: duplicateSelection,
    onAlign: alignSelection,
    onDistribute: distributeSelection,
    onScatter: scatterSelection,
    onMirror: mirrorSelection,
    getBookmarks: () => camBookmarks.map((b) => b !== null),
    onSaveBookmark: saveBookmark,
    onRecallBookmark: recallBookmark,
    getSnap: () => ({ enabled: snapEnabled, grid: snapGrid }),
    onToggleSnap: () => { snapEnabled = !snapEnabled; },
    onSetGrid: (v) => { snapGrid = Math.max(0.05, v); },
    getTool: () => tool,
    onSetTool: (t) => { tool = t; ui.refresh(); },
    getGizmoMode: () => gizmoMode,
    onSetGizmoMode: (m) => { gizmoMode = m; ui.refresh(); },
    onCommit: () => history.commit(),
    onEdit: () => history.commitSoon(),
  });
  history = new History(
    scene,
    () => ui.getSelection(),
    (items) => ui.setSelection(items),
    () => { ui.refresh(); renderer.resetAccumulation(); },
  );
  applyMode();

  // Drag-and-drop import anywhere on the window.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (!f) return;
    if (/\.(aerie|csbryce|json)$/i.test(f.name)) loadScene(f);
    else doImport(f);
  });

  let uploadedVersion = -1;
  let uploadedMeshStruct = -1;
  let uploadedInst = -1;
  let uploadedMeshMat = -1;
  let uploadedWorld = -1;
  let uploadedPrimTex = -1;

  // ---- particle field (emitters) ----
  // The stateless model is evaluated on the CPU into this scratch buffer and
  // uploaded to the tracer. In live Render mode particles are FROZEN at
  // `particleTime` (0, or the cutscene playhead) so accumulation converges;
  // scrubbing/exporting re-evaluates at each new time. See gen/particles.ts.
  const particleData = new Float32Array(MAX_PARTICLES * PARTICLE_FLOATS);
  let uploadedParticleVersion = -2; // scene.version the particle field was built for
  let uploadedParticleTime = NaN;   // scene-time the particle field was built for

  /** Evaluate every emitter at time `t` into `particleData`; returns total count. */
  function evaluateSceneParticles(t: number): number {
    let count = 0;
    for (const e of scene.emitters) {
      if (count >= MAX_PARTICLES) break;
      count += evaluateEmitter(e, t, particleData, count * PARTICLE_FLOATS);
    }
    return count;
  }

  const particleBound = new Float32Array(4); // cloud bounding sphere for the shader reject

  /** Evaluate all emitters at `t`, compute a bounding sphere over the live
   *  particles (so the tracer can skip the per-particle loop for rays that miss
   *  the whole cloud), and upload. `shutter` drives motion blur (1/fps on export). */
  function uploadSceneParticles(t: number, shutter: number): void {
    const count = scene.emitters.length > 0 ? evaluateSceneParticles(t) : 0;
    let minX = 1e30, minY = 1e30, minZ = 1e30, maxX = -1e30, maxY = -1e30, maxZ = -1e30, maxR = 0;
    for (let i = 0; i < count; i++) {
      const o = i * PARTICLE_FLOATS;
      if (particleData[o + 7] <= 0.003) continue; // dead slot
      const x = particleData[o], y = particleData[o + 1], z = particleData[o + 2];
      minX = Math.min(minX, x); minY = Math.min(minY, y); minZ = Math.min(minZ, z);
      maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); maxZ = Math.max(maxZ, z);
      maxR = Math.max(maxR, particleData[o + 3]);
    }
    if (count > 0 && maxX >= minX) {
      const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2, cz = (minZ + maxZ) / 2;
      particleBound[0] = cx; particleBound[1] = cy; particleBound[2] = cz;
      particleBound[3] = Math.hypot(maxX - cx, maxY - cy, maxZ - cz) + maxR + 0.5;
    } else {
      particleBound.fill(0);
    }
    renderer.uploadParticles(particleData, count, shutter, particleBound);
  }

  /** The frozen scene-time particles are shown at in live Render mode: the
   *  cutscene playhead while the cutscene is open, otherwise a still t=0. */
  const liveParticleTime = (): number => (cutsceneMode ? cutsceneTime : 0);

  // ---- sizing (cap DPR so the path tracer stays interactive) ----
  const MAX_SAMPLES = 512;
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const w = Math.floor(canvas.clientWidth * dpr);
    const h = Math.floor(canvas.clientHeight * dpr);
    preview.resize(canvas.clientWidth, canvas.clientHeight);
    overlay.width = canvas.clientWidth;
    overlay.height = canvas.clientHeight;
    if (w === renderer.width && h === renderer.height) return;
    canvas.width = w;
    canvas.height = h;
    renderer.resize(w, h);
  }
  new ResizeObserver(resize).observe(canvas);
  resize();

  // ---- mouse: click-select, drag-move, orbit, look, pan, zoom ----
  // Left-drag empty space orbits; left-drag an object moves it (Shift = up/down).
  // Right-drag looks around in place (no camera move). Middle-drag pans the
  // camera across the horizontal plane. Wheel zooms.
  type Drag = "none" | "pending" | "orbit" | "object" | "look" | "pan" | "axis" | "vert" | "bone" | "ik" | "multi" | "gzbone" | "marquee";
  let drag: Drag = "none";
  let lastX = 0;
  let lastY = 0;
  let dragTarget: Primitive | MeshInstance | Light | Emitter | null = null;
  // Box-select rectangle (CSS px, same space as projectToScreen).
  let marqStartX = 0, marqStartY = 0, marqCurX = 0, marqCurY = 0;
  // Group transform: snapshot of each selected object's full transform at drag
  // start, so rotate/scale can orbit/scale the whole selection about its pivot.
  type Movable = Primitive | MeshInstance | Light | Emitter;
  interface TSnap { pos: Vector3; rot: Euler | null; dir: Vector3 | null; scale: number; a: number; b: number; c: number; }
  const xformSnap = new Map<Movable, TSnap>();
  const objDragHit0 = new Vector3();
  const groupDelta = new Vector3();
  const gizmoPivot = new Vector3();
  let activeGizmoMode: "translate" | "rotate" | "scale" = "translate";
  let gizmoDownX = 0, gizmoDownY = 0;
  const rotQ = new Quaternion();
  const tmpQ = new Quaternion();
  const tmpPivot = new Vector3();
  let movedDuringDrag = false; // an actual transform happened → worth an undo step
  let draggingDirLight = false; // re-aiming a directional light's handle
  let dragVertical = false;
  const planePoint = new Vector3();
  const planeNormal = new Vector3();
  const scratch = new Vector3();
  // Axis-constrained gizmo drag.
  const WORLD_AXES = [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)];
  const dragAxis = new Vector3();
  const axisStartPos = new Vector3();
  let axisStartS = 0;
  let activeAxisIdx = -1;

  function screenRay(cx: number, cy: number): { ro: Vector3; rd: Vector3 } {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const ndcx = (cx / w) * 2 - 1;
    const ndcy = 1 - (cy / h) * 2;
    const rd = new Vector3()
      .copy(cam.forward)
      .addScaledVector(cam.right, ndcx * (w / h) * cam.fovScale)
      .addScaledVector(cam.up, ndcy * cam.fovScale)
      .normalize();
    return { ro: cam.position, rd };
  }
  function rayPlane(ro: Vector3, rd: Vector3, p0: Vector3, n: Vector3): Vector3 | null {
    const denom = rd.dot(n);
    if (Math.abs(denom) < 1e-6) return null;
    const t = scratch.copy(p0).sub(ro).dot(n) / denom;
    return t < 0 ? null : new Vector3().copy(ro).addScaledVector(rd, t);
  }

  // ---- group transform (single object or multi-selection) ----
  function snapOf(it: Movable): TSnap {
    return {
      pos: it.position.clone(),
      rot: (it instanceof Light || it instanceof Emitter) ? null : it.rotation.clone(),
      dir: it instanceof Light ? it.direction.clone() : null,
      scale: it instanceof MeshInstance ? it.scale : 1,
      a: it instanceof Primitive ? it.a : 0,
      b: it instanceof Primitive ? it.b : 0,
      c: it instanceof Primitive ? it.c : 0,
    };
  }
  /** Snapshot the transform of every item to move with `target` (the group if it
   *  belongs to a multi-selection, otherwise just itself). */
  function beginGroupSnapshot(target: Movable): void {
    xformSnap.clear();
    const sel = ui.getSelection() as Movable[];
    const group = sel.includes(target) && sel.length > 1 ? sel : [target];
    for (const it of group) xformSnap.set(it, snapOf(it));
  }
  /** Snapshot the whole current selection (for pivot rotate/scale). */
  function beginSelectionSnapshot(): void {
    xformSnap.clear();
    for (const it of ui.getSelection() as Movable[]) xformSnap.set(it, snapOf(it));
  }
  function touchAfterXform(): void {
    let inst = false, other = false;
    for (const it of xformSnap.keys()) {
      if (it instanceof MeshInstance) inst = true;
      else other = true;
    }
    if (inst) scene.touchInstances();
    if (other) scene.touch();
    movedDuringDrag = true;
  }
  function applyTranslate(): void {
    if (snapEnabled) { // quantize the movement to grid increments (keeps formation)
      groupDelta.x = Math.round(groupDelta.x / snapGrid) * snapGrid;
      groupDelta.y = Math.round(groupDelta.y / snapGrid) * snapGrid;
      groupDelta.z = Math.round(groupDelta.z / snapGrid) * snapGrid;
    }
    for (const [it, s] of xformSnap) it.position.copy(s.pos).add(groupDelta);
    touchAfterXform();
  }
  function applyRotate(angle: number): void {
    if (snapEnabled) { const step = Math.PI / 12; angle = Math.round(angle / step) * step; } // 15°
    rotQ.setFromAxisAngle(WORLD_AXES[activeAxisIdx], angle);
    for (const [it, s] of xformSnap) {
      it.position.copy(s.pos).sub(gizmoPivot).applyQuaternion(rotQ).add(gizmoPivot);
      if (s.rot && !(it instanceof Light) && !(it instanceof Emitter)) { tmpQ.setFromEuler(s.rot).premultiply(rotQ); it.rotation.setFromQuaternion(tmpQ); }
      if (s.dir && it instanceof Light) it.direction.copy(s.dir).applyQuaternion(rotQ);
    }
    touchAfterXform();
  }
  function applyScale(factor: number): void {
    if (snapEnabled) factor = Math.round(factor / 0.25) * 0.25; // 0.25 steps
    const f = Math.max(0.02, factor);
    for (const [it, s] of xformSnap) {
      it.position.copy(s.pos).sub(gizmoPivot).multiplyScalar(f).add(gizmoPivot);
      if (it instanceof MeshInstance) it.scale = Math.max(0.02, s.scale * f);
      else if (it instanceof Primitive) { it.a = s.a * f; it.b = s.b * f; it.c = s.c * f; }
    }
    touchAfterXform();
  }
  // The gizmo shows for anything with a meaningful position; a lone directional
  // light keeps its dedicated handle re-aim (drag the sun marker) instead.
  function gizmoActive(): boolean {
    return ui.getSelection().some(
      (it) => it instanceof Primitive || it instanceof MeshInstance || it instanceof Emitter ||
        (it instanceof Light && it.type === LightType.Point),
    );
  }
  // Pivot of the current selection (average of gizmo centres) and a handle radius.
  function selectionPivot(out: Vector3): Vector3 {
    const sel = ui.getSelection();
    out.set(0, 0, 0);
    if (!sel.length) return out;
    for (const it of sel) out.add(gizmoFrame(it).center);
    return out.multiplyScalar(1 / sel.length);
  }
  function selectionRadius(pivot: Vector3): number {
    let r = 6;
    for (const it of ui.getSelection()) {
      const f = gizmoFrame(it);
      r = Math.max(r, f.center.distanceTo(pivot) + f.radius * 0.6);
    }
    return r;
  }

  // ---- box (marquee) selection ----
  const inMarquee = (p: Vector3, x0: number, y0: number, x1: number, y1: number): boolean => {
    const s = projectToScreen(p);
    return !!s && s.x >= x0 && s.x <= x1 && s.y >= y0 && s.y <= y1;
  };
  function resolveMarquee(e: PointerEvent): void {
    const x0 = Math.min(marqStartX, marqCurX), x1 = Math.max(marqStartX, marqCurX);
    const y0 = Math.min(marqStartY, marqCurY), y1 = Math.max(marqStartY, marqCurY);
    const isClick = x1 - x0 < 4 && y1 - y0 < 4;
    const additive = e.shiftKey || e.ctrlKey;
    if (poseInst) {
      if (!additive) boneSel.clear();
      let last = -1;
      for (let i = 0; i < poseBones.length; i++) {
        if (!boneOffered(i)) continue;
        if (!inMarquee(poseBones[i].getWorldPosition(boneWorld), x0, y0, x1, y1)) continue;
        boneSel.add(i);
        last = i;
      }
      if (last >= 0) selectedBone = last;
      else if (!additive && selectedBone >= 0) boneSel.add(selectedBone); // never empty
      ui.refresh();
      return;
    }
    if (editInstance && editMesh) {
      if (isClick) clickSelectVertex(e, additive);
      else boxSelectVerts(x0, y0, x1, y1, additive);
    } else {
      if (isClick) clickSelectObject(e, additive);
      else boxSelectObjects(x0, y0, x1, y1, additive);
    }
  }
  function boxSelectObjects(x0: number, y0: number, x1: number, y1: number, additive: boolean): void {
    const hits: Movable[] = [];
    for (const p of scene.prims) if (inMarquee(p.position, x0, y0, x1, y1)) hits.push(p);
    for (const m of scene.instances) if (inMarquee(m.position, x0, y0, x1, y1)) hits.push(m);
    for (const l of scene.lights) if (inMarquee(lightGizmoPos(l, tmpV), x0, y0, x1, y1)) hits.push(l);
    for (const em of scene.emitters) if (inMarquee(tmpV.copy(em.position), x0, y0, x1, y1)) hits.push(em);
    if (additive) {
      const cur = ui.getSelection();
      for (const h of hits) if (!cur.includes(h)) cur.push(h);
      ui.setSelection(cur);
    } else {
      ui.setSelection(hits);
    }
  }
  function applyClickSelection(target: Movable | null, additive: boolean): void {
    if (!target) { if (!additive) ui.select(null); return; }
    if (additive) {
      const cur = ui.getSelection();
      const i = cur.indexOf(target);
      if (i >= 0) cur.splice(i, 1);
      else cur.push(target);
      ui.setSelection(cur);
    } else {
      ui.select(target);
    }
  }
  function clickSelectObject(e: PointerEvent, additive: boolean): void {
    const light = pickLight(e.clientX, e.clientY);
    if (light) { applyClickSelection(light, additive); return; }
    const emitter = pickEmitter(e.clientX, e.clientY);
    if (emitter) { applyClickSelection(emitter, additive); return; }
    const dpr = renderer.width / Math.max(1, canvas.clientWidth);
    renderer.pick(cam, e.clientX * dpr, e.clientY * dpr).then((r) => {
      let target: Movable | null = null;
      if (r.kind === 4 && r.index >= 0 && r.index < scene.prims.length) target = scene.prims[r.index];
      else if (r.kind === 5 && r.index >= 0 && r.index < scene.instances.length) target = scene.instances[r.index];
      applyClickSelection(target, additive);
    });
  }
  function boxSelectVerts(x0: number, y0: number, x1: number, y1: number, additive: boolean): void {
    updateEditMatrices();
    if (!additive) selectedVerts.clear();
    const n = editMesh!.verts.length / 3;
    for (let i = 0; i < n; i++) {
      if (inMarquee(vertWorld(i, vTmp), x0, y0, x1, y1)) selectedVerts.add(i);
    }
  }
  function clickSelectVertex(e: PointerEvent, additive: boolean): void {
    updateEditMatrices();
    const vIdx = pickVertex(e.clientX, e.clientY);
    if (vIdx < 0) { if (!additive) selectedVerts.clear(); return; }
    if (additive) {
      if (selectedVerts.has(vIdx)) selectedVerts.delete(vIdx);
      else selectedVerts.add(vIdx);
    } else {
      selectedVerts.clear();
      selectedVerts.add(vIdx);
    }
  }

  for (const c of [canvas, previewCanvas]) {
    c.addEventListener("contextmenu", (e) => e.preventDefault()); // allow right-drag

    c.addEventListener("pointerdown", (e) => {
      cam.update();
      lastX = e.clientX;
      lastY = e.clientY;
      c.setPointerCapture(e.pointerId);
      if (e.button === 1) { drag = "look"; e.preventDefault(); return; } // middle
      if (e.button === 2) { drag = "pan"; return; } // right
      if (e.button === 0 && e.altKey) { drag = "orbit"; return; } // Alt+left always orbits

      // Click-to-focus (armed from the Render lab): the next left click sets the
      // depth-of-field focal distance to whatever surface is under the cursor.
      if (focusPick && e.button === 0) {
        drag = "none";
        const dpr = renderer.width / Math.max(1, canvas.clientWidth);
        renderer.pick(cam, e.clientX * dpr, e.clientY * dpr).then((r) => {
          if (r.kind !== 0 && Number.isFinite(r.dist) && r.dist > 0) {
            scene.world.focusDistance = r.dist;
            scene.touchWorld();
            hud.textContent = `Focus set: ${r.dist.toFixed(1)} units`;
          } else {
            hud.textContent = "Focus: nothing under the cursor";
          }
          focusPick = false;
          ui.refresh();
        });
        return;
      }

      if (editInstance) { editPointerDown(e); return; } // vertex editing
      if (poseInst) {
        // FK: grab a rotation ring of the selected bone before falling to joints.
        if (!ikEnabled && selectedBone >= 0) {
          const axis = pickBoneRing(e.clientX, e.clientY, selectedBone);
          if (axis >= 0) { poseDragSnap = capturePose(); beginBoneGizmo(axis, e); return; }
        }
        const bi = pickBone(e.clientX, e.clientY);
        if (bi >= 0) {
          // Ctrl-click builds the selection instead of starting a drag.
          if (e.ctrlKey || e.metaKey) {
            if (boneSel.has(bi) && bi !== selectedBone) boneSel.delete(bi);
            else { boneSel.add(bi); selectedBone = bi; }
            boneSel.add(selectedBone); // the active joint is always part of it
            ui.refresh();
            return;
          }
          // Grabbing a joint that's in a multi-selection drags the whole set;
          // grabbing anything else collapses the selection onto it.
          if (!boneSel.has(bi)) { boneSel.clear(); boneSel.add(bi); }
          selectedBone = bi;
          ui.refresh();
          if (boneSel.size > 1 && ikEnabled && beginMultiDrag(e)) return;
          const chain = ikEnabled ? ikChain(bi, e.shiftKey) : [];
          if (chain.length >= 2) {
            // Grab the effector: solve the chain as the cursor drags it, on a
            // camera-facing plane at the effector's depth.
            poseDragSnap = capturePose(); // pre-edit pose → one undo step on release
            const tip = chain[chain.length - 1];
            tip.getWorldPosition(ikPlanePoint);
            if (chain.length === 3) {
              // Exact two-bone solve for a shoulder/elbow/wrist-style limb.
              ikChainArr = null;
              ikRoot = chain[0]; ikMid = chain[1]; ikTip = chain[2];
              autoPole(ikRoot, ikMid, ikTip, ikPole);
            } else {
              // CCD for any other chain length — tails, tentacles, spines.
              ikChainArr = chain;
              ikRoot = ikMid = ikTip = null;
            }
            drag = "ik";
          } else {
            drag = "bone"; // no usable chain, or IK off → FK arcball rotate
          }
          return;
        }
        // Empty space: rubber-band over joints with the Box-Select tool, else orbit.
        if (tool === "select") {
          marqStartX = marqCurX = e.clientX;
          marqStartY = marqCurY = e.clientY;
          drag = "marquee";
          return;
        }
        if (!(e.ctrlKey || e.metaKey)) { boneSel.clear(); boneSel.add(selectedBone); }
        drag = "orbit"; // empty space → orbit the camera
        return;
      }

      // Grab a gizmo handle on the current selection? The gizmo sits at the
      // selection pivot and obeys the active mode (translate/rotate/scale).
      if (gizmoActive()) {
        const pivot = selectionPivot(tmpPivot);
        const radius = Math.max(8, selectionRadius(pivot));
        const idx = pickAxisAt(pivot, radius, e.clientX, e.clientY);
        if (idx >= 0) {
          activeAxisIdx = idx;
          activeGizmoMode = gizmoMode;
          dragAxis.copy(WORLD_AXES[idx]);
          gizmoPivot.copy(pivot);
          gizmoDownX = e.clientX; gizmoDownY = e.clientY;
          beginSelectionSnapshot();
          if (gizmoMode === "translate") {
            axisStartPos.copy(pivot);
            const ray = screenRay(e.clientX, e.clientY);
            axisStartS = closestAxisParam(ray.ro, ray.rd, axisStartPos, dragAxis) ?? 0;
          }
          drag = "axis";
          return;
        }
      }

      // Box-Select tool: left-drag rubber-bands a selection (resolved on
      // pointerup; a click with no drag falls back to a single pick).
      if (tool === "select") {
        marqStartX = marqCurX = e.clientX;
        marqStartY = marqCurY = e.clientY;
        drag = "marquee";
        return;
      }

      // --- Move tool: click-select + drag-move ---
      // Lights and emitters aren't in the GPU pick buffer — test their viewport
      // handles first (nearest handle wins if they overlap).
      const light = pickLight(e.clientX, e.clientY);
      const emitter = pickEmitter(e.clientX, e.clientY);
      if (light || emitter) {
        // Prefer whichever handle is closer to the cursor.
        const lp = light ? projectToScreen(lightGizmoPos(light, tmpV)) : null;
        const ep = emitter ? projectToScreen(tmpV.copy(emitter.position)) : null;
        const ld = lp ? Math.hypot(e.clientX - lp.x, e.clientY - lp.y) : Infinity;
        const ed = ep ? Math.hypot(e.clientX - ep.x, e.clientY - ep.y) : Infinity;
        if (emitter && ed <= ld) { ui.select(emitter); startEmitterDrag(emitter, e); }
        else { ui.select(light!); startLightDrag(light!, e); }
        return;
      }

      drag = "pending";
      const dpr = renderer.width / Math.max(1, canvas.clientWidth);
      renderer.pick(cam, e.clientX * dpr, e.clientY * dpr).then((r) => {
        if (drag !== "pending") return;
        let target: Primitive | MeshInstance | null = null;
        if (r.kind === 4 && r.index >= 0 && r.index < scene.prims.length) {
          target = scene.prims[r.index];
        } else if (r.kind === 5 && r.index >= 0 && r.index < scene.instances.length) {
          target = scene.instances[r.index];
        }
        if (target) {
          // Dragging an object that's already part of a multi-selection moves the
          // whole group; otherwise it becomes the sole selection.
          const selList = ui.getSelection();
          if (!(selList.includes(target) && selList.length > 1)) ui.select(target);
          dragTarget = target;
          dragVertical = e.shiftKey;
          planePoint.copy(target.position);
          if (dragVertical) planeNormal.copy(cam.forward).setY(0).normalize();
          else planeNormal.set(0, 1, 0);
          const ray = screenRay(lastX, lastY);
          const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
          if (hit) objDragHit0.copy(hit);
          beginGroupSnapshot(target);
          drag = "object";
        } else {
          ui.select(null); // clicked the world → deselect
          drag = "orbit";
        }
      });
    });
    c.addEventListener("pointerup", (e) => {
      if (drag === "object" || drag === "axis" || drag === "bone" || drag === "ik" || drag === "multi" || drag === "gzbone") ui.refresh(); // sync inspector fields
      if (drag === "ik") { ikRoot = ikMid = ikTip = null; ikChainArr = null; }
      if (drag === "multi") multiGrab = [];
      if (drag === "gzbone") { gzAxis = -1; }
      // Commit a joint edit as one pose-undo step (only if it actually moved).
      if ((drag === "ik" || drag === "multi" || drag === "gzbone" || drag === "bone") && poseDragSnap) {
        if (posesDiffer(poseDragSnap, capturePose())) pushPoseUndo(poseDragSnap);
        poseDragSnap = null;
      }
      if ((drag === "object" || drag === "axis") && movedDuringDrag) history.commit();
      if (drag === "vert" && vertDragMode !== "none") commitEdit(); // rebuild BLAS
      if (drag === "marquee") resolveMarquee(e);
      drag = "none";
      dragTarget = null;
      draggingDirLight = false;
      activeAxisIdx = -1;
      xformSnap.clear();
      movedDuringDrag = false;
      vertDragMode = "none";
    });
    c.addEventListener("pointermove", (e) => {
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      if (drag === "gzbone") {
        dragBoneGizmo(e);
        lastX = e.clientX;
        lastY = e.clientY;
      } else if (drag === "ik") {
        // Solve the grabbed chain so its tip chases the cursor across a camera-
        // facing plane at the tip's depth. Exact two-bone when we have a limb;
        // CCD for longer chains (tails/tentacles). The pole holds the elbow bend.
        const ray = screenRay(e.clientX, e.clientY);
        const hit = rayPlane(ray.ro, ray.rd, ikPlanePoint, cam.forward);
        if (hit) {
          if (ikRoot && ikMid && ikTip) solveTwoBoneIK(ikRoot, ikMid, ikTip, hit, { pole: ikPole });
          else if (ikChainArr) solveCCD(ikChainArr, hit, { iterations: 12, damping: 0.5 });
        }
        lastX = e.clientX;
        lastY = e.clientY;
      } else if (drag === "multi") {
        dragMulti(e);
        lastX = e.clientX;
        lastY = e.clientY;
      } else if (drag === "bone" && poseInst && selectedBone >= 0) {
        // Arcball: rotate the selected bone around the camera's screen axes.
        const b = poseBones[selectedBone];
        const k = 0.012; // radians per pixel
        if (b.parent) b.parent.getWorldQuaternion(qParent);
        else qParent.identity();
        qWorld.copy(qParent).multiply(b.quaternion); // current world orientation
        qDeltaA.setFromAxisAngle(cam.up, dx * k);
        qDeltaB.setFromAxisAngle(cam.right, dy * k);
        qDeltaA.multiply(qDeltaB);
        qWorld.premultiply(qDeltaA);
        b.quaternion.copy(qParent.invert()).multiply(qWorld).normalize(); // back to local
        lastX = e.clientX;
        lastY = e.clientY;
      } else if (drag === "orbit") {
        cam.orbit(dx, dy);
        lastX = e.clientX;
        lastY = e.clientY;
        renderer.resetAccumulation();
      } else if (drag === "look") {
        cam.look(dx, dy);
        lastX = e.clientX;
        lastY = e.clientY;
        renderer.resetAccumulation();
      } else if (drag === "pan") {
        const k = cam.distance * 0.0015;
        cam.moveRelative(-dy * k, dx * k, 0); // truck across the horizontal plane
        lastX = e.clientX;
        lastY = e.clientY;
        renderer.resetAccumulation();
      } else if (drag === "marquee") {
        marqCurX = e.clientX;
        marqCurY = e.clientY;
      } else if (drag === "axis") {
        if (activeGizmoMode === "rotate") {
          applyRotate((e.clientX - gizmoDownX) * 0.01); // dx → radians about the axis
        } else if (activeGizmoMode === "scale") {
          applyScale(Math.exp(-(e.clientY - gizmoDownY) * 0.01)); // drag up = larger
        } else {
          const ray = screenRay(e.clientX, e.clientY);
          const s = closestAxisParam(ray.ro, ray.rd, axisStartPos, dragAxis);
          if (s !== null) { groupDelta.copy(dragAxis).multiplyScalar(s - axisStartS); applyTranslate(); }
        }
      } else if (drag === "vert") {
        const ray = screenRay(e.clientX, e.clientY);
        if (vertDragMode === "axis") {
          const s = closestAxisParam(ray.ro, ray.rd, axisStartPos, dragAxis);
          if (s !== null) applyWorldDelta(vTmp.copy(dragAxis).multiplyScalar(s - axisStartS));
        } else if (vertDragMode === "plane") {
          const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
          if (hit) applyWorldDelta(hit.sub(editPlaneStart));
        }
      } else if (draggingDirLight && dragTarget instanceof Light) {
        const ray = screenRay(e.clientX, e.clientY);
        const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
        if (hit) {
          const d = hit.sub(cam.target);
          d.y = Math.max(d.y, 2); // keep the sun above the horizon
          dragTarget.direction.copy(d).normalize();
          scene.touch();
        }
      } else if (drag === "object" && dragTarget) {
        const ray = screenRay(e.clientX, e.clientY);
        const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
        if (hit) {
          groupDelta.copy(hit).sub(objDragHit0);
          if (dragVertical) { groupDelta.x = 0; groupDelta.z = 0; } // Shift = vertical only
          else groupDelta.y = 0;
          applyTranslate();
        }
      } else if (poseInst && drag === "none") {
        // Idle in pose mode: track the joint under the cursor so the overlay can
        // show what a drag from here would move, before you commit to it.
        hoverBone = pickBone(e.clientX, e.clientY);
        hoverExtend = e.shiftKey;
        canvas.style.cursor = hoverBone >= 0 ? "grab" : "";
      }
    });
    c.addEventListener("pointerleave", () => { hoverBone = -1; });
    // Shift extends the chain — reflect it in the hover preview even if the
    // cursor is holding still.
    const trackShift = (e: KeyboardEvent) => { if (poseInst && drag === "none") hoverExtend = e.shiftKey; };
    window.addEventListener("keydown", trackShift);
    window.addEventListener("keyup", trackShift);
    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        cam.dolly(e.deltaY);
        renderer.resetAccumulation();
      },
      { passive: false },
    );
  }

  // ---- selection ergonomics (Delete / Esc / Ctrl+A / F) ----
  function deleteSelection(): void {
    const sel = ui.getSelection();
    if (!sel.length) return;
    for (const it of sel) scene.remove(it);
    const fb = scene.prims[0] ?? scene.instances[0] ?? scene.lights[0] ?? null;
    ui.setSelection(fb ? [fb] : []); // refreshes the UI
    history.commit();
  }
  function duplicateSelection(): void {
    if (editInstance || poseInst) return;
    const sel = ui.getSelection();
    if (!sel.length) return;
    const copies: Movable[] = [];
    for (const it of sel) {
      const dup = scene.addDuplicate(it) as Movable;
      dup.position.x += 4; dup.position.z += 4; // offset so the copy is visible
      if (dup instanceof MeshInstance) preview.addInstance(dup.id, previewFromBlas(scene, dup.blasIndex), []);
      copies.push(dup);
    }
    scene.touchInstances();
    ui.setSelection(copies);
    history.commit();
    ui.refresh();
  }
  // Align selected objects' centres along one axis (0=X,1=Y,2=Z).
  function alignSelection(axis: 0 | 1 | 2, mode: "min" | "center" | "max"): void {
    const sel = ui.getSelection();
    if (sel.length < 2) return;
    const vals = sel.map((it) => it.position.getComponent(axis));
    const target = mode === "min" ? Math.min(...vals)
      : mode === "max" ? Math.max(...vals)
      : vals.reduce((a, b) => a + b, 0) / vals.length;
    for (const it of sel) it.position.setComponent(axis, target);
    afterArrange(sel);
  }
  // Space selected objects evenly between the two extremes along one axis.
  function distributeSelection(axis: 0 | 1 | 2): void {
    const sel = ui.getSelection();
    if (sel.length < 3) return;
    const sorted = [...sel].sort((a, b) => a.position.getComponent(axis) - b.position.getComponent(axis));
    const lo = sorted[0].position.getComponent(axis);
    const hi = sorted[sorted.length - 1].position.getComponent(axis);
    const step = (hi - lo) / (sorted.length - 1);
    sorted.forEach((it, i) => it.position.setComponent(axis, lo + i * step));
    afterArrange(sel);
  }
  // Randomly jitter selected objects across the ground plane (uniform disk),
  // with a little random spin so non-spherical shapes look naturally scattered.
  function scatterSelection(radius: number): void {
    const sel = ui.getSelection();
    if (!sel.length) return;
    for (const it of sel) {
      const ang = Math.random() * Math.PI * 2;
      const r = Math.sqrt(Math.random()) * radius; // sqrt → even areal spread
      it.position.x += Math.cos(ang) * r;
      it.position.z += Math.sin(ang) * r;
      if (!(it instanceof Light) && !(it instanceof Emitter)) it.rotation.y += (Math.random() - 0.5) * Math.PI;
    }
    afterArrange(sel);
  }
  // Mirror the selection across the plane (perpendicular to `axis`) through the
  // selection pivot: reflect positions, conjugate rotations (S·R·S keeps a proper
  // rotation), and flip a directional light's aim. Mesh handedness is unchanged
  // (no negative scale), so asymmetric meshes are repositioned/re-oriented only.
  function mirrorSelection(axis: 0 | 1 | 2): void {
    const sel = ui.getSelection();
    if (!sel.length) return;
    const pivot = selectionPivot(new Vector3());
    const s = new Matrix4().makeScale(axis === 0 ? -1 : 1, axis === 1 ? -1 : 1, axis === 2 ? -1 : 1);
    const m = new Matrix4();
    for (const it of sel) {
      it.position.setComponent(axis, 2 * pivot.getComponent(axis) - it.position.getComponent(axis));
      if (it instanceof Light) {
        if (it.type === LightType.Directional) it.direction.setComponent(axis, -it.direction.getComponent(axis));
      } else if (!(it instanceof Emitter)) {
        // Emitters are radially symmetric — reflect position only, no rotation.
        m.makeRotationFromEuler(it.rotation).premultiply(s).multiply(s);
        it.rotation.setFromRotationMatrix(m);
      }
    }
    afterArrange(sel);
  }
  function afterArrange(sel: ReturnType<typeof ui.getSelection>): void {
    if (sel.some((it) => it instanceof MeshInstance)) scene.touchInstances();
    else scene.touch();
    history.commit();
    ui.refresh();
    renderer.resetAccumulation();
  }
  function saveBookmark(i: number): void {
    camBookmarks[i] = { target: [cam.target.x, cam.target.y, cam.target.z], yaw: cam.yaw, pitch: cam.pitch, distance: cam.distance };
    ui.refresh();
  }
  function recallBookmark(i: number): void {
    const b = camBookmarks[i];
    if (!b) return;
    cam.target.set(b.target[0], b.target[1], b.target[2]);
    cam.yaw = b.yaw; cam.pitch = b.pitch; cam.distance = b.distance;
    cam.update();
    renderer.resetAccumulation();
  }
  function selectAll(): void {
    if (editInstance && editMesh) {
      const n = editMesh.verts.length / 3;
      for (let i = 0; i < n; i++) selectedVerts.add(i); // edit mode: all vertices
      return;
    }
    ui.setSelection(scene.selectables);
  }
  function escAction(): void {
    if (editInstance) { toggleEdit(editInstance); ui.refresh(); return; } // exit vertex edit
    if (poseInst) { togglePose(poseInst); ui.refresh(); return; }          // exit pose
    ui.select(null); // deselect and drop back to the Move tool
    tool = "move";
    ui.refresh();
  }
  function frameSelection(): void {
    const sel = ui.getSelection();
    if (!sel.length) return;
    const frames = sel.map((it) => gizmoFrame(it));
    const c = new Vector3();
    for (const f of frames) c.add(f.center);
    c.multiplyScalar(1 / frames.length);
    let r = 6;
    for (const f of frames) r = Math.max(r, f.center.distanceTo(c) + f.radius);
    cam.target.copy(c);
    cam.distance = Math.min(400, Math.max(12, r * 2.4));
    renderer.resetAccumulation();
  }

  // ---- keyboard: arrows fly, Ctrl/Shift + arrows look (rotate view in place) ----
  const ARROWS = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]);
  const pressed = new Set<string>();
  let rotateMod = false;
  const typingInField = () => {
    const a = document.activeElement;
    return a instanceof HTMLInputElement || a instanceof HTMLSelectElement;
  };
  window.addEventListener("keydown", (e) => {
    if (typingInField()) return;
    rotateMod = e.ctrlKey || e.shiftKey;
    if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
      e.preventDefault();
      if (poseInst) { if (e.shiftKey) poseRedoStep(); else poseUndoStep(); return; } // pose-local undo
      if (editInstance) return; // don't rewrite the scene mid-edit
      if (e.shiftKey) history.redo(); else history.undo();
      return;
    }
    if ((e.ctrlKey || e.metaKey) && (e.key === "y" || e.key === "Y")) {
      e.preventDefault();
      if (poseInst) { poseRedoStep(); return; }
      if (editInstance) return;
      history.redo();
      return;
    }
    if (e.key === "1" && tool !== "move") { tool = "move"; ui.refresh(); return; }
    if (e.key === "2" && tool !== "select") { tool = "select"; ui.refresh(); return; }
    if ((e.ctrlKey || e.metaKey) && (e.key === "a" || e.key === "A")) { e.preventDefault(); selectAll(); return; }
    if ((e.ctrlKey || e.metaKey) && (e.key === "d" || e.key === "D")) { e.preventDefault(); duplicateSelection(); return; }
    if (e.key === "Delete" || e.key === "Backspace") {
      if (editInstance || poseInst) return; // leave vertex/bone handling alone
      e.preventDefault();
      deleteSelection();
      return;
    }
    if (e.key === "Escape") { escAction(); return; }
    if ((e.key === "f" || e.key === "F") && !e.ctrlKey && !e.metaKey) { frameSelection(); return; }
    if (!e.ctrlKey && !e.metaKey) {
      if (e.key === "w" || e.key === "W") { gizmoMode = "translate"; ui.refresh(); return; }
      if (e.key === "e" || e.key === "E") { gizmoMode = "rotate"; ui.refresh(); return; }
      if (e.key === "r" || e.key === "R") { gizmoMode = "scale"; ui.refresh(); return; }
    }
    if (ARROWS.has(e.key)) {
      e.preventDefault();
      pressed.add(e.key);
    }
  });
  window.addEventListener("keyup", (e) => {
    rotateMod = e.ctrlKey || e.shiftKey;
    pressed.delete(e.key);
  });
  window.addEventListener("blur", () => pressed.clear());

  function applyCameraKeys(dt: number): boolean {
    if (pressed.size === 0) return false;
    const move = 36 * dt; // world units / sec
    const rot = 1.6 * dt; // radians / sec
    if (rotateMod) {
      // Look around in place — camera position stays fixed, only the view turns.
      if (pressed.has("ArrowLeft")) cam.lookAngles(rot, 0);
      if (pressed.has("ArrowRight")) cam.lookAngles(-rot, 0);
      if (pressed.has("ArrowUp")) cam.lookAngles(0, -rot);
      if (pressed.has("ArrowDown")) cam.lookAngles(0, rot);
    } else {
      if (pressed.has("ArrowUp")) cam.moveRelative(move, 0, 0);
      if (pressed.has("ArrowDown")) cam.moveRelative(-move, 0, 0);
      if (pressed.has("ArrowLeft")) cam.moveRelative(0, -move, 0);
      if (pressed.has("ArrowRight")) cam.moveRelative(0, move, 0);
    }
    return true;
  }

  // ---- selection gizmo overlay (2D, drawn over both views) ----
  const tmpV = new Vector3();
  const axPt = new Vector3();
  const lightTmp = new Vector3();
  const AXIS_COLORS = ["#ff5566", "#55dd66", "#5599ff"];
  const AXIS_LABELS = ["X", "Y", "Z"];

  function projectToScreen(p: Vector3): { x: number; y: number } | null {
    const v = tmpV.copy(p).sub(cam.position);
    const z = v.dot(cam.forward);
    if (z <= 0.05) return null; // behind camera
    const aspect = overlay.width / Math.max(1, overlay.height);
    const ndcx = v.dot(cam.right) / (z * aspect * cam.fovScale);
    const ndcy = v.dot(cam.up) / (z * cam.fovScale);
    return {
      x: (ndcx * 0.5 + 0.5) * overlay.width,
      y: (1 - (ndcy * 0.5 + 0.5)) * overlay.height,
    };
  }

  // Directional lights have no position; show their handle this far from the
  // current view target, along the light direction.
  const DIR_HANDLE_DIST = 45;
  function lightGizmoPos(l: Light, out = new Vector3()): Vector3 {
    if (l.type === LightType.Directional) {
      return out.copy(cam.target).addScaledVector(l.direction, DIR_HANDLE_DIST);
    }
    return out.copy(l.position);
  }

  function gizmoFrame(sel: Selectable): { center: Vector3; radius: number } {
    if (sel instanceof Light) return { center: lightGizmoPos(sel), radius: 7 };
    if (sel instanceof Emitter) return { center: new Vector3().copy(sel.position), radius: Math.max(2, sel.size * 3) };
    const radius = sel instanceof Primitive ? Math.max(sel.a, sel.b, sel.c) + 1.5 : 12 * sel.scale;
    return { center: new Vector3().copy(sel.position), radius };
  }

  /** Nearest light whose viewport handle is under the cursor, or null. */
  function pickLight(cx: number, cy: number): Light | null {
    let best: Light | null = null;
    let bestD = 14; // px hit radius
    for (const l of scene.lights) {
      const s = projectToScreen(lightGizmoPos(l, tmpV));
      if (!s) continue;
      const d = Math.hypot(cx - s.x, cy - s.y);
      if (d < bestD) { bestD = d; best = l; }
    }
    return best;
  }

  /** Begin dragging a light: point lights move by position (Shift = up/down),
   *  directional lights re-aim by their viewport handle. */
  function startLightDrag(l: Light, e: PointerEvent): void {
    dragTarget = l;
    if (l.type === LightType.Directional) {
      draggingDirLight = true;
      planePoint.copy(lightGizmoPos(l));
      planeNormal.copy(cam.forward).multiplyScalar(-1).normalize(); // screen-facing plane
      drag = "object";
      return;
    }
    draggingDirLight = false;
    dragVertical = e.shiftKey;
    planePoint.copy(l.position);
    if (dragVertical) planeNormal.copy(cam.forward).setY(0).normalize();
    else planeNormal.set(0, 1, 0);
    const ray = screenRay(e.clientX, e.clientY);
    const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
    if (hit) objDragHit0.copy(hit);
    beginGroupSnapshot(l);
    drag = "object";
  }

  /** Nearest emitter whose viewport handle is under the cursor, or null.
   *  Emitters aren't ray-traced geometry, so — like lights — they can't be GPU-
   *  picked; a small handle at the emitter origin is the click target. */
  function pickEmitter(cx: number, cy: number): Emitter | null {
    let best: Emitter | null = null;
    let bestD = 14; // px hit radius
    for (const em of scene.emitters) {
      const s = projectToScreen(tmpV.copy(em.position));
      if (!s) continue;
      const d = Math.hypot(cx - s.x, cy - s.y);
      if (d < bestD) { bestD = d; best = em; }
    }
    return best;
  }

  /** Begin dragging an emitter on the ground plane (Shift = vertical). */
  function startEmitterDrag(em: Emitter, e: PointerEvent): void {
    dragTarget = em;
    draggingDirLight = false;
    dragVertical = e.shiftKey;
    planePoint.copy(em.position);
    if (dragVertical) planeNormal.copy(cam.forward).setY(0).normalize();
    else planeNormal.set(0, 1, 0);
    const ray = screenRay(e.clientX, e.clientY);
    const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
    if (hit) objDragHit0.copy(hit);
    beginGroupSnapshot(em);
    drag = "object";
  }

  // Parameter along an axis line (base + s·axis) closest to the camera ray.
  function closestAxisParam(ro: Vector3, rd: Vector3, base: Vector3, axis: Vector3): number | null {
    const w0 = tmpV.copy(ro).sub(base);
    const b = rd.dot(axis); // rd and axis are unit length
    const denom = 1 - b * b;
    if (Math.abs(denom) < 1e-5) return null; // ray ∥ axis
    return (axis.dot(w0) - b * rd.dot(w0)) / denom;
  }

  function pointSegDist(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
    const dx = bx - ax, dy = by - ay;
    const len2 = dx * dx + dy * dy;
    const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
    return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
  }

  /** Which axis handle (0/1/2) of a gizmo at `center` is under the cursor, or -1. */
  function pickAxisAt(center: Vector3, radius: number, cx: number, cy: number): number {
    const c = projectToScreen(center);
    if (!c) return -1;
    let best = -1;
    let bestDist = 11; // px hit threshold
    for (let i = 0; i < 3; i++) {
      const tip = projectToScreen(axPt.copy(center).addScaledVector(WORLD_AXES[i], radius));
      if (!tip) continue;
      const d = pointSegDist(cx, cy, c.x, c.y, tip.x, tip.y);
      if (d < bestDist) { bestDist = d; best = i; }
    }
    return best;
  }

  // Draw the X/Y/Z axis handles of a gizmo centered at a world point.
  function drawAxes(center: Vector3, radius: number) {
    const c = projectToScreen(center);
    if (!c) return;
    octx.font = "bold 12px ui-monospace, monospace";
    for (let i = 0; i < 3; i++) {
      const e = projectToScreen(axPt.copy(center).addScaledVector(WORLD_AXES[i], radius));
      if (!e) continue;
      const active = i === activeAxisIdx;
      octx.strokeStyle = AXIS_COLORS[i];
      octx.lineWidth = active ? 4 : 2.5;
      octx.beginPath();
      octx.moveTo(c.x, c.y);
      octx.lineTo(e.x, e.y);
      octx.stroke();
      octx.fillStyle = AXIS_COLORS[i];
      octx.beginPath();
      octx.arc(e.x, e.y, active ? 5.5 : 4, 0, Math.PI * 2);
      octx.fill();
      octx.fillText(AXIS_LABELS[i], e.x + 5, e.y + 4);
    }
    octx.fillStyle = "rgba(255,255,255,0.9)";
    octx.beginPath();
    octx.arc(c.x, c.y, 3, 0, Math.PI * 2);
    octx.fill();
  }

  // Draw clickable handles for every light; the selected one gets extra chrome.
  function drawLightMarkers() {
    const sel = ui.getSelected();
    const selSet = ui.getSelection();
    for (const l of scene.lights) {
      const s = projectToScreen(lightGizmoPos(l, lightTmp));
      if (!s) continue;
      const on = selSet.includes(l);
      octx.save();
      if (l.type === LightType.Directional) {
        if (on) {
          const tc = projectToScreen(cam.target);
          if (tc) {
            octx.strokeStyle = "rgba(255,225,150,0.5)";
            octx.lineWidth = 1.5;
            octx.setLineDash([5, 4]);
            octx.beginPath(); octx.moveTo(tc.x, tc.y); octx.lineTo(s.x, s.y); octx.stroke();
            octx.setLineDash([]);
          }
        }
        const r = on ? 6 : 5;
        octx.strokeStyle = octx.fillStyle = on ? "#ffe680" : "rgba(255,225,150,0.85)";
        octx.lineWidth = on ? 2 : 1.5;
        octx.beginPath(); octx.arc(s.x, s.y, r, 0, Math.PI * 2); octx.fill();
        for (let i = 0; i < 8; i++) {
          const a = (i / 8) * Math.PI * 2;
          octx.beginPath();
          octx.moveTo(s.x + Math.cos(a) * (r + 2), s.y + Math.sin(a) * (r + 2));
          octx.lineTo(s.x + Math.cos(a) * (r + 6), s.y + Math.sin(a) * (r + 6));
          octx.stroke();
        }
      } else {
        const r = on ? 6 : 5;
        octx.strokeStyle = octx.fillStyle = on ? "#bfe0ff" : "rgba(150,200,255,0.85)";
        octx.lineWidth = on ? 2 : 1.5;
        octx.beginPath(); octx.arc(s.x, s.y, r, 0, Math.PI * 2); octx.stroke();
        octx.beginPath(); octx.arc(s.x, s.y, r - 2.5, 0, Math.PI * 2); octx.fill();
      }
      octx.restore();
    }
    if (sel instanceof Light && sel.type === LightType.Point) {
      const { center, radius } = gizmoFrame(sel);
      drawAxes(center, radius);
    }
  }

  // Draw a clickable diamond handle at every emitter origin (they have no
  // ray-traced geometry to click, like lights). Selected ones get extra chrome.
  function drawEmitterMarkers() {
    const selSet = ui.getSelection();
    for (const em of scene.emitters) {
      const s = projectToScreen(lightTmp.copy(em.position));
      if (!s) continue;
      const on = selSet.includes(em);
      const r = on ? 6 : 5;
      octx.save();
      octx.strokeStyle = octx.fillStyle = on ? "#ffd08a" : "rgba(255,180,110,0.85)";
      octx.lineWidth = on ? 2 : 1.5;
      octx.beginPath();
      octx.moveTo(s.x, s.y - r); octx.lineTo(s.x + r, s.y);
      octx.lineTo(s.x, s.y + r); octx.lineTo(s.x - r, s.y);
      octx.closePath();
      octx.stroke();
      octx.beginPath(); octx.arc(s.x, s.y, 1.6, 0, Math.PI * 2); octx.fill();
      octx.restore();
    }
  }

  // Oriented half-extents of a primitive's bounding cage (matches the SDF dims).
  const ghostQuat = new Quaternion();
  const ghostCorner = new Vector3();
  function primHalfExtents(p: Primitive, out: Vector3): Vector3 {
    switch (p.type) {
      case PrimType.Box:
      case PrimType.RoundedBox: return out.set(p.a, p.b, p.c);
      case PrimType.Torus: return out.set(p.a + p.b, p.b, p.a + p.b);
      case PrimType.Cylinder:
      case PrimType.Cone:
      case PrimType.Pyramid: return out.set(p.a, p.b, p.a);
      case PrimType.Capsule: return out.set(p.a, p.b + p.a, p.a);
      default: return out.set(p.a, p.a, p.a); // sphere, octahedron
    }
  }

  // Wireframe cages over every "carve" primitive, so invisible cutters are visible.
  function drawCarverGhosts() {
    if (!showCarverGhosts) return;
    octx.save();
    octx.setLineDash([4, 3]);
    octx.lineWidth = 1.25;
    for (const p of scene.prims) {
      if (!p.subtractive) continue;
      ghostQuat.setFromEuler(p.rotation);
      const h = primHalfExtents(p, scratch);
      const pts: ({ x: number; y: number } | null)[] = [];
      for (let i = 0; i < 8; i++) {
        ghostCorner.set(
          (i & 1 ? h.x : -h.x),
          (i & 2 ? h.y : -h.y),
          (i & 4 ? h.z : -h.z),
        ).applyQuaternion(ghostQuat).add(p.position);
        pts.push(projectToScreen(ghostCorner));
      }
      octx.strokeStyle = ui.getSelection().includes(p) ? "#ff9a5a" : "rgba(255,140,90,0.7)";
      for (let i = 0; i < 8; i++) {
        for (const bit of [1, 2, 4]) {
          if (i & bit) continue;
          const a = pts[i], b = pts[i | bit];
          if (!a || !b) continue;
          octx.beginPath();
          octx.moveTo(a.x, a.y);
          octx.lineTo(b.x, b.y);
          octx.stroke();
        }
      }
    }
    octx.restore();
  }

  // ---- what a joint drag moves ----
  // Grabbing a joint drags it to the cursor; everything hanging off it (the rest
  // of the arm, the hand, the fingers) rides along rigidly, because they're its
  // children. How far UP the edit is felt comes from the skeleton's own shape --
  // see `mesh/poseChain` -- so there's no "reach" dial to set before you drag.
  const ikChain = (effIdx: number, extend = false) => poseIkChain(poseBones, effIdx, extend);

  // ---- moving several joints at once ----
  // Each selected joint keeps its own chain and is solved toward its own start
  // position plus the cursor's travel, so the selection moves as a formation
  // rather than collapsing onto one point. Joints are solved shallowest-first:
  // when a selection contains both ends of a limb, the upper solve runs before
  // the lower one corrects on top of it.

  /** Depth of a bone in the skeleton, for solve ordering. */
  function boneDepth(b: Object3D, set: Set<Object3D>): number {
    let d = 0;
    for (let p = b.parent; p && set.has(p); p = p.parent) d++;
    return d;
  }

  function beginMultiDrag(e: PointerEvent): boolean {
    const set = new Set(poseBones);
    const grabs: MultiGrab[] = [];
    for (const i of boneSel) {
      const bone = poseBones[i];
      if (!bone) continue;
      const chain = ikChain(i, e.shiftKey);
      if (chain.length < 2) continue; // nothing to solve this one with
      const pole = chain.length === 3 ? autoPole(chain[0], chain[1], chain[2], new Vector3()) : null;
      grabs.push({ bone, start: bone.getWorldPosition(new Vector3()), chain, pole, depth: boneDepth(bone, set) });
    }
    if (grabs.length < 2) return false; // fall through to the single-joint drag
    grabs.sort((a, b) => a.depth - b.depth);
    // Drag plane faces the camera through the joint that was actually grabbed.
    poseBones[selectedBone].getWorldPosition(ikPlanePoint);
    const ray = screenRay(e.clientX, e.clientY);
    const hit = rayPlane(ray.ro, ray.rd, ikPlanePoint, cam.forward);
    if (!hit) return false;
    multiHit0.copy(hit);
    multiGrab = grabs;
    poseDragSnap = capturePose(); // whole group edit = one undo step
    drag = "multi";
    return true;
  }

  const multiTarget = new Vector3();
  function dragMulti(e: PointerEvent): void {
    const ray = screenRay(e.clientX, e.clientY);
    const hit = rayPlane(ray.ro, ray.rd, ikPlanePoint, cam.forward);
    if (!hit) return;
    hit.sub(multiHit0); // world-space travel since the grab
    for (const g of multiGrab) {
      multiTarget.copy(g.start).add(hit);
      if (g.chain.length === 3 && g.pole) {
        solveTwoBoneIK(g.chain[0], g.chain[1], g.chain[2], multiTarget, { pole: g.pole });
      } else {
        solveCCD(g.chain, multiTarget, { iterations: 8, damping: 0.5 });
      }
    }
  }

  // Skeleton overlay: bone lines + clickable joints.
  const boneWorld = new Vector3();
  const PICK_RADIUS = 16; // screen px a joint dot answers to
  const PICK_TIE = 6;     // dots this close to the best are treated as overlapping
  /**
   * The joint under the cursor, or -1. Where dots overlap — fingers, the spine
   * seen edge-on — the nearest one to the CAMERA wins rather than whichever
   * happens to be a pixel closer to the cursor, so you grab the joint you can
   * actually see.
   */
  function pickBone(cx: number, cy: number): number {
    let best = -1;
    let bestD = PICK_RADIUS;
    let bestZ = Infinity;
    for (let i = 0; i < poseBones.length; i++) {
      if (!boneOffered(i)) continue; // helper bones aren't grabbable
      poseBones[i].getWorldPosition(boneWorld);
      const s = projectToScreen(boneWorld);
      if (!s) continue;
      const d = Math.hypot(cx - s.x, cy - s.y);
      if (d >= PICK_RADIUS) continue;
      const z = boneWorld.sub(cam.position).dot(cam.forward); // depth along the view axis
      if (best < 0 || d < bestD - PICK_TIE || (d < bestD + PICK_TIE && z < bestZ)) {
        if (d < bestD) bestD = d;
        bestZ = z;
        best = i;
      }
    }
    return best;
  }

  // ---- FK rotation-ring gizmo ----
  // A bone's local axis `k` expressed in world space (unit) → `out`.
  function worldBoneAxis(bone: Object3D, k: number, out: Vector3): Vector3 {
    bone.getWorldQuaternion(gzBoneWQ);
    return out.copy(gzUnit[k]).applyQuaternion(gzBoneWQ).normalize();
  }
  // Orthonormal basis (u, v) of the plane ⟂ `n`, with u × v = n.
  function ringBasis(n: Vector3, u: Vector3, v: Vector3): void {
    u.set(0, 1, 0);
    if (Math.abs(n.y) > 0.9) u.set(1, 0, 0);
    u.cross(n).normalize();
    v.copy(n).cross(u).normalize();
  }
  // World radius that projects to a roughly constant on-screen ring size.
  function ringRadius(center: Vector3): number {
    const z = gzHit.copy(center).sub(cam.position).dot(cam.forward);
    return (46 * Math.max(z, 0.1) * cam.fovScale * 2) / Math.max(1, overlay.height);
  }
  // Signed angle of a world point around the current drag plane (origin = gzRefU).
  function ringAngle(hit: Vector3): number {
    gzTmpU.copy(hit).sub(gzCenter);
    const x = gzTmpU.dot(gzRefU);
    gzTmpV.copy(gzAxisWorld).cross(gzRefU); // second in-plane basis vector
    return Math.atan2(gzTmpU.dot(gzTmpV), x);
  }
  // Which rotation ring (0/1/2) of `boneIdx` is under the cursor, or -1.
  function pickBoneRing(cx: number, cy: number, boneIdx: number): number {
    const bone = poseBones[boneIdx];
    if (!bone) return -1;
    bone.getWorldPosition(gzCenter);
    const R = ringRadius(gzCenter);
    let best = -1, bestD = 8;
    for (let k = 0; k < 3; k++) {
      worldBoneAxis(bone, k, gzN);
      ringBasis(gzN, gzTmpU, gzTmpV);
      for (let a = 0; a < 48; a++) {
        const th = (a / 48) * Math.PI * 2;
        gzHit.copy(gzCenter).addScaledVector(gzTmpU, Math.cos(th) * R).addScaledVector(gzTmpV, Math.sin(th) * R);
        const s = projectToScreen(gzHit);
        if (!s) continue;
        const d = Math.hypot(cx - s.x, cy - s.y);
        if (d < bestD) { bestD = d; best = k; }
      }
    }
    return best;
  }
  function beginBoneGizmo(axis: number, e: PointerEvent): void {
    const bone = poseBones[selectedBone];
    if (!bone) return;
    gzAxis = axis;
    gzStartQuat.copy(bone.quaternion);
    // A multi-selection turns together: every selected joint takes the same
    // rotation about its OWN local axis, from its own starting orientation, so
    // e.g. all the finger joints curl by the same amount at once.
    gzGroup = [...boneSel]
      .filter((i) => i !== selectedBone && poseBones[i])
      .map((i) => ({ bone: poseBones[i], start: poseBones[i].quaternion.clone() }));
    bone.getWorldPosition(gzCenter);
    worldBoneAxis(bone, axis, gzAxisWorld);
    ringBasis(gzAxisWorld, gzRefU, gzTmpV); // gzRefU = angle origin in the plane
    const ray = screenRay(e.clientX, e.clientY);
    const hit = rayPlane(ray.ro, ray.rd, gzCenter, gzAxisWorld);
    gzPrevAngle = hit ? ringAngle(hit) : 0;
    gzAccum = 0;
    drag = "gzbone";
  }
  function dragBoneGizmo(e: PointerEvent): void {
    const bone = poseBones[selectedBone];
    if (!bone) return;
    const ray = screenRay(e.clientX, e.clientY);
    const hit = rayPlane(ray.ro, ray.rd, gzCenter, gzAxisWorld);
    if (!hit) return;
    // Unwrap: accumulate the shortest step each move so sweeps past ±180° don't flip.
    const a = ringAngle(hit);
    let step = a - gzPrevAngle;
    if (step > Math.PI) step -= 2 * Math.PI;
    if (step < -Math.PI) step += 2 * Math.PI;
    gzAccum += step;
    gzPrevAngle = a;
    let delta = gzAccum;
    if (snapEnabled) { const s = Math.PI / 12; delta = Math.round(delta / s) * s; } // 15°
    gzQ.setFromAxisAngle(gzUnit[gzAxis], delta);
    bone.quaternion.copy(gzStartQuat).multiply(gzQ).normalize();
    for (const g of gzGroup) g.bone.quaternion.copy(g.start).multiply(gzQ).normalize();
  }
  function drawBoneGizmo(boneIdx: number): void {
    const bone = poseBones[boneIdx];
    if (!bone) return;
    bone.getWorldPosition(gzCenter);
    const R = ringRadius(gzCenter);
    const active = drag === "gzbone";
    for (let k = 0; k < 3; k++) {
      worldBoneAxis(bone, k, gzN);
      ringBasis(gzN, gzTmpU, gzTmpV);
      octx.strokeStyle = AXIS_COLORS[k];
      octx.lineWidth = active && gzAxis === k ? 3 : 1.5;
      octx.globalAlpha = active && gzAxis !== k ? 0.3 : 0.9;
      octx.beginPath();
      let started = false;
      for (let a = 0; a <= 48; a++) {
        const th = (a / 48) * Math.PI * 2;
        gzHit.copy(gzCenter).addScaledVector(gzTmpU, Math.cos(th) * R).addScaledVector(gzTmpV, Math.sin(th) * R);
        const s = projectToScreen(gzHit);
        if (!s) { started = false; continue; }
        if (!started) { octx.moveTo(s.x, s.y); started = true; } else octx.lineTo(s.x, s.y);
      }
      octx.stroke();
    }
    octx.globalAlpha = 1;
  }

  /**
   * What a drag on `idx` would move, for the hover preview: the bones that will
   * be SOLVED (rotated), the subtree that RIDES along rigidly, and the joint that
   * stays put and acts as the pivot.
   */
  function dragAffects(idx: number, extend: boolean): { solved: Set<Object3D>; riding: Set<Object3D>; pivot: Object3D | null } {
    const set = new Set(poseBones);
    const solved = new Set<Object3D>();
    const riding = boneSubtree(poseBones[idx], set, new Set<Object3D>());
    let pivot: Object3D | null = null;
    if (ikEnabled) {
      const chain = ikChain(idx, extend);
      if (chain.length >= 2) {
        pivot = chain[0];
        for (let k = 1; k < chain.length; k++) solved.add(chain[k]); // segments that swing
      }
    }
    if (!pivot) pivot = poseBones[idx]; // FK / no chain: the joint rotates in place
    return { solved, riding, pivot };
  }

  // The hover preview is recomputed only when the grab it describes changes —
  // on a 1082-bone rig, rebuilding the bone sets every frame is real work.
  let affectCache: { key: string; value: ReturnType<typeof dragAffects> } | null = null;
  function hoverAffect(): ReturnType<typeof dragAffects> | null {
    if (drag !== "none" || hoverBone < 0) return null;
    const key = `${hoverBone}:${hoverExtend}:${ikEnabled}`;
    if (affectCache?.key !== key) affectCache = { key, value: dragAffects(hoverBone, hoverExtend) };
    return affectCache.value;
  }

  function drawPose() {
    if (!poseInst || poseBones.length === 0) return;
    const boneSet = new Set(poseBones);
    // Preview of the pending edit: which bones this grab would move, and where it
    // stops. Only while idle — during a drag you can see the real thing.
    const affect = hoverAffect();

    const boneIdx = new Map<Object3D, number>(poseBones.map((b, i) => [b, i]));
    for (let i = 0; i < poseBones.length; i++) {
      if (!boneOffered(i)) continue;
      const b = poseBones[i];
      // Draw to the nearest offered ancestor, so hiding helper bones doesn't
      // leave gaps in the skeleton.
      let par = b.parent;
      while (par && boneSet.has(par) && !boneOffered(boneIdx.get(par) ?? -1)) par = par.parent;
      if (!par || !boneSet.has(par)) continue;
      const a = projectToScreen(b.getWorldPosition(boneWorld));
      const c = projectToScreen(par.getWorldPosition(boneWorld));
      if (!a || !c) continue;
      // Amber = solved by the drag, cyan = riding along rigidly, green = unaffected.
      const solved = affect?.solved.has(b);
      const riding = affect?.riding.has(b);
      octx.strokeStyle = solved ? "rgba(255,200,90,0.95)"
        : riding ? "rgba(120,210,255,0.85)"
        : affect ? "rgba(120,220,160,0.25)" // dim the rest so the affected part reads
        : "rgba(120,220,160,0.55)";
      octx.lineWidth = solved || riding ? 2.5 : 1.5;
      octx.beginPath(); octx.moveTo(a.x, a.y); octx.lineTo(c.x, c.y); octx.stroke();
    }
    octx.lineWidth = 1.5;
    for (let i = 0; i < poseBones.length; i++) {
      if (!boneOffered(i)) continue;
      const s = projectToScreen(poseBones[i].getWorldPosition(boneWorld));
      if (!s) continue;
      const on = i === selectedBone;
      const inSel = boneSel.size > 1 && boneSel.has(i);
      const hot = i === hoverBone && affect !== null;
      octx.fillStyle = hot ? "#ffffff" : on ? "#ffe680" : inSel ? "#ffc14d" : "rgba(150,235,180,0.95)";
      octx.beginPath();
      octx.arc(s.x, s.y, hot ? 6.5 : on || inSel ? 5.5 : 3.5, 0, Math.PI * 2);
      octx.fill();
      if (inSel && !on) { // ring the co-selected joints so the set is readable
        octx.strokeStyle = "rgba(255,225,150,0.9)";
        octx.beginPath(); octx.arc(s.x, s.y, 8, 0, Math.PI * 2); octx.stroke();
      }
    }
    // The joint the edit pivots around — everything above it holds still.
    if (affect?.pivot) {
      const p = projectToScreen(affect.pivot.getWorldPosition(boneWorld));
      if (p) {
        octx.strokeStyle = "rgba(255,255,255,0.9)";
        octx.beginPath(); octx.arc(p.x, p.y, 8, 0, Math.PI * 2); octx.stroke();
        octx.beginPath(); octx.arc(p.x, p.y, 2, 0, Math.PI * 2); octx.stroke();
      }
    }
    // FK: the rotation-ring gizmo on the selected joint (X/Y/Z, quaternion rotate).
    if (!ikEnabled && selectedBone >= 0) drawBoneGizmo(selectedBone);
    octx.fillStyle = "rgba(255,255,255,0.85)";
    octx.font = "12px ui-monospace, monospace";
    // Name the joint under the cursor — on a rig with hundreds of bones you
    // otherwise can't tell an elbow from the twist helper sitting on top of it.
    if (hoverBone >= 0) {
      const b = poseBones[hoverBone];
      const s = projectToScreen(b.getWorldPosition(boneWorld));
      if (s) octx.fillText(b.name || `bone ${hoverBone}`, s.x + 10, s.y - 8);
    }
    const hint = boneSel.size > 1
      ? `${boneSel.size} joints selected — drag any of them to move the set${ikEnabled ? "" : " · the rings turn them all"}`
      : !ikEnabled
      ? "POSE MODE · drag the X/Y/Z rings to rotate the joint · Shift snaps 15°"
      : affect
      ? "amber = bends · blue = follows · ○ = holds still · Shift reaches one bone further"
      : "POSE MODE · drag any joint — what's below it follows · Ctrl-click to select several";
    const hidden = poseShowAll ? 0 : poseDeforms.reduce((n, d) => n + (d ? 0 : 1), 0);
    octx.fillText(hidden ? `${hint} · ${hidden} helper joints hidden` : hint, 12, overlay.height - 16);
  }

  function drawGizmo() {
    octx.clearRect(0, 0, overlay.width, overlay.height);
    if (editInstance && editMesh) { drawEditOverlay(); drawMarquee(); return; }
    if (poseInst) { drawPose(); drawMarquee(); return; }
    drawCarverGhosts();
    drawLightMarkers();
    drawEmitterMarkers();
    drawSelectionBoxes(); // dashed outline on every selected primitive/instance
    // One manipulator at the selection pivot, in the active transform mode.
    if (gizmoActive()) {
      const pivot = selectionPivot(tmpPivot);
      const radius = Math.max(8, selectionRadius(pivot));
      const c = projectToScreen(pivot);
      if (c) {
        if (gizmoMode === "rotate") {
          const edge = projectToScreen(tmpV.copy(pivot).addScaledVector(cam.right, radius));
          const rpx = edge ? Math.max(12, Math.hypot(edge.x - c.x, edge.y - c.y)) : 24;
          octx.save();
          octx.strokeStyle = "rgba(255,255,255,0.5)";
          octx.lineWidth = 1.25;
          octx.beginPath();
          octx.arc(c.x, c.y, rpx, 0, Math.PI * 2);
          octx.stroke();
          octx.restore();
        }
        drawAxes(pivot, radius);
        octx.save();
        octx.fillStyle = "rgba(255,255,255,0.85)";
        octx.font = "11px ui-monospace, monospace";
        octx.fillText(gizmoMode, c.x + 9, c.y - 9);
        octx.restore();
      }
    }
    drawMarquee();
  }

  // Dashed outline around every selected primitive/instance (group context).
  function drawSelectionBoxes() {
    const sel = ui.getSelection();
    octx.save();
    octx.strokeStyle = "rgba(255,255,255,0.8)";
    octx.lineWidth = 1.25;
    octx.setLineDash([5, 4]);
    for (const it of sel) {
      if (!(it instanceof Primitive || it instanceof MeshInstance)) continue;
      const { center, radius } = gizmoFrame(it);
      const c = projectToScreen(center);
      if (!c) continue;
      const edge = projectToScreen(tmpV.copy(center).addScaledVector(cam.right, radius));
      const rpx = edge ? Math.max(8, Math.hypot(edge.x - c.x, edge.y - c.y)) : 12;
      octx.strokeRect(c.x - rpx, c.y - rpx, rpx * 2, rpx * 2);
    }
    octx.setLineDash([]);
    octx.restore();
  }

  // The in-progress box-select rectangle.
  function drawMarquee() {
    if (drag !== "marquee") return;
    const x = Math.min(marqStartX, marqCurX), y = Math.min(marqStartY, marqCurY);
    const w = Math.abs(marqCurX - marqStartX), h = Math.abs(marqCurY - marqStartY);
    octx.save();
    octx.fillStyle = "rgba(120,200,255,0.12)";
    octx.strokeStyle = "rgba(120,200,255,0.9)";
    octx.lineWidth = 1;
    octx.fillRect(x, y, w, h);
    octx.strokeRect(x, y, w, h);
    octx.restore();
  }

  // ---- mesh vertex editing ----
  const selectedVerts = new Set<number>();
  const editModel = new Matrix4();
  const editInvRot = new Matrix3();
  const editTmpMat = new Matrix4();
  const editPlaneStart = new Vector3();
  const vertDragStart = new Map<number, [number, number, number]>();
  let vertDragMode: "none" | "axis" | "plane" = "none";
  const VERT_GIZMO_RADIUS = 6;
  const vTmp = new Vector3();
  const vTmp2 = new Vector3();

  function updateEditMatrices() {
    if (!editInstance) return;
    editInstance.modelMatrix(editModel);
    editInvRot.setFromMatrix4(editTmpMat.copy(editModel).invert());
  }
  function vertWorld(i: number, out: Vector3): Vector3 {
    const v = editMesh!.verts;
    return out.set(v[3 * i], v[3 * i + 1], v[3 * i + 2]).applyMatrix4(editModel);
  }
  function vertCentroidWorld(out: Vector3): Vector3 {
    out.set(0, 0, 0);
    const v = editMesh!.verts;
    for (const i of selectedVerts) out.set(out.x + v[3 * i], out.y + v[3 * i + 1], out.z + v[3 * i + 2]);
    if (selectedVerts.size > 0) out.multiplyScalar(1 / selectedVerts.size);
    return out.applyMatrix4(editModel);
  }
  function pickVertex(cx: number, cy: number): number {
    const n = editMesh!.verts.length / 3;
    let best = -1;
    let bestD = 10;
    for (let i = 0; i < n; i++) {
      const s = projectToScreen(vertWorld(i, vTmp));
      if (!s) continue;
      const d = Math.hypot(cx - s.x, cy - s.y);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }
  function snapshotVerts() {
    vertDragStart.clear();
    const v = editMesh!.verts;
    for (const i of selectedVerts) vertDragStart.set(i, [v[3 * i], v[3 * i + 1], v[3 * i + 2]]);
  }
  function applyWorldDelta(worldDelta: Vector3) {
    const local = vTmp2.copy(worldDelta).applyMatrix3(editInvRot);
    const v = editMesh!.verts;
    for (const [i, s] of vertDragStart) {
      v[3 * i] = s[0] + local.x; v[3 * i + 1] = s[1] + local.y; v[3 * i + 2] = s[2] + local.z;
    }
  }
  function commitEdit() {
    if (editInstance && editMesh) {
      scene.replaceBlas(editInstance.blasIndex, rebuildBlas(editMesh));
      history.commit(); // a vertex edit is one undo step (undo after exiting edit mode)
    }
  }
  function toggleEdit(inst: MeshInstance) {
    if (editInstance === inst) {
      editInstance = null;
      editMesh = null;
      selectedVerts.clear();
      return;
    }
    editInstance = inst;
    editMesh = buildEditMesh(scene.blases[inst.blasIndex]);
    selectedVerts.clear();
    mode = "render";
    applyMode(); // edits are only visible in the ray-traced view
  }

  // Left-button pointerdown while editing: pick/move vertices.
  function editPointerDown(e: PointerEvent): void {
    updateEditMatrices();
    if (selectedVerts.size > 0) {
      const idx = pickAxisAt(vertCentroidWorld(vTmp), VERT_GIZMO_RADIUS, e.clientX, e.clientY);
      if (idx >= 0) {
        activeAxisIdx = idx;
        dragAxis.copy(WORLD_AXES[idx]);
        axisStartPos.copy(vertCentroidWorld(vTmp));
        const ray = screenRay(e.clientX, e.clientY);
        axisStartS = closestAxisParam(ray.ro, ray.rd, axisStartPos, dragAxis) ?? 0;
        snapshotVerts();
        vertDragMode = "axis";
        drag = "vert";
        return;
      }
    }
    // Box-Select tool: left-drag rubber-bands vertices (single-picks on a click).
    if (tool === "select") {
      marqStartX = marqCurX = e.clientX;
      marqStartY = marqCurY = e.clientY;
      drag = "marquee";
      return;
    }
    const vIdx = pickVertex(e.clientX, e.clientY);
    if (vIdx < 0) {
      selectedVerts.clear();
      drag = "orbit";
      return;
    }
    if (e.shiftKey) {
      if (selectedVerts.has(vIdx)) selectedVerts.delete(vIdx);
      else selectedVerts.add(vIdx);
      drag = "none";
      return;
    }
    if (!selectedVerts.has(vIdx)) { selectedVerts.clear(); selectedVerts.add(vIdx); }
    planeNormal.set(0, 1, 0);
    planePoint.copy(vertCentroidWorld(vTmp));
    const ray = screenRay(e.clientX, e.clientY);
    const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
    if (hit) editPlaneStart.copy(hit);
    snapshotVerts();
    vertDragMode = "plane";
    drag = "vert";
  }

  function drawEditOverlay() {
    const em = editMesh!;
    updateEditMatrices();
    const n = em.verts.length / 3;
    const step = Math.max(1, Math.ceil(n / 4000)); // decimate display on big meshes
    octx.fillStyle = "rgba(120,200,255,0.5)";
    for (let i = 0; i < n; i += step) {
      if (selectedVerts.has(i)) continue;
      const s = projectToScreen(vertWorld(i, vTmp));
      if (s) octx.fillRect(s.x - 1.5, s.y - 1.5, 3, 3);
    }
    octx.fillStyle = "#ffcc33";
    for (const i of selectedVerts) {
      const s = projectToScreen(vertWorld(i, vTmp));
      if (s) octx.fillRect(s.x - 2.5, s.y - 2.5, 5, 5);
    }
    if (selectedVerts.size > 0) drawAxes(vertCentroidWorld(vTmp), VERT_GIZMO_RADIUS);
    octx.fillStyle = "rgba(255,255,255,0.85)";
    octx.font = "12px ui-monospace, monospace";
    octx.fillText(
      `EDIT MODE · ${selectedVerts.size} selected / ${n} verts · ` +
        (tool === "select" ? "drag to box-select (1=Move)" : "click+drag points, Shift=multi (2=Box-Select)"),
      12,
      overlay.height - 16,
    );
  }

  // ---- loop ----
  let fps = 0;
  let lastTime = performance.now();
  function frame() {
    const now = performance.now();
    const dt = Math.min(0.05, (now - lastTime) / 1000);

    // An off-screen export (turntable / cutscene) drives the renderer at its own
    // size; keep the rAF loop alive but idle so it doesn't fight the GPU/uniforms.
    if (offlineRendering) {
      lastTime = now;
      requestAnimationFrame(frame);
      return;
    }

    // Cutscene playback: advance the clock and pose the camera + DoF from the path.
    if (cutscenePlaying) {
      cutsceneTime += dt;
      const total = cutsceneDuration(cutsceneKeys);
      applyCutsceneAt(cutsceneTime);
      if (cutsceneTime >= total) {
        cutsceneTime = total; cutscenePlaying = false;
        ui.refresh(); // playback ended → restore Play button state
      } else if (now - lastCutsceneSync > 110) {
        lastCutsceneSync = now;
        ui.refreshCutscene(); // move the scrubber/time without rebuilding everything
      }
    }

    resize();
    if (scene.version !== uploadedVersion) {
      renderer.uploadScene(scene);
      uploadedVersion = scene.version;
    }
    if (scene.meshStructVersion !== uploadedMeshStruct) {
      renderer.uploadMeshPools(scene); // also re-uploads instances + materials
      uploadedMeshStruct = scene.meshStructVersion;
      uploadedInst = scene.instanceVersion;
      uploadedMeshMat = scene.meshMatVersion;
    } else if (scene.instanceVersion !== uploadedInst) {
      renderer.uploadInstances(scene);
      uploadedInst = scene.instanceVersion;
    }
    if (scene.meshMatVersion !== uploadedMeshMat) {
      renderer.uploadMeshMaterials(scene);
      uploadedMeshMat = scene.meshMatVersion;
    }
    if (scene.worldVersion !== uploadedWorld) {
      renderer.uploadWorld(scene);
      uploadedWorld = scene.worldVersion;
    }
    if (scene.primTexVersion !== uploadedPrimTex) {
      renderer.uploadPrimTextures(scene);
      uploadedPrimTex = scene.primTexVersion;
    }
    // Particle field: re-evaluate + upload only when the emitters changed or the
    // frozen scene-time moved (scrub/play), so a still render still accumulates.
    {
      const pt = liveParticleTime();
      if (scene.version !== uploadedParticleVersion || pt !== uploadedParticleTime) {
        uploadSceneParticles(pt, 0);
        uploadedParticleVersion = scene.version;
        uploadedParticleTime = pt;
      }
    }
    if (applyCameraKeys(dt)) renderer.resetAccumulation();
    cam.update();

    if (importing) {
      requestAnimationFrame(frame);
      return;
    }

    if (mode === "preview") {
      preview.syncInstances(scene);
      preview.render(cam);
      drawGizmo();
      hud.textContent =
        `Aerie · Preview (raster)\n` +
        `${tool === "select" ? "Box-Select" : "Move/Camera"} tool` +
        `${scene.instances.length ? ` · ${scene.instances.length} model(s)` : ""}`;
      lastTime = now;
      requestAnimationFrame(frame);
      return;
    }

    // The ray tracer draws from baked geometry, not the live skeleton, so a rig
    // only moves here if we step it and re-bake it each frame.
    const rebaked = stepRenderAnimation();

    if (renderer.frame < MAX_SAMPLES) renderer.renderSample(cam);
    drawGizmo();

    fps = fps * 0.9 + (1000 / Math.max(1, now - lastTime)) * 0.1;
    lastTime = now;

    const done = renderer.frame >= MAX_SAMPLES;
    hud.textContent =
      `Aerie · Render (ray-traced)${rebaked ? " · animating (1 sample/frame)" : ""}\n` +
      `${renderer.width}×${renderer.height} · ${Math.min(renderer.frame, MAX_SAMPLES)}/${MAX_SAMPLES}` +
      `${done ? " ✓" : ""} · ${fps.toFixed(0)} fps` +
      `${scene.instances.length ? ` · ${scene.instances.length} model(s)` : ""}\n` +
      `${tool === "select" ? "Box-Select" : "Move/Camera"} · gizmo ${gizmoMode}` +
      `${snapEnabled ? ` · snap ${snapGrid}` : ""}`;

    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

main().catch((e) => fail(String(e?.stack ?? e)));
