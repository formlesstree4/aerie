import {
  Mesh,
  SkinnedMesh,
  Object3D,
  Vector3,
  Matrix3,
  Matrix4,
  Box3,
  MeshStandardMaterial,
  Material,
  Group,
  AnimationClip,
  type BufferGeometry,
} from "three";
import { buildBVH } from "./bvh";
import { formatHint } from "./loaders/formatHints";
import type { BLAS, RawModel } from "../scene/scene";

export const TRI_STRIDE = 32; // floats per triangle (8 vec4)
const TEX_SIZE = 256;
const TARGET_SIZE = 24;

export interface ImportResult {
  model: RawModel;
  group: Group; // outer group (instance transform) wrapping normalized geometry
  inner: Object3D; // normalized geometry root — used for animation baking
  animations: AnimationClip[];
}

// ---------------------------------------------------------------- format registry
/** What a format loader hands back: a three.js scene graph + optional clips.
 *  The shared pipeline below turns it into an ImportResult. */
export interface LoadedScene {
  root: Object3D;
  animations?: AnimationClip[];
}

export interface FormatLoader {
  label: string; // human name, e.g. "glTF"
  exts: string[]; // lowercase extensions, no dot: ["glb", "gltf"]
  load: (file: File) => Promise<LoadedScene>;
}

const registry: FormatLoader[] = [];

/** Register a format loader (call at module load — see ./loaders). */
export function registerLoader(loader: FormatLoader): void {
  registry.push(loader);
}

export function supportedExtensions(): string[] {
  return registry.flatMap((l) => l.exts);
}

/** Value for an <input type="file"> accept attribute, e.g. ".glb,.gltf,.fbx". */
export function acceptAttribute(): string {
  return supportedExtensions().map((e) => `.${e}`).join(",");
}

function loaderFor(name: string): FormatLoader | undefined {
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  return registry.find((l) => l.exts.includes(ext));
}

/** Import any registered model format → a ready-to-place ImportResult. */
export async function importModel(file: File): Promise<ImportResult> {
  const loader = loaderFor(file.name);
  if (!loader) {
    const hint = formatHint(file.name);
    if (hint) throw new Error(hint);
    throw new Error(`Unsupported file type. Supported: ${acceptAttribute()}`);
  }
  const { root, animations } = await loader.load(file);
  return buildImportResult(root, file.name, animations ?? []);
}

/** Load ONLY the animation clips from a file — no geometry baking — so motion can
 *  be added to an already-imported rig (e.g. a downloaded Mixamo animation FBX,
 *  which often carries a skeleton and clips but little or no mesh). */
export async function importAnimations(file: File): Promise<AnimationClip[]> {
  const loader = loaderFor(file.name);
  if (!loader) {
    const hint = formatHint(file.name);
    throw new Error(hint ?? `Unsupported file type. Supported: ${acceptAttribute()}`);
  }
  const { animations } = await loader.load(file);
  return animations ?? [];
}

// ---------------------------------------------------------------- shared pipeline
const scratchCanvas = document.createElement("canvas");
scratchCanvas.width = scratchCanvas.height = TEX_SIZE;
const scratch2d = scratchCanvas.getContext("2d", { willReadFrequently: true })!;

function imageToLayer(img: CanvasImageSource): Uint8Array<ArrayBuffer> | null {
  try {
    scratch2d.clearRect(0, 0, TEX_SIZE, TEX_SIZE);
    scratch2d.drawImage(img, 0, 0, TEX_SIZE, TEX_SIZE);
    return new Uint8Array(scratch2d.getImageData(0, 0, TEX_SIZE, TEX_SIZE).data);
  } catch {
    return null; // undecodable / cross-origin
  }
}

/** Pull every mesh's triangles into a flat world-space buffer. `resolveMat`
 *  returns the (local) material id for a material; the caller owns the pools. */
function extractWorldTriangles(root: Object3D, resolveMat: (m: Material) => number): number[] {
  const tris: number[] = [];
  const v = new Vector3();
  const n = new Vector3();

  root.traverse((obj) => {
    if (!(obj instanceof Mesh)) return;
    const mesh = obj as Mesh;
    const geom = mesh.geometry as BufferGeometry;
    const pos = geom.getAttribute("position");
    if (!pos) return;
    const nrm = geom.getAttribute("normal");
    const uv = geom.getAttribute("uv");
    const col = geom.getAttribute("color"); // COLOR_0 (linear, optional)
    const index = geom.getIndex();
    const skinned = mesh instanceof SkinnedMesh ? (mesh as SkinnedMesh) : null;
    skinned?.skeleton.update();

    const normalMat = new Matrix3().getNormalMatrix(mesh.matrixWorld);
    const triCount = index ? index.count / 3 : pos.count / 3;
    const groups = geom.groups.length
      ? geom.groups
      : [{ start: 0, count: triCount * 3, materialIndex: 0 }];
    const matArray = Array.isArray(mesh.material) ? mesh.material : [mesh.material];

    const wp = (i: number): [number, number, number] => {
      v.fromBufferAttribute(pos, i);
      if (skinned) skinned.applyBoneTransform(i, v);
      v.applyMatrix4(mesh.matrixWorld);
      return [v.x, v.y, v.z];
    };
    const wn = (i: number): [number, number, number] => {
      if (!nrm) return [0, 1, 0];
      n.fromBufferAttribute(nrm, i).applyMatrix3(normalMat).normalize();
      return [n.x, n.y, n.z];
    };
    const uvAt = (i: number): [number, number] => (uv ? [uv.getX(i), uv.getY(i)] : [0, 0]);
    // Per-vertex color packed into one float (r + g·256 + b·65536, exact in f32).
    // White (no COLOR_0) decodes to (1,1,1) → a no-op multiply in the shader.
    const cAt = (i: number): number => {
      if (!col) return 0xffffff;
      const q = (x: number) => Math.max(0, Math.min(255, Math.round(x * 255)));
      return q(col.getX(i)) + q(col.getY(i)) * 256 + q(col.getZ(i)) * 65536;
    };

    for (const g of groups) {
      const mat = matArray[g.materialIndex ?? 0] ?? matArray[0];
      const mid = resolveMat(mat as Material);
      const end = g.start + g.count;
      for (let f = g.start; f < end; f += 3) {
        const a = index ? index.getX(f) : f;
        const b = index ? index.getX(f + 1) : f + 1;
        const c = index ? index.getX(f + 2) : f + 2;
        const p0 = wp(a), p1 = wp(b), p2 = wp(c);
        const n0 = wn(a), n1 = wn(b), n2 = wn(c);
        const u0 = uvAt(a), u1 = uvAt(b), u2 = uvAt(c);
        tris.push(
          p0[0], p0[1], p0[2], 0,
          p1[0], p1[1], p1[2], 0,
          p2[0], p2[1], p2[2], 0,
          n0[0], n0[1], n0[2], 0,
          n1[0], n1[1], n1[2], cAt(a), // n1.w = vertex-color 0 (packed)
          n2[0], n2[1], n2[2], cAt(b), // n2.w = vertex-color 1 (packed)
          u0[0], u0[1], u1[0], u1[1],
          u2[0], u2[1], mid, cAt(c),   // uv2m.w = vertex-color 2 (packed)
        );
      }
    }
  });
  return tris;
}

/** Map a three.js material (Standard, Phong, Lambert, …) into our 12-float slot. */
function resolveMatInto(
  mat: Material,
  materials: number[],
  materialNames: string[],
  texLayers: Uint8Array<ArrayBuffer>[],
  normalLayers: Uint8Array<ArrayBuffer>[],
): void {
  materialNames.push(mat.name || `material ${materialNames.length + 1}`);

  const std = mat as MeshStandardMaterial & { shininess?: number };
  const base = std.color ?? { r: 0.8, g: 0.8, b: 0.8 };
  const emis = std.emissive ?? { r: 0, g: 0, b: 0 };
  const eInt = std.emissiveIntensity ?? 1;

  // Standard materials carry metalness/roughness; Phong/Lambert don't, so fall
  // back to a sensible matte (deriving roughness from Phong shininess if present).
  const metalness = std.metalness ?? 0;
  const roughness =
    std.roughness ??
    (std.shininess !== undefined ? Math.min(1, Math.max(0.15, 1 - std.shininess / 100)) : 0.6);

  let texLayer = -1;
  if (std.map?.image) {
    const layer = imageToLayer(std.map.image);
    if (layer) { texLayer = texLayers.length; texLayers.push(layer); }
  }
  let normalLayer = -1;
  if (std.normalMap?.image) {
    const layer = imageToLayer(std.normalMap.image);
    if (layer) { normalLayer = normalLayers.length; normalLayers.push(layer); }
  }

  materials.push(
    base.r, base.g, base.b, std.opacity ?? 1,
    emis.r * eInt, emis.g * eInt, emis.b * eInt, 0,
    metalness, roughness, texLayer, normalLayer,
  );
}

/** Format-agnostic: bake a three.js scene graph into a placeable model. */
export function buildImportResult(
  root: Object3D,
  name: string,
  animations: AnimationClip[],
): ImportResult {
  root.updateMatrixWorld(true);

  const materials: number[] = [];
  const materialNames: string[] = [];
  const matIndex = new Map<Material, number>();
  const texLayers: Uint8Array<ArrayBuffer>[] = [];
  const normalLayers: Uint8Array<ArrayBuffer>[] = [];

  const resolveMat = (mat: Material): number => {
    const existing = matIndex.get(mat);
    if (existing !== undefined) return existing;
    const id = matIndex.size;
    matIndex.set(mat, id);
    resolveMatInto(mat, materials, materialNames, texLayers, normalLayers);
    return id;
  };

  const worldTris = extractWorldTriangles(root, resolveMat);
  if (worldTris.length === 0) throw new Error("No triangle geometry found in file.");

  const raw = new Float32Array(worldTris);
  const { matrix: normalize, halfHeight } = normalizeToLocal(raw);
  computeFaceTangents(raw);

  // Mirror the normalization onto the preview geometry, wrapped in an outer
  // group that carries the instance transform.
  root.applyMatrix4(normalize);
  const outer = new Group();
  outer.add(root);

  const { nodes, tris: ordered, nodeCount } = buildBVH(raw, TRI_STRIDE);

  const model: RawModel = {
    name,
    blas: { tris: ordered, triCount: ordered.length / TRI_STRIDE, nodes, nodeCount },
    materials,
    materialNames,
    texLayers,
    normalLayers,
    texSize: TEX_SIZE,
    halfHeight,
  };
  return { model, group: outer, inner: root, animations };
}

/** Flatten a posed/transformed scene graph into world-space triangles (our
 *  32-float layout) with per-triangle *local* material ids (0-based, encounter
 *  order — matching the model's scene-material order). For CSG baking. */
export function worldMesh(root: Object3D): { tris: Float32Array; matCount: number } {
  root.updateMatrixWorld(true);
  const map = new Map<Material, number>();
  const tris = extractWorldTriangles(root, (mat) => {
    let id = map.get(mat);
    if (id === undefined) { id = map.size; map.set(mat, id); }
    return id;
  });
  return { tris: new Float32Array(tris), matCount: map.size };
}

/** Re-bake `inner` at its current animation pose into a fresh BLAS. The caller
 *  must have placed `inner` in LOCAL space (instance transform removed) and
 *  updated its world matrices. `matBase` rebases material ids into the pool. */
export function bakePose(inner: Object3D, matBase: number): BLAS {
  // Fresh resolver — same traversal order as import, so ids line up with the pool.
  const map = new Map<Material, number>();
  const resolve = (m: Material): number => {
    const e = map.get(m);
    if (e !== undefined) return e;
    const id = map.size;
    map.set(m, id);
    return id;
  };
  const raw = new Float32Array(extractWorldTriangles(inner, resolve));
  computeFaceTangents(raw);
  for (let i = 0; i < raw.length; i += TRI_STRIDE) raw[i + 30] += matBase;
  const { nodes, tris, nodeCount } = buildBVH(raw, TRI_STRIDE);
  return { tris, triCount: tris.length / TRI_STRIDE, nodes, nodeCount };
}

/** Center the model on all axes and scale to TARGET_SIZE, so the local origin
 *  is the geometric center (rotation pivots there). Mutates `raw`; returns the
 *  equivalent matrix and the scaled half-height (to rest the base on import). */
function normalizeToLocal(raw: Float32Array): { matrix: Matrix4; halfHeight: number } {
  const box = new Box3();
  const p = new Vector3();
  for (let i = 0; i < raw.length; i += TRI_STRIDE) {
    for (let vtx = 0; vtx < 3; vtx++) {
      const o = i + vtx * 4;
      box.expandByPoint(p.set(raw[o], raw[o + 1], raw[o + 2]));
    }
  }
  const size = box.getSize(new Vector3());
  const center = box.getCenter(new Vector3());
  const s = TARGET_SIZE / Math.max(size.x, size.y, size.z, 1e-3);
  const ox = -s * center.x, oy = -s * center.y, oz = -s * center.z;
  for (let i = 0; i < raw.length; i += TRI_STRIDE) {
    for (let vtx = 0; vtx < 3; vtx++) {
      const o = i + vtx * 4;
      raw[o] = raw[o] * s + ox;
      raw[o + 1] = raw[o + 1] * s + oy;
      raw[o + 2] = raw[o + 2] * s + oz;
    }
  }
  const matrix = new Matrix4().makeTranslation(ox, oy, oz).multiply(new Matrix4().makeScale(s, s, s));
  return { matrix, halfHeight: (size.y * s) / 2 };
}

/** Compute a per-face tangent from positions + UVs, packed into the unused
 *  .w lanes: tangent xyz → p0.w/p1.w/p2.w, handedness → n0.w. */
export function computeFaceTangents(raw: Float32Array): void {
  for (let i = 0; i < raw.length; i += TRI_STRIDE) {
    const p0x = raw[i], p0y = raw[i + 1], p0z = raw[i + 2];
    const e1x = raw[i + 4] - p0x, e1y = raw[i + 5] - p0y, e1z = raw[i + 6] - p0z;
    const e2x = raw[i + 8] - p0x, e2y = raw[i + 9] - p0y, e2z = raw[i + 10] - p0z;
    const u0 = raw[i + 24], v0 = raw[i + 25];
    const du1 = raw[i + 26] - u0, dv1 = raw[i + 27] - v0;
    const du2 = raw[i + 28] - u0, dv2 = raw[i + 29] - v0;
    const denom = du1 * dv2 - du2 * dv1;

    let tx: number, ty: number, tz: number, hand = 1;
    if (Math.abs(denom) > 1e-10) {
      const r = 1 / denom;
      tx = (e1x * dv2 - e2x * dv1) * r;
      ty = (e1y * dv2 - e2y * dv1) * r;
      tz = (e1z * dv2 - e2z * dv1) * r;
      const len = Math.hypot(tx, ty, tz) || 1;
      tx /= len; ty /= len; tz /= len;
      // Handedness from the geometric normal and the UV bitangent.
      const fnx = e1y * e2z - e1z * e2y;
      const fny = e1z * e2x - e1x * e2z;
      const fnz = e1x * e2y - e1y * e2x;
      const bx = (e2x * du1 - e1x * du2) * r;
      const by = (e2y * du1 - e1y * du2) * r;
      const bz = (e2z * du1 - e1z * du2) * r;
      const cx = fny * tz - fnz * ty;
      const cy = fnz * tx - fnx * tz;
      const cz = fnx * ty - fny * tx;
      hand = cx * bx + cy * by + cz * bz < 0 ? -1 : 1;
    } else {
      const len = Math.hypot(e1x, e1y, e1z) || 1;
      tx = e1x / len; ty = e1y / len; tz = e1z / len;
    }
    raw[i + 3] = tx; raw[i + 7] = ty; raw[i + 11] = tz; raw[i + 15] = hand;
  }
}
