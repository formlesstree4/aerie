import {
  BufferGeometry,
  Float32BufferAttribute,
  Mesh,
  MeshStandardMaterial,
  Material,
  Group,
  Vector3,
  DataTexture,
  RGBAFormat,
  SRGBColorSpace,
} from "three";
import { Evaluator, Brush, ADDITION, SUBTRACTION } from "three-bvh-csg";
import { buildBVH } from "./bvh";
import { computeFaceTangents, TRI_STRIDE } from "./modelImport";
import type { BLAS, RawModel } from "../scene/scene";

/** A material as we store it (color + PBR + raw texture bytes). */
export interface MatDesc {
  name: string;
  color: [number, number, number];
  metalness: number;
  roughness: number;
  tex: Uint8Array<ArrayBuffer> | null;
  normal: Uint8Array<ArrayBuffer> | null;
}

/** One object entering a boolean op: world-space geometry (with per-material
 *  groups + uv) and the materials those groups index. */
export interface BoolInput {
  geometry: BufferGeometry;
  descs: MatDesc[];
  subtractive: boolean;
}

const evaluator = new Evaluator();
evaluator.attributes = ["position", "normal", "uv"];

/** Build the renderer's 32-float world tris from a flattened model into a
 *  grouped, uv-bearing BufferGeometry (one group per material id). */
export function trisToGeometry(tris: Float32Array): BufferGeometry {
  const tcount = tris.length / TRI_STRIDE;
  // Sort triangles by material id so each material is one contiguous group.
  const order = [...Array(tcount).keys()].sort(
    (a, b) => tris[a * TRI_STRIDE + 30] - tris[b * TRI_STRIDE + 30],
  );
  const pos = new Float32Array(tcount * 9);
  const nor = new Float32Array(tcount * 9);
  const uv = new Float32Array(tcount * 6);
  const groups: { start: number; count: number; mat: number }[] = [];
  let cur = -1;
  for (let ti = 0; ti < tcount; ti++) {
    const o = order[ti] * TRI_STRIDE;
    const mid = tris[o + 30];
    if (mid !== cur) { groups.push({ start: ti * 3, count: 0, mat: mid }); cur = mid; }
    groups[groups.length - 1].count += 3;
    for (let v = 0; v < 3; v++) {
      const d3 = (ti * 3 + v) * 3;
      const d2 = (ti * 3 + v) * 2;
      const vo = o + v * 4;
      pos[d3] = tris[vo]; pos[d3 + 1] = tris[vo + 1]; pos[d3 + 2] = tris[vo + 2];
      nor[d3] = tris[o + 12 + v * 4]; nor[d3 + 1] = tris[o + 12 + v * 4 + 1]; nor[d3 + 2] = tris[o + 12 + v * 4 + 2];
      const uo = o + 24 + v * 2;
      uv[d2] = tris[uo]; uv[d2 + 1] = tris[uo + 1];
    }
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(pos, 3));
  g.setAttribute("normal", new Float32BufferAttribute(nor, 3));
  g.setAttribute("uv", new Float32BufferAttribute(uv, 2));
  for (const grp of groups) g.addGroup(grp.start, grp.count, grp.mat);
  return g;
}

function texFromBytes(bytes: Uint8Array<ArrayBuffer>, srgb: boolean): DataTexture {
  const t = new DataTexture(bytes, 256, 256, RGBAFormat);
  if (srgb) t.colorSpace = SRGBColorSpace;
  t.needsUpdate = true;
  return t;
}

/** A throwaway three material carrying our descriptor (for CSG + preview). */
function threeMatFor(desc: MatDesc): MeshStandardMaterial {
  const m = new MeshStandardMaterial({ metalness: desc.metalness, roughness: desc.roughness });
  m.color.setRGB(desc.color[0], desc.color[1], desc.color[2]);
  if (desc.tex) m.map = texFromBytes(desc.tex, true);
  if (desc.normal) m.normalMap = texFromBytes(desc.normal, false);
  m.userData.desc = desc;
  return m;
}

function ensureUV(g: BufferGeometry): void {
  if (!g.getAttribute("uv")) {
    const n = g.getAttribute("position").count;
    g.setAttribute("uv", new Float32BufferAttribute(new Float32Array(n * 2), 2));
  }
}

function brushFor(input: BoolInput): Brush {
  ensureUV(input.geometry);
  if (!input.geometry.getAttribute("normal")) input.geometry.computeVertexNormals();
  const b = new Brush(input.geometry, input.descs.map(threeMatFor));
  b.updateMatrixWorld();
  return b;
}

/** Bake a boolean group into one multi-material model (textures preserved) +
 *  a preview group. Re-centers the result; returns its world center. */
export function csgBuildModel(
  inputs: BoolInput[],
  name: string,
): { model: RawModel; group: Group; center: Vector3 } | null {
  const solids = inputs.filter((i) => !i.subtractive);
  const negatives = inputs.filter((i) => i.subtractive);
  if (solids.length === 0) return null;

  let acc = brushFor(solids[0]);
  for (let i = 1; i < solids.length; i++) acc = evaluator.evaluate(acc, brushFor(solids[i]), ADDITION);
  for (const neg of negatives) acc = evaluator.evaluate(acc, brushFor(neg), SUBTRACTION);

  const geom = acc.geometry;
  const mats: Material[] = Array.isArray(acc.material) ? acc.material : [acc.material];

  // Re-center so the new object pivots about its own middle.
  geom.computeBoundingBox();
  const center = geom.boundingBox!.getCenter(new Vector3());
  geom.translate(-center.x, -center.y, -center.z);

  // Pool the materials' textures (deduped) and emit our material floats.
  const texPool: Uint8Array<ArrayBuffer>[] = [];
  const normalPool: Uint8Array<ArrayBuffer>[] = [];
  const texIndex = new Map<Uint8Array<ArrayBuffer>, number>();
  const normalIndex = new Map<Uint8Array<ArrayBuffer>, number>();
  const layerFor = (
    bytes: Uint8Array<ArrayBuffer> | null,
    pool: Uint8Array<ArrayBuffer>[],
    map: Map<Uint8Array<ArrayBuffer>, number>,
  ): number => {
    if (!bytes) return -1;
    let idx = map.get(bytes);
    if (idx === undefined) { idx = pool.length; pool.push(bytes); map.set(bytes, idx); }
    return idx;
  };

  const materials: number[] = [];
  const materialNames: string[] = [];
  for (const m of mats) {
    const d = (m.userData.desc as MatDesc) ?? {
      name: "material", color: [0.8, 0.8, 0.85], metalness: 0, roughness: 0.6, tex: null, normal: null,
    };
    const tl = layerFor(d.tex, texPool, texIndex);
    const nl = layerFor(d.normal, normalPool, normalIndex);
    materials.push(d.color[0], d.color[1], d.color[2], 1, 0, 0, 0, 0, d.metalness, d.roughness, tl, nl);
    materialNames.push(d.name);
  }

  const model: RawModel = {
    name,
    blas: geometryToBlas(geom),
    materials,
    materialNames,
    texLayers: texPool,
    normalLayers: normalPool,
    texSize: 256,
    halfHeight: 0,
  };
  const group = new Group();
  group.add(new Mesh(geom, mats.length === 1 ? mats[0] : mats));
  return { model, group, center };
}

/** De-index a grouped geometry into our triangle BLAS layout, carrying uv +
 *  per-triangle material id (from the geometry's groups). */
function geometryToBlas(geom: BufferGeometry): BLAS {
  const pos = geom.getAttribute("position");
  if (!geom.getAttribute("normal")) geom.computeVertexNormals();
  const nrm = geom.getAttribute("normal");
  const uv = geom.getAttribute("uv");
  const index = geom.getIndex();
  const triCount = index ? index.count / 3 : pos.count / 3;

  // Per-triangle material id from the geometry groups (default 0).
  const triMat = new Int32Array(triCount);
  for (const g of geom.groups) {
    const startTri = g.start / 3;
    const cnt = g.count / 3;
    for (let t = startTri; t < startTri + cnt && t < triCount; t++) triMat[t] = g.materialIndex ?? 0;
  }

  const raw = new Float32Array(triCount * TRI_STRIDE);
  for (let t = 0; t < triCount; t++) {
    const o = t * TRI_STRIDE;
    for (let v = 0; v < 3; v++) {
      const i = index ? index.getX(t * 3 + v) : t * 3 + v;
      raw[o + v * 4] = pos.getX(i); raw[o + v * 4 + 1] = pos.getY(i); raw[o + v * 4 + 2] = pos.getZ(i);
      raw[o + 12 + v * 4] = nrm.getX(i); raw[o + 12 + v * 4 + 1] = nrm.getY(i); raw[o + 12 + v * 4 + 2] = nrm.getZ(i);
      const uo = o + 24 + v * 2;
      raw[uo] = uv ? uv.getX(i) : 0; raw[uo + 1] = uv ? uv.getY(i) : 0;
    }
    raw[o + 30] = triMat[t];
    raw[o + 19] = 0xffffff; raw[o + 23] = 0xffffff; raw[o + 31] = 0xffffff; // white vertex color
  }
  computeFaceTangents(raw);
  const { nodes, tris, nodeCount } = buildBVH(raw, TRI_STRIDE);
  return { tris, triCount: tris.length / TRI_STRIDE, nodes, nodeCount };
}
