import {
  BufferGeometry,
  Float32BufferAttribute,
  Mesh,
  MeshStandardMaterial,
  Group,
  DataTexture,
  RGBAFormat,
  SRGBColorSpace,
} from "three";
import { Scene, Primitive, Light, MeshInstance } from "./scene";
import type { CamKey } from "./cutscene";
import { buildBVH } from "../mesh/bvh";
import { TRI_STRIDE } from "../mesh/modelImport";

const SAVE_VERSION = 2;

// ---- base64 <-> typed arrays ----
function u8ToB64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk) as unknown as number[]);
  }
  return btoa(binary);
}
export function b64ToU8(s: string): Uint8Array<ArrayBuffer> {
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function f32ToB64(a: Float32Array): string {
  return u8ToB64(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
}
function b64ToF32(s: string): Float32Array<ArrayBuffer> {
  return new Float32Array(b64ToU8(s).buffer);
}

const v3 = (o: { x: number; y: number; z: number }) => [o.x, o.y, o.z];

/** Per-instance pose (bone local transforms) supplied by the preview. */
export type BonePose = { p: [number, number, number]; q: [number, number, number, number] }[];

export interface SerializeHooks {
  fileFor: (blasIndex: number) => { name: string; bytes: Uint8Array<ArrayBuffer> } | null;
  poseFor: (instanceId: number) => { bones: BonePose; paused: boolean } | null;
  cutscene: CamKey[];
}

// ---- serialize ----
export function serializeScene(scene: Scene, hooks: SerializeHooks): string {
  return JSON.stringify({
    version: SAVE_VERSION,
    world: scene.world,
    prims: scene.prims.map((p) => ({
      type: p.type, name: p.name, position: v3(p.position), rotation: v3(p.rotation),
      a: p.a, b: p.b, c: p.c, color: p.color, reflectivity: p.reflectivity,
      pattern: p.pattern, colorB: p.colorB, patternScale: p.patternScale,
      bump: p.bump, bumpScale: p.bumpScale, imageLayer: p.imageLayer,
      cornerRadius: p.cornerRadius, subtractive: p.subtractive, group: p.group,
    })),
    lights: scene.lights.map((l) => ({
      type: l.type, name: l.name, direction: v3(l.direction), position: v3(l.position),
      color: l.color, intensity: l.intensity, softness: l.softness, range: l.range,
      bodyRadius: l.bodyRadius, inSky: l.inSky, ringInner: l.ringInner, ringOuter: l.ringOuter,
      ringOpacity: l.ringOpacity, ringTilt: l.ringTilt, ringColor: l.ringColor,
    })),
    mesh: {
      texSize: scene.meshTexSize,
      blases: scene.blases.map((b) => f32ToB64(b.tris)),
      blasMatBase: scene.blasMatBase,
      blasMatCount: scene.blasMatCount,
      materials: scene.meshMaterials,
      materialNames: scene.meshMaterialNames,
      texLayers: scene.meshTexLayers.map(u8ToB64),
      normalLayers: scene.meshNormalLayers.map(u8ToB64),
      instances: scene.instances.map((m) => {
        const file = hooks.fileFor(m.blasIndex);
        return {
          blasIndex: m.blasIndex, name: m.name, position: v3(m.position),
          rotation: v3(m.rotation), scale: m.scale, subtractive: m.subtractive, group: m.group,
          landform: m.landform ?? null, // editable recipe + seed for procedural landforms
          file: file ? { name: file.name, bytes: u8ToB64(file.bytes) } : null,
          pose: file ? hooks.poseFor(m.id) : null,
        };
      }),
    },
    primImages: { size: scene.primImageSize, layers: scene.primImageLayers.map(u8ToB64) },
    cutscene: hooks.cutscene,
  });
}

// ---- deserialize (repopulates the Scene; returns the parsed data for the caller
// to rebuild preview groups / re-import rigged models) ----
export function deserializeScene(scene: Scene, json: string): any {
  const o = JSON.parse(json);
  if (!o || typeof o.version !== "number") throw new Error("Not a CSBryce scene file.");

  const clear = (a: unknown[]) => (a.length = 0);
  clear(scene.prims); clear(scene.lights); clear(scene.instances);
  clear(scene.blases); clear(scene.blasMatBase); clear(scene.blasMatCount); clear(scene.blasFile);
  clear(scene.meshMaterials); clear(scene.meshMaterialNames);
  clear(scene.meshTexLayers); clear(scene.meshNormalLayers); clear(scene.primImageLayers);

  Object.assign(scene.world, o.world);

  for (const sp of o.prims ?? []) {
    const p = new Primitive(sp.type);
    p.name = sp.name;
    p.position.set(sp.position[0], sp.position[1], sp.position[2]);
    p.rotation.set(sp.rotation[0], sp.rotation[1], sp.rotation[2]);
    p.a = sp.a; p.b = sp.b; p.c = sp.c;
    p.color = sp.color; p.reflectivity = sp.reflectivity;
    p.pattern = sp.pattern; p.colorB = sp.colorB; p.patternScale = sp.patternScale;
    p.bump = sp.bump; p.bumpScale = sp.bumpScale; p.imageLayer = sp.imageLayer;
    p.cornerRadius = sp.cornerRadius ?? 0; p.subtractive = !!sp.subtractive; p.group = sp.group ?? 0;
    scene.prims.push(p);
  }

  for (const sl of o.lights ?? []) {
    const l = new Light(sl.type);
    l.name = sl.name;
    l.direction.set(sl.direction[0], sl.direction[1], sl.direction[2]);
    l.position.set(sl.position[0], sl.position[1], sl.position[2]);
    l.color = sl.color; l.intensity = sl.intensity; l.softness = sl.softness; l.range = sl.range;
    l.bodyRadius = sl.bodyRadius ?? 0;
    l.inSky = sl.inSky ?? false;
    l.ringInner = sl.ringInner ?? 1.5; l.ringOuter = sl.ringOuter ?? 2.4;
    l.ringOpacity = sl.ringOpacity ?? 0; l.ringTilt = sl.ringTilt ?? 0.45;
    l.ringColor = sl.ringColor ?? [0.82, 0.75, 0.6];
    scene.lights.push(l);
  }

  const mesh = o.mesh ?? {};
  scene.meshTexSize = mesh.texSize ?? 256;
  for (const b64 of mesh.blases ?? []) {
    const raw = b64ToF32(b64);
    const { nodes, tris, nodeCount } = buildBVH(raw, TRI_STRIDE);
    scene.blases.push({ tris, triCount: tris.length / TRI_STRIDE, nodes, nodeCount });
    scene.blasFile.push(null);
  }
  scene.blasMatBase.push(...(mesh.blasMatBase ?? []));
  scene.blasMatCount.push(...(mesh.blasMatCount ?? []));
  scene.meshMaterials.push(...(mesh.materials ?? []));
  scene.meshMaterialNames.push(...(mesh.materialNames ?? []));
  for (const s of mesh.texLayers ?? []) scene.meshTexLayers.push(b64ToU8(s));
  for (const s of mesh.normalLayers ?? []) scene.meshNormalLayers.push(b64ToU8(s));
  for (const si of mesh.instances ?? []) {
    const m = new MeshInstance(si.blasIndex, si.name);
    m.position.set(si.position[0], si.position[1], si.position[2]);
    m.rotation.set(si.rotation[0], si.rotation[1], si.rotation[2]);
    m.scale = si.scale; m.subtractive = !!si.subtractive; m.group = si.group ?? 0;
    if (si.landform) m.landform = si.landform; // restore procedural-landform recipe + seed
    scene.instances.push(m);
  }

  const pi = o.primImages ?? { size: 256, layers: [] };
  scene.primImageSize = pi.size ?? 256;
  for (const s of pi.layers ?? []) scene.primImageLayers.push(b64ToU8(s));

  scene.meshStructVersion++;
  scene.instanceVersion++;
  scene.worldVersion++;
  scene.primTexVersion++;
  scene.meshMatVersion++;
  scene.touch();
  return o;
}

// ---- textured raster preview rebuilt from a BLAS + the material pool ----
function texFromBytes(bytes: Uint8Array<ArrayBuffer>, srgb: boolean): DataTexture {
  const t = new DataTexture(bytes, 256, 256, RGBAFormat);
  if (srgb) t.colorSpace = SRGBColorSpace;
  t.needsUpdate = true;
  return t;
}

/** Multi-material, textured preview group for a baked BLAS (boolean / converted
 *  prim / landscape). Decodes per-vertex colors and per-material groups. */
export function previewFromBlas(scene: Scene, blasIndex: number): Group {
  const blas = scene.blases[blasIndex];
  const base = scene.blasMatBase[blasIndex] ?? 0;
  const count = Math.max(1, scene.materialCount(blasIndex));
  const t = blas.tris;
  const n = blas.triCount;

  const order = [...Array(n).keys()].sort((a, b) => t[a * TRI_STRIDE + 30] - t[b * TRI_STRIDE + 30]);
  const pos = new Float32Array(n * 9);
  const nor = new Float32Array(n * 9);
  const uv = new Float32Array(n * 6);
  const col = new Float32Array(n * 9);
  const groups: { start: number; count: number; mat: number }[] = [];
  let cur = -1;
  const decode = (packed: number, d: number) => {
    const p = packed | 0;
    col[d] = (p & 255) / 255; col[d + 1] = ((p >> 8) & 255) / 255; col[d + 2] = ((p >> 16) & 255) / 255;
  };
  for (let ti = 0; ti < n; ti++) {
    const o = order[ti] * TRI_STRIDE;
    const mid = Math.max(0, (t[o + 30] | 0) - base);
    if (mid !== cur) { groups.push({ start: ti * 3, count: 0, mat: mid }); cur = mid; }
    groups[groups.length - 1].count += 3;
    const cw = [t[o + 19], t[o + 23], t[o + 31]];
    for (let v = 0; v < 3; v++) {
      const d3 = (ti * 3 + v) * 3;
      const d2 = (ti * 3 + v) * 2;
      pos[d3] = t[o + v * 4]; pos[d3 + 1] = t[o + v * 4 + 1]; pos[d3 + 2] = t[o + v * 4 + 2];
      nor[d3] = t[o + 12 + v * 4]; nor[d3 + 1] = t[o + 12 + v * 4 + 1]; nor[d3 + 2] = t[o + 12 + v * 4 + 2];
      uv[d2] = t[o + 24 + v * 2]; uv[d2 + 1] = t[o + 24 + v * 2 + 1];
      decode(cw[v], d3);
    }
  }
  const geom = new BufferGeometry();
  geom.setAttribute("position", new Float32BufferAttribute(pos, 3));
  geom.setAttribute("normal", new Float32BufferAttribute(nor, 3));
  geom.setAttribute("uv", new Float32BufferAttribute(uv, 2));
  geom.setAttribute("color", new Float32BufferAttribute(col, 3));
  for (const g of groups) geom.addGroup(g.start, g.count, g.mat);

  const mats: MeshStandardMaterial[] = [];
  for (let k = 0; k < count; k++) {
    const d = scene.materialDescriptor(blasIndex, k);
    const m = new MeshStandardMaterial({ vertexColors: true, metalness: d.metalness, roughness: d.roughness });
    m.color.setRGB(d.color[0], d.color[1], d.color[2]);
    if (d.tex) m.map = texFromBytes(d.tex, true);
    if (d.normal) m.normalMap = texFromBytes(d.normal, false);
    mats.push(m);
  }

  const group = new Group();
  group.add(new Mesh(geom, mats.length === 1 ? mats[0] : mats));
  return group;
}
