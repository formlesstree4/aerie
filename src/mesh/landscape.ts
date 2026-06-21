import {
  BufferGeometry,
  Float32BufferAttribute,
  Uint32BufferAttribute,
  Mesh,
  MeshStandardMaterial,
  Group,
} from "three";
import { buildBVH } from "./bvh";
import { computeFaceTangents, TRI_STRIDE } from "./modelImport";
import type { RawModel } from "../scene/scene";
import { evaluateHeight } from "../gen/evaluate";
import { smoothstep } from "../gen/noise";
import {
  TERRAIN_CATALOG, TERRAIN_KINDS, cloneRecipe,
  type TerrainKind, type TerrainRecipe, type LandformSpec,
} from "../gen/recipe";

// Placeable landforms are now driven by the composable recipe catalog in
// src/gen. Adding a new landform is a data entry in TERRAIN_CATALOG.
export type LandscapeType = TerrainKind;
export const LANDSCAPE_LABELS: Record<LandscapeType, string> = Object.fromEntries(
  TERRAIN_KINDS.map((k) => [k, TERRAIN_CATALOG[k].name]),
) as Record<LandscapeType, string>;
export { TERRAIN_KINDS as LANDSCAPE_TYPES };

const randomSeed = () => Math.floor(Math.random() * 1000);

/** World half-extent of a baked landform mesh in local space (x,z ∈ [-H, H]).
 *  Exported so scatter/placement can map normalized coords back to local space. */
export const LANDSCAPE_HALF = 34;

/** A fresh authorable landform spec for a catalog kind (recipe copy + seed). */
export function defaultLandform(type: LandscapeType): LandformSpec {
  return { kind: type, recipe: cloneRecipe(TERRAIN_CATALOG[type]), seed: randomSeed() };
}

const mix3 = (a: number[], b: number[], t: number, o: number[]) => {
  o[0] = a[0] + (b[0] - a[0]) * t; o[1] = a[1] + (b[1] - a[1]) * t; o[2] = a[2] + (b[2] - a[2]) * t;
};
const packColor = (r: number, g: number, b: number) =>
  Math.round(Math.min(1, r) * 255) + Math.round(Math.min(1, g) * 255) * 256 + Math.round(Math.min(1, b) * 255) * 65536;

/** Generate a heightfield model (vertex-colored) from an explicit recipe + seed. */
export function landscapeModel(recipe: TerrainRecipe, seed: number): { model: RawModel; group: Group } {
  const pal = recipe.palette;
  const N = 96; // grid cells per side
  const half = LANDSCAPE_HALF; // world half-extent
  const amp = recipe.amplitude;
  const W = N + 1;
  const step = (half * 2) / N;

  const H = new Float32Array(W * W);
  for (let i = 0; i < W; i++) {
    for (let j = 0; j < W; j++) {
      H[i * W + j] = evaluateHeight(recipe, (i / N) * 2 - 1, (j / N) * 2 - 1, seed) * amp;
    }
  }

  const positions = new Float32Array(W * W * 3);
  const normals = new Float32Array(W * W * 3);
  const colors = new Float32Array(W * W * 3); // 0..1 for preview
  const colf = new Float32Array(W * W); // packed for the BLAS
  const c = [0, 0, 0];
  for (let i = 0; i < W; i++) {
    for (let j = 0; j < W; j++) {
      const k = i * W + j;
      positions[k * 3] = (i / N) * 2 * half - half;
      positions[k * 3 + 1] = H[k];
      positions[k * 3 + 2] = (j / N) * 2 * half - half;
      const hl = H[Math.max(0, i - 1) * W + j], hr = H[Math.min(N, i + 1) * W + j];
      const hd = H[i * W + Math.max(0, j - 1)], hu = H[i * W + Math.min(N, j + 1)];
      let nx = hl - hr, ny = 2 * step, nz = hd - hu;
      const len = Math.hypot(nx, ny, nz) || 1;
      nx /= len; ny /= len; nz /= len;
      normals[k * 3] = nx; normals[k * 3 + 1] = ny; normals[k * 3 + 2] = nz;
      // palette-driven shading: sand at base, grass→rock by slope, snow by altitude
      const t = H[k] / amp, slope = 1 - ny;
      mix3(pal.low, pal.rock, smoothstep(pal.slopeRock - 0.15, pal.slopeRock + 0.15, slope), c);
      mix3(c, pal.high, smoothstep(pal.snowLine, pal.snowLine + 0.2, t) * (1 - smoothstep(pal.slopeRock, pal.slopeRock + 0.3, slope)), c);
      mix3(pal.sand, c, smoothstep(0.02, 0.12, t), c);
      colors[k * 3] = c[0]; colors[k * 3 + 1] = c[1]; colors[k * 3 + 2] = c[2];
      colf[k] = packColor(c[0], c[1], c[2]);
    }
  }

  const index = new Uint32Array(N * N * 6);
  let ii = 0;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const a = i * W + j, b = (i + 1) * W + j, d = i * W + j + 1, e = (i + 1) * W + j + 1;
      index[ii++] = a; index[ii++] = b; index[ii++] = e;
      index[ii++] = a; index[ii++] = e; index[ii++] = d;
    }
  }

  // BLAS: de-index into our 32-float triangle layout, baking vertex colors.
  const triCount = index.length / 3;
  const raw = new Float32Array(triCount * TRI_STRIDE);
  for (let t = 0; t < triCount; t++) {
    const o = t * TRI_STRIDE;
    for (let v = 0; v < 3; v++) {
      const vi = index[t * 3 + v];
      raw[o + v * 4] = positions[vi * 3]; raw[o + v * 4 + 1] = positions[vi * 3 + 1]; raw[o + v * 4 + 2] = positions[vi * 3 + 2];
      raw[o + 12 + v * 4] = normals[vi * 3]; raw[o + 12 + v * 4 + 1] = normals[vi * 3 + 1]; raw[o + 12 + v * 4 + 2] = normals[vi * 3 + 2];
    }
    raw[o + 19] = colf[index[t * 3]];     // vertex-color 0
    raw[o + 23] = colf[index[t * 3 + 1]]; // vertex-color 1
    raw[o + 31] = colf[index[t * 3 + 2]]; // vertex-color 2
  }
  computeFaceTangents(raw);
  const { nodes, tris, nodeCount } = buildBVH(raw, TRI_STRIDE);

  const model: RawModel = {
    name: recipe.name,
    blas: { tris, triCount: tris.length / TRI_STRIDE, nodes, nodeCount },
    materials: [1, 1, 1, 1, 0, 0, 0, 0, 0, 0.92, -1, -1], // white base × vertex colors
    materialNames: [recipe.name],
    texLayers: [],
    normalLayers: [],
    texSize: 256,
    halfHeight: 0,
  };

  const geom = new BufferGeometry();
  geom.setAttribute("position", new Float32BufferAttribute(positions, 3));
  geom.setAttribute("normal", new Float32BufferAttribute(normals, 3));
  geom.setAttribute("color", new Float32BufferAttribute(colors, 3));
  geom.setIndex(new Uint32BufferAttribute(index, 1));
  const group = new Group();
  group.add(new Mesh(geom, new MeshStandardMaterial({ vertexColors: true, roughness: 0.92 })));
  return { model, group };
}
