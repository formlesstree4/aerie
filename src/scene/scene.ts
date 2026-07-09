import { Vector3, Euler, Quaternion, Matrix4 } from "three";
import { TERRAIN_CATALOG, BASIS_ID, FRACTAL_ID, type TerrainKind, type LandformSpec } from "../gen/recipe";

// Keep these in sync with the WGSL structs in raytrace.wgsl.
export const MAX_PRIMS = 64;
export const MAX_LIGHTS = 16;
export const MAX_BOOL_GROUPS = 8; // boolean (CSG) groups: a carver only cuts solids in its group
// mat4x4 (16) + shape (4) + mat0 (4) + mat1 (4) + mat2 (4) + flags (4)
const PRIM_FLOATS = 36;
export const LIGHT_FLOATS = 20; // data + color + vec + ring0 + ring1 (5 vec4)

export enum PrimType {
  Sphere = 0,
  Box = 1,
  Torus = 2,
  Cylinder = 3,
  Cone = 4,
  Octahedron = 5,
  Capsule = 6,
  Pyramid = 7,
  RoundedBox = 8,
}

export enum LightType {
  Directional = 0,
  Point = 1,
}

export enum PrimPattern {
  Solid = 0,
  Checker = 1,
  Noise = 2,
  Wood = 3,
  Marble = 4,
  Gradient = 5,
  Image = 6,
}

export const PATTERN_LABELS: Record<PrimPattern, string> = {
  [PrimPattern.Solid]: "Solid",
  [PrimPattern.Checker]: "Checker",
  [PrimPattern.Noise]: "Noise",
  [PrimPattern.Wood]: "Wood",
  [PrimPattern.Marble]: "Marble",
  [PrimPattern.Gradient]: "Gradient",
  [PrimPattern.Image]: "Image",
};

export const PRIM_LABELS: Record<PrimType, string> = {
  [PrimType.Sphere]: "Sphere",
  [PrimType.Box]: "Cube",
  [PrimType.Torus]: "Torus",
  [PrimType.Cylinder]: "Cylinder",
  [PrimType.Cone]: "Cone",
  [PrimType.Octahedron]: "Octahedron",
  [PrimType.Capsule]: "Capsule",
  [PrimType.Pyramid]: "Pyramid",
  [PrimType.RoundedBox]: "Rounded Box",
};

let nextId = 1;

/** Shape params a/b/c carry per-type dimensions (radius, half-extents, etc.). */
export class Primitive {
  readonly id = nextId++;
  name: string;
  position = new Vector3();
  rotation = new Euler();
  a = 5;
  b = 5;
  c = 5;
  color: [number, number, number] = [0.8, 0.8, 0.85]; // primary / pattern color A
  reflectivity = 0;
  // Material
  pattern = PrimPattern.Solid;
  colorB: [number, number, number] = [0.2, 0.2, 0.25];
  patternScale = 4;
  bump = 0; // 0..1 surface-relief amount
  bumpScale = 6;
  imageLayer = -1; // index into the primitive image-texture array
  cornerRadius = 0; // rounding radius (rounded box)
  // Boolean CSG
  subtractive = false; // "negative": carves volume out of solids in the same group
  group = 0; // 0 = ungrouped; 1..MAX_BOOL_GROUPS = boolean group membership

  private readonly mat = new Matrix4();
  private readonly quat = new Quaternion();
  private static readonly ONE = new Vector3(1, 1, 1);

  constructor(public readonly type: PrimType) {
    this.name = `${PRIM_LABELS[type]} ${this.id}`;
    switch (type) {
      case PrimType.Sphere: this.a = 5; break;
      case PrimType.Box: this.a = 4; this.b = 4; this.c = 4; break;
      case PrimType.Torus: this.a = 5; this.b = 1.6; break;
      case PrimType.Cylinder: this.a = 3; this.b = 5; break;
      case PrimType.Cone: this.a = 4; this.b = 5; break;
      case PrimType.Octahedron: this.a = 5; break;
      case PrimType.Capsule: this.a = 3; this.b = 4; break; // radius, half-height of the shaft
      case PrimType.Pyramid: this.a = 5; this.b = 5; break; // base half-width, half-height
      case PrimType.RoundedBox: this.a = 4; this.b = 4; this.c = 4; this.cornerRadius = 1; break;
    }
  }

  /** Write PRIM_FLOATS (matrix, shape, material, flags) at offset. */
  write(out: Float32Array, offset: number): void {
    this.quat.setFromEuler(this.rotation);
    this.mat.compose(this.position, this.quat, Primitive.ONE).invert();
    out.set(this.mat.elements, offset); // column-major, matches WGSL mat4x4f
    out[offset + 16] = this.type;
    out[offset + 17] = this.a;
    out[offset + 18] = this.b;
    out[offset + 19] = this.c;
    // mat0: color A + reflectivity
    out[offset + 20] = this.color[0];
    out[offset + 21] = this.color[1];
    out[offset + 22] = this.color[2];
    out[offset + 23] = this.reflectivity;
    // mat1: color B + pattern type
    out[offset + 24] = this.colorB[0];
    out[offset + 25] = this.colorB[1];
    out[offset + 26] = this.colorB[2];
    out[offset + 27] = this.pattern;
    // mat2: pattern scale, bump, bump scale, image layer
    out[offset + 28] = this.patternScale;
    out[offset + 29] = this.bump;
    out[offset + 30] = this.bumpScale;
    out[offset + 31] = this.imageLayer;
    // flags: csg subtractive, corner radius, boolean group
    out[offset + 32] = this.subtractive ? 1 : 0;
    out[offset + 33] = this.cornerRadius;
    out[offset + 34] = this.group;
    out[offset + 35] = 0;
  }

  /** Independent copy preserving the id (for the undo/redo history). */
  clone(): Primitive {
    const p = new Primitive(this.type);
    (p as { id: number }).id = this.id;
    p.name = this.name;
    p.position.copy(this.position);
    p.rotation.copy(this.rotation);
    p.a = this.a; p.b = this.b; p.c = this.c;
    p.color = [...this.color] as [number, number, number];
    p.reflectivity = this.reflectivity;
    p.pattern = this.pattern;
    p.colorB = [...this.colorB] as [number, number, number];
    p.patternScale = this.patternScale;
    p.bump = this.bump; p.bumpScale = this.bumpScale;
    p.imageLayer = this.imageLayer;
    p.cornerRadius = this.cornerRadius;
    p.subtractive = this.subtractive;
    p.group = this.group;
    return p;
  }
}

export class Light {
  readonly id = nextId++;
  name: string;
  direction = new Vector3(0.5, 0.55, 0.3).normalize(); // toward the light
  position = new Vector3(20, 30, 10);
  color: [number, number, number] = [1, 1, 1];
  intensity: number;
  softness: number; // angular/area softness for shadows
  range = 120; // point-light falloff distance
  bodyRadius = 0; // visible glowing sphere (planet); 0 = invisible point light
  // When true, the planet lives in the "skybox": it is pushed far into the
  // distance (like the sun) so nothing in the scene can ever collide with it,
  // and it casts directional light. `position` then acts as its sky direction.
  inSky = false;
  // Planet rings (only when bodyRadius > 0 and ringOpacity > 0).
  ringInner = 1.5; // × bodyRadius
  ringOuter = 2.4; // × bodyRadius
  ringOpacity = 0; // 0 = no rings; Saturn ≈ 0.9, Jupiter ≈ 0.25
  ringTilt = 0.45; // radians
  ringColor: [number, number, number] = [0.82, 0.75, 0.6];

  constructor(public readonly type: LightType) {
    if (type === LightType.Directional) {
      this.name = `Sun ${this.id}`;
      this.intensity = 3;
      this.softness = 0.02;
    } else {
      this.name = `Point ${this.id}`;
      this.intensity = 800;
      this.softness = 0.05;
    }
  }

  write(out: Float32Array, offset: number): void {
    out[offset + 0] = this.type;
    out[offset + 1] = this.intensity;
    out[offset + 2] = this.softness;
    out[offset + 3] = this.range;
    out[offset + 4] = this.color[0];
    out[offset + 5] = this.color[1];
    out[offset + 6] = this.color[2];
    out[offset + 7] = this.bodyRadius;
    const v = this.type === LightType.Directional ? this.direction : this.position;
    out[offset + 8] = v.x;
    out[offset + 9] = v.y;
    out[offset + 10] = v.z;
    out[offset + 11] = this.inSky ? 1 : 0; // sky/distant flag (planets only)
    // ring0: inner, outer, opacity, tilt
    out[offset + 12] = this.ringInner;
    out[offset + 13] = this.ringOuter;
    out[offset + 14] = this.ringOpacity;
    out[offset + 15] = this.ringTilt;
    // ring1: color
    out[offset + 16] = this.ringColor[0];
    out[offset + 17] = this.ringColor[1];
    out[offset + 18] = this.ringColor[2];
    out[offset + 19] = 0;
  }

  /** Independent copy preserving the id (for the undo/redo history). */
  clone(): Light {
    const l = new Light(this.type);
    (l as { id: number }).id = this.id;
    l.name = this.name;
    l.direction.copy(this.direction);
    l.position.copy(this.position);
    l.color = [...this.color] as [number, number, number];
    l.intensity = this.intensity;
    l.softness = this.softness;
    l.range = this.range;
    l.bodyRadius = this.bodyRadius;
    l.inSky = this.inSky;
    l.ringInner = this.ringInner;
    l.ringOuter = this.ringOuter;
    l.ringOpacity = this.ringOpacity;
    l.ringTilt = this.ringTilt;
    l.ringColor = [...this.ringColor] as [number, number, number];
    return l;
  }
}

export type Selectable = Primitive | Light | MeshInstance;

// ---- clouds: stacked 2D layers, each a named meteorological type ----
export const MAX_CLOUD_LAYERS = 4;

// The ten meteorological genera. New values are appended (not reordered) so
// existing saved scenes keep their cloud types. The numeric value is the
// `ctype` the shader branches on.
export enum CloudType {
  Cirrus = 0,
  Cirrocumulus = 1,
  Cumulus = 2,
  Cumulonimbus = 3,
  Stratus = 4,
  Cirrostratus = 5,
  Altocumulus = 6,
  Altostratus = 7,
  Nimbostratus = 8,
  Stratocumulus = 9,
}

export const CLOUD_LABELS: Record<CloudType, string> = {
  [CloudType.Cirrus]: "Cirrus",
  [CloudType.Cirrocumulus]: "Cirrocumulus",
  [CloudType.Cumulus]: "Cumulus",
  [CloudType.Cumulonimbus]: "Cumulonimbus",
  [CloudType.Stratus]: "Stratus",
  [CloudType.Cirrostratus]: "Cirrostratus",
  [CloudType.Altocumulus]: "Altocumulus",
  [CloudType.Altostratus]: "Altostratus",
  [CloudType.Nimbostratus]: "Nimbostratus",
  [CloudType.Stratocumulus]: "Stratocumulus",
};

export interface CloudLayer {
  type: CloudType;
  coverage: number; // 0..1 how much of the sky it fills
  density: number; // 0..1 opacity
  altitude: number; // 0 (low) .. 1 (high) — drives parallax
  scale: number; // feature size multiplier
  seed: number; // noise offset — re-roll to randomize the shapes
  color: [number, number, number]; // lit (top) cloud color
  thickness: number; // 0..1 vertical body → drives self-shadowing / volume
  darkness: number; // 0..1 base brightness (low = dark stormy underside)
}

/** Per-type starting parameters; the seed randomizes the noise field. */
export function makeCloudLayer(type: CloudType): CloudLayer {
  const seed = Math.random() * 100;
  const white: [number, number, number] = [1, 0.99, 0.97];
  switch (type) {
    case CloudType.Cirrus:
      return { type, coverage: 0.4, density: 0.3, altitude: 0.95, scale: 1, seed, color: white, thickness: 0.12, darkness: 0.8 };
    case CloudType.Cirrocumulus:
      return { type, coverage: 0.5, density: 0.4, altitude: 0.85, scale: 1, seed, color: white, thickness: 0.2, darkness: 0.75 };
    case CloudType.Cumulus:
      return { type, coverage: 0.5, density: 0.7, altitude: 0.5, scale: 1, seed, color: white, thickness: 0.55, darkness: 0.5 };
    case CloudType.Cumulonimbus:
      return { type, coverage: 0.75, density: 0.97, altitude: 0.35, scale: 0.8, seed, color: [0.62, 0.64, 0.69], thickness: 1, darkness: 0.18 };
    case CloudType.Stratus:
      return { type, coverage: 0.85, density: 0.6, altitude: 0.2, scale: 0.7, seed, color: [0.72, 0.74, 0.78], thickness: 0.4, darkness: 0.45 };
    case CloudType.Cirrostratus: // high, thin, near-uniform veil (halo sky)
      return { type, coverage: 0.85, density: 0.18, altitude: 0.9, scale: 1.3, seed, color: white, thickness: 0.1, darkness: 0.85 };
    case CloudType.Altocumulus: // mid-level rounded clumps / "mackerel sky"
      return { type, coverage: 0.55, density: 0.5, altitude: 0.6, scale: 0.85, seed, color: white, thickness: 0.25, darkness: 0.62 };
    case CloudType.Altostratus: // mid, featureless translucent grey sheet
      return { type, coverage: 0.8, density: 0.45, altitude: 0.55, scale: 1.4, seed, color: [0.8, 0.82, 0.85], thickness: 0.3, darkness: 0.5 };
    case CloudType.Nimbostratus: // low-mid thick dark rain layer
      return { type, coverage: 0.95, density: 0.95, altitude: 0.3, scale: 1, seed, color: [0.5, 0.52, 0.56], thickness: 0.7, darkness: 0.22 };
    case CloudType.Stratocumulus: // low lumpy rolls with breaks
      return { type, coverage: 0.7, density: 0.6, altitude: 0.25, scale: 0.7, seed, color: [0.78, 0.8, 0.84], thickness: 0.35, darkness: 0.5 };
  }
}

/** Editable procedural-world parameters (the Bryce "labs"). */
export interface World {
  terrainAmp: number;
  terrainFreq: number;
  terrainRidge: number;
  terrainOffset: number;
  terrainOctaves: number; // fbm detail (1..8)
  terrainWarp: number; // domain-warp strength (world units) for natural ridgelines
  terrainEnabled: boolean;
  waterEnabled: boolean;
  cloudsEnabled: boolean;
  zenith: [number, number, number];
  horizon: [number, number, number];
  sunSize: number; // 0 = pinpoint, 1 = large
  cloudLayers: CloudLayer[]; // up to MAX_CLOUD_LAYERS
  hazeDensity: number;
  waterColor: [number, number, number];
  waterReflectivity: number; // Fresnel F0
  exposure: number;
  warmth: number;
  giBounces: number;      // indirect diffuse bounces (0 = fast direct-only/legacy look)
  aperture: number;       // thin-lens radius (world units); 0 = pinhole (DoF off)
  focusDistance: number;  // distance to the sharp focal plane (world units)
  timeOfDay: number;      // 0..24 hours; drives the sun arc + sky tint (applyTimeOfDay)
  starsEnabled: boolean;
  starDensity: number; // 0..1
  starBrightness: number;
  starSize: number; // 0..1
  starTwinkle: number; // 0..1
  starColor: [number, number, number];
  starVariation: number; // 0..1 cool↔warm color spread
  starClustering: number; // 0..1 spatial clumping
  starSeed: number; // re-roll to regenerate the star field
  terrainLow: [number, number, number]; // ground / valley color
  terrainRock: [number, number, number]; // slope color
  terrainHigh: [number, number, number]; // peak color
  // Composable-recipe block (mirrors src/gen on the GPU). The active world
  // recipe drives the infinite background terrain's noise.
  terrainRecipe: TerrainKind; // last preset applied (UI selection)
  terrainBasis: number;       // layer 0 basis: 0 value, 1 gradient, 2 worley
  terrainFractal: number;     // layer 0 fractal: 0 fbm, 1 ridged, 2 billow
  terrainBasis2: number;      // layer 1 basis
  terrainFractal2: number;    // layer 1 fractal
  terrainFreq2: number;       // layer 1 frequency relative to layer 0
  terrainOctaves2: number;    // layer 1 octaves
  terrainWeight2: number;     // layer 1 blend weight (0 = single layer)
  terrainTerraceSteps: number;
  terrainTerraceSharp: number;
  terrainWarpFreq: number;    // domain-warp frequency
  terrainSnowLine: number;    // normalized altitude where snow begins
  terrainSlopeRock: number;   // slope where rock takes over
  terrainSeed: number;        // re-roll to pan into a different region of the noise field
}

export function defaultWorld(): World {
  return {
    terrainAmp: 22,
    terrainFreq: 0.012,
    terrainRidge: 1.5,
    terrainOffset: -4,
    terrainOctaves: 7,
    terrainWarp: 0,
    terrainEnabled: true,
    waterEnabled: true,
    cloudsEnabled: true,
    zenith: [0.16, 0.33, 0.72],
    horizon: [0.78, 0.84, 0.92],
    sunSize: 0.5,
    cloudLayers: [makeCloudLayer(CloudType.Cumulus)],
    hazeDensity: 0.0016,
    waterColor: [0.02, 0.07, 0.09],
    waterReflectivity: 0.02,
    exposure: 1.05,
    warmth: 0.5,
    giBounces: 2,
    aperture: 0,
    focusDistance: 60,
    timeOfDay: 12,
    starsEnabled: false,
    starDensity: 0.4,
    starBrightness: 1,
    starSize: 0.5,
    starTwinkle: 0.5,
    starColor: [0.9, 0.93, 1.0],
    starVariation: 0.5,
    starClustering: 0.3,
    starSeed: 0,
    terrainLow: [0.2, 0.32, 0.14],
    terrainRock: [0.34, 0.3, 0.27],
    terrainHigh: [0.92, 0.94, 0.97],
    terrainRecipe: "mountain",
    terrainBasis: BASIS_ID.gradient,
    terrainFractal: FRACTAL_ID.fbm,
    terrainBasis2: BASIS_ID.gradient,
    terrainFractal2: FRACTAL_ID.fbm,
    terrainFreq2: 1,
    terrainOctaves2: 7,
    terrainWeight2: 0,
    terrainTerraceSteps: 0,
    terrainTerraceSharp: 0,
    terrainWarpFreq: 0.5,
    terrainSnowLine: 0.62,
    terrainSlopeRock: 0.4,
    terrainSeed: 0,
  };
}

/** An imported model, baked to world-space triangles + a BVH for the tracer. */
export const TRI_FLOATS = 32; // floats per triangle (8 vec4)
export const MAT_FLOATS = 12; // floats per material (3 vec4)
export const INST_FLOATS = 20; // invModel (16) + info vec4 (nodeBase, triBase, _, _)
const TRI_MATID_OFFSET = 30; // uv2m.z within a triangle

/** Bottom-level acceleration structure: one imported model in LOCAL space. */
export interface BLAS {
  tris: Float32Array<ArrayBuffer>; // 32 floats/triangle, BVH leaf order, global matId baked
  triCount: number;
  nodes: Float32Array<ArrayBuffer>; // 8 floats/node
  nodeCount: number;
}

/** A freshly imported model, before it joins the scene's global pools. */
export interface RawModel {
  name: string;
  blas: BLAS;
  materials: number[]; // 12/material, local matIds + local layer ids
  materialNames: string[]; // one per material, for the inspector
  texLayers: Uint8Array<ArrayBuffer>[]; // one per base-color texture
  normalLayers: Uint8Array<ArrayBuffer>[]; // one per normal map
  texSize: number;
  halfHeight: number; // scaled half-height; offsets spawn so the base rests there
}

/** A placed instance of a BLAS — pickable and movable like a primitive. */
export class MeshInstance {
  readonly id = nextId++;
  position = new Vector3();
  rotation = new Euler();
  scale = 1;
  // Boolean (CSG) membership — same scheme as Primitive; resolved by baking.
  subtractive = false;
  group = 0;
  // Set on procedurally-generated landforms so they can be re-authored in place.
  landform?: LandformSpec;

  private readonly mat = new Matrix4();
  private readonly quat = new Quaternion();
  private readonly scl = new Vector3();

  constructor(
    public blasIndex: number,
    public name: string,
  ) {}

  /** Write the world→local inverse model matrix (16 floats) at offset. */
  writeInvModel(out: Float32Array, offset: number): void {
    this.quat.setFromEuler(this.rotation);
    this.scl.setScalar(this.scale);
    this.mat.compose(this.position, this.quat, this.scl).invert();
    out.set(this.mat.elements, offset);
  }

  /** Object→world model matrix (for vertex editing transforms). */
  modelMatrix(out: Matrix4): Matrix4 {
    this.quat.setFromEuler(this.rotation);
    this.scl.setScalar(this.scale);
    return out.compose(this.position, this.quat, this.scl);
  }

  /** Independent copy preserving the id (for the undo/redo history). */
  clone(): MeshInstance {
    const m = new MeshInstance(this.blasIndex, this.name);
    (m as { id: number }).id = this.id;
    m.position.copy(this.position);
    m.rotation.copy(this.rotation);
    m.scale = this.scale;
    m.subtractive = this.subtractive;
    m.group = this.group;
    if (this.landform) m.landform = structuredClone(this.landform);
    return m;
  }
}

/** A point-in-time copy of the scene's editable state (see captureState). */
export interface SceneState {
  prims: Primitive[];
  lights: Light[];
  instances: MeshInstance[];
  world: World;
  blases: BLAS[];
  blasMatBase: number[];
  blasMatCount: number[];
  blasFile: ({ name: string; bytes: Uint8Array<ArrayBuffer> } | null)[];
  meshMaterials: number[];
  meshMaterialNames: string[];
  meshTexLayers: Uint8Array<ArrayBuffer>[];
  meshNormalLayers: Uint8Array<ArrayBuffer>[];
  meshTexSize: number;
  primImageLayers: Uint8Array<ArrayBuffer>[];
  primImageSize: number;
}

export class Scene {
  readonly prims: Primitive[] = [];
  readonly lights: Light[] = [];
  /** Bumped on any structural or property change; the renderer watches it. */
  version = 0;

  // Mesh data: per-model BLASes + global material/texture pools + instances.
  readonly blases: BLAS[] = [];
  readonly blasMatBase: number[] = []; // global material offset per BLAS (for re-baking)
  readonly blasMatCount: number[] = []; // material count per BLAS (for PBR overrides)
  // Original imported file per BLAS (for save/load: re-import restores rig + textured
  // preview). null for procedurally baked BLASes (boolean / converted prim / landscape).
  readonly blasFile: ({ name: string; bytes: Uint8Array<ArrayBuffer> } | null)[] = [];
  readonly instances: MeshInstance[] = [];
  readonly meshMaterials: number[] = []; // flat, MAT_FLOATS each, global
  readonly meshMaterialNames: string[] = []; // one per material (global)
  readonly meshTexLayers: Uint8Array<ArrayBuffer>[] = [];
  readonly meshNormalLayers: Uint8Array<ArrayBuffer>[] = [];
  meshTexSize = 256;
  /** Bumped when BLAS/material/texture pools change (import/remove model). */
  meshStructVersion = 0;
  /** Bumped on a cheap material-only edit (PBR override) — re-uploads materials. */
  meshMatVersion = 0;
  /** Bumped when instances are added/removed or transformed. */
  instanceVersion = 0;

  // Procedural world (terrain / sky / water / render) parameters.
  readonly world = defaultWorld();
  worldVersion = 0;

  // Uploaded image textures available to primitives (triplanar-mapped).
  readonly primImageLayers: Uint8Array<ArrayBuffer>[] = [];
  primImageSize = 256;
  primTexVersion = 0;

  readonly primData = new Float32Array(MAX_PRIMS * PRIM_FLOATS);
  readonly lightData = new Float32Array(MAX_LIGHTS * LIGHT_FLOATS);

  touch(): void {
    this.version++;
  }

  /** Mark instance transforms dirty (e.g. after dragging one). */
  touchInstances(): void {
    this.instanceVersion++;
    this.touch();
  }

  /** Mark world (terrain/sky/water/render) parameters dirty. */
  touchWorld(): void {
    this.worldVersion++;
  }

  /** Register an uploaded image as a primitive texture layer; returns its index. */
  addPrimImage(layer: Uint8Array<ArrayBuffer>): number {
    const index = this.primImageLayers.length;
    this.primImageLayers.push(layer);
    this.primTexVersion++;
    return index;
  }

  /** Append a cloud layer (capped at MAX_CLOUD_LAYERS); returns it, or null if full. */
  addCloudLayer(type: CloudType): CloudLayer | null {
    if (this.world.cloudLayers.length >= MAX_CLOUD_LAYERS) return null;
    const l = makeCloudLayer(type);
    this.world.cloudLayers.push(l);
    this.world.cloudsEnabled = true;
    this.touchWorld();
    return l;
  }

  removeCloudLayer(layer: CloudLayer): void {
    const i = this.world.cloudLayers.indexOf(layer);
    if (i >= 0) this.world.cloudLayers.splice(i, 1);
    this.touchWorld();
  }

  /** Re-roll a layer's noise offset to randomize its shapes. */
  randomizeCloudLayer(layer: CloudLayer): void {
    layer.seed = Math.random() * 100;
    this.touchWorld();
  }

  /** Re-roll the star field's pattern. */
  randomizeStars(): void {
    this.world.starSeed = Math.random() * 100;
    this.touchWorld();
  }

  /** Re-roll the background terrain into a different region of the noise field. */
  randomizeTerrain(): void {
    this.world.terrainSeed = Math.floor(Math.random() * 1000);
    this.touchWorld();
  }

  /** Apply a composable terrain recipe to the infinite background terrain. */
  applyTerrainRecipe(kind: TerrainKind): void {
    const r = TERRAIN_CATALOG[kind];
    const w = this.world;
    const l0 = r.layers[0];
    const l1 = r.layers[1];
    w.terrainRecipe = kind;
    w.terrainAmp = r.amplitude;
    w.terrainFreq = 0.012 * l0.freq;     // recipe-space freq → world scale
    w.terrainOctaves = l0.octaves;
    w.terrainRidge = 1;                   // fractal mode handles crest sharpening now
    w.terrainWarp = r.warpStrength * 120; // recipe-space displacement → world units
    w.terrainWarpFreq = r.warpFreq;
    w.terrainBasis = BASIS_ID[l0.basis];
    w.terrainFractal = FRACTAL_ID[l0.fractal];
    if (l1) {
      w.terrainBasis2 = BASIS_ID[l1.basis];
      w.terrainFractal2 = FRACTAL_ID[l1.fractal];
      w.terrainFreq2 = l1.freq / l0.freq;
      w.terrainOctaves2 = l1.octaves;
      w.terrainWeight2 = l1.weight / (l0.weight + l1.weight);
    } else {
      w.terrainBasis2 = w.terrainBasis;
      w.terrainFractal2 = w.terrainFractal;
      w.terrainFreq2 = 1;
      w.terrainOctaves2 = l0.octaves;
      w.terrainWeight2 = 0;
    }
    w.terrainTerraceSteps = r.terraceSteps;
    w.terrainTerraceSharp = r.terraceSharpness;
    w.terrainSnowLine = r.palette.snowLine;
    w.terrainSlopeRock = r.palette.slopeRock;
    w.terrainLow = [...r.palette.low];
    w.terrainRock = [...r.palette.rock];
    w.terrainHigh = [...r.palette.high];
    this.touchWorld();
  }

  /** Apply a terrain biome (palette + wetness). */
  applyBiome(name: "alpine" | "desert" | "tundra" | "badlands"): void {
    const w = this.world;
    switch (name) {
      case "alpine":
        w.terrainLow = [0.2, 0.32, 0.14]; w.terrainRock = [0.34, 0.3, 0.27]; w.terrainHigh = [0.92, 0.94, 0.97];
        w.waterEnabled = true;
        break;
      case "desert":
        w.terrainLow = [0.78, 0.66, 0.42]; w.terrainRock = [0.62, 0.44, 0.28]; w.terrainHigh = [0.88, 0.79, 0.58];
        w.waterEnabled = false;
        break;
      case "tundra":
        w.terrainLow = [0.55, 0.58, 0.5]; w.terrainRock = [0.4, 0.4, 0.43]; w.terrainHigh = [0.95, 0.96, 0.98];
        w.waterEnabled = true;
        break;
      case "badlands":
        w.terrainLow = [0.56, 0.31, 0.2]; w.terrainRock = [0.4, 0.22, 0.16]; w.terrainHigh = [0.72, 0.56, 0.4];
        w.waterEnabled = false;
        break;
    }
    this.touchWorld();
  }

  /** Apply a sky/lighting preset to the world and the primary sun light. */
  applySkyPreset(name: "day" | "sunset" | "night" | "overcast"): void {
    const w = this.world;
    const sun = this.lights.find((l) => l.type === LightType.Directional);
    const set = (
      zen: [number, number, number], hor: [number, number, number],
      dir: [number, number, number], col: [number, number, number],
      intensity: number, exposure: number, layers: CloudLayer[],
      stars: boolean, haze: number,
    ) => {
      w.zenith = zen; w.horizon = hor; w.exposure = exposure;
      w.cloudLayers = layers; w.cloudsEnabled = layers.length > 0;
      w.starsEnabled = stars; w.hazeDensity = haze;
      if (sun) {
        sun.direction.set(dir[0], dir[1], dir[2]).normalize();
        sun.color = col;
        sun.intensity = intensity;
      }
    };
    const layer = (type: CloudType, over: Partial<CloudLayer>): CloudLayer =>
      ({ ...makeCloudLayer(type), ...over });
    switch (name) {
      case "day":
        set([0.16, 0.33, 0.72], [0.78, 0.84, 0.92], [0.4, 0.7, 0.3], [1, 0.96, 0.9], 3.2, 1.05,
          [layer(CloudType.Cumulus, { coverage: 0.4 })], false, 0.0016);
        break;
      case "sunset":
        set([0.2, 0.2, 0.42], [0.98, 0.46, 0.22], [0.75, 0.1, 0.2], [1.0, 0.5, 0.25], 2.4, 1.1,
          [layer(CloudType.Cirrus, { coverage: 0.55 }), layer(CloudType.Cumulus, { coverage: 0.3 })], false, 0.0035);
        break;
      case "night":
        set([0.01, 0.02, 0.06], [0.04, 0.06, 0.12], [0.3, 0.25, 0.2], [0.5, 0.62, 0.95], 0.3, 1.5,
          [], true, 0.001);
        break;
      case "overcast":
        set([0.52, 0.55, 0.6], [0.72, 0.74, 0.77], [0.3, 0.6, 0.2], [0.85, 0.85, 0.88], 1.2, 1.0,
          [layer(CloudType.Stratus, { coverage: 0.95 })], false, 0.004);
        break;
    }
    w.timeOfDay = name === "day" ? 12 : name === "sunset" ? 17.5 : name === "night" ? 23 : 11;
    this.touchWorld();
    this.touch(); // sun light changed
  }

  /**
   * Continuous time-of-day: from one clock value (0..24h) derive the sun's
   * direction along its arc, its colour + intensity, and retint the sky
   * (zenith/horizon) so sun and sky always agree. Leaves clouds/water/haze as
   * the user set them.
   */
  applyTimeOfDay(hours: number): void {
    const w = this.world;
    w.timeOfDay = ((hours % 24) + 24) % 24;
    const sun = this.lights.find((l) => l.type === LightType.Directional);

    const ALT_MAX = 1.30; // sun altitude at local noon (~74°)
    const alt = ALT_MAX * Math.sin(((w.timeOfDay - 6) / 12) * Math.PI);
    const az = ((w.timeOfDay - 12) / 6) * (Math.PI / 2); // east at dawn → west at dusk
    const ca = Math.cos(alt);
    const dir: [number, number, number] = [ca * Math.sin(az), Math.sin(alt), ca * Math.cos(az)];
    const up = dir[1]; // sun height: -1 midnight .. ~0.96 noon

    const clamp01 = (x: number) => Math.min(1, Math.max(0, x));
    const smooth = (a: number, b: number, x: number) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };
    const mix3 = (a: number[], b: number[], t: number): [number, number, number] =>
      [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

    const day = smooth(-0.04, 0.35, up);                              // 0 night → 1 high sun
    const gold = smooth(-0.10, 0.06, up) * (1 - smooth(0.06, 0.40, up)); // golden band near horizon
    const night = 1 - smooth(-0.12, 0.04, up);                        // 1 below horizon

    const NIGHT_ZEN = [0.012, 0.02, 0.06], NIGHT_HOR = [0.03, 0.05, 0.11];
    const DAY_ZEN = [0.16, 0.33, 0.72], DAY_HOR = [0.78, 0.84, 0.92];
    const GOLD_ZEN = [0.26, 0.24, 0.40], GOLD_HOR = [0.98, 0.46, 0.22];

    let zen = mix3(NIGHT_ZEN, DAY_ZEN, day);
    let hor = mix3(NIGHT_HOR, DAY_HOR, day);
    zen = mix3(zen, GOLD_ZEN, gold * 0.6);
    hor = mix3(hor, GOLD_HOR, gold);
    w.zenith = zen;
    w.horizon = hor;
    w.starsEnabled = up < -0.04;
    w.exposure = 1.05 + night * 0.4 + gold * 0.05;

    if (sun) {
      sun.direction.set(dir[0], dir[1], dir[2]).normalize();
      const daySun = 3.4 * clamp01((up + 0.12) / 1.0);
      sun.intensity = Math.max(daySun, night * 0.22); // moonlight floor at night
      const warmWhite = mix3([1.0, 0.5, 0.25], [1.0, 0.97, 0.92], smooth(0.04, 0.42, up));
      sun.color = mix3(warmWhite, [0.5, 0.62, 0.95], night); // → cool moonlight after dark
    }
    this.touchWorld();
    this.touch();
  }

  /** Number of editable materials in a model. */
  materialCount(blasIndex: number): number {
    return this.blasMatCount[blasIndex] ?? 0;
  }

  /** Read one material's editable fields. */
  getMaterial(blasIndex: number, mi: number): {
    name: string;
    color: [number, number, number];
    metalness: number;
    roughness: number;
    hidden: boolean;
    texLayer: number;
  } {
    const gi = (this.blasMatBase[blasIndex] ?? 0) + mi;
    const o = gi * MAT_FLOATS;
    const m = this.meshMaterials;
    return {
      name: this.meshMaterialNames[gi] ?? `material ${mi + 1}`,
      color: [m[o], m[o + 1], m[o + 2]],
      metalness: m[o + 8],
      roughness: m[o + 9],
      hidden: m[o + 7] > 0.5,
      texLayer: m[o + 10],
    };
  }

  /** Patch one material's editable fields and re-upload (cheap). */
  setMaterial(
    blasIndex: number,
    mi: number,
    f: {
      color?: [number, number, number];
      metalness?: number;
      roughness?: number;
      hidden?: boolean;
      texLayer?: number;
    },
  ): void {
    const o = ((this.blasMatBase[blasIndex] ?? 0) + mi) * MAT_FLOATS;
    const m = this.meshMaterials;
    if (f.color) { m[o] = f.color[0]; m[o + 1] = f.color[1]; m[o + 2] = f.color[2]; }
    if (f.metalness !== undefined) m[o + 8] = f.metalness;
    if (f.roughness !== undefined) m[o + 9] = f.roughness;
    if (f.hidden !== undefined) m[o + 7] = f.hidden ? 1 : 0;
    if (f.texLayer !== undefined) m[o + 10] = f.texLayer;
    this.meshMatVersion++;
    this.touch();
  }

  /** Full descriptor for a material (incl. texture bytes) — used to rebuild
   *  materials when baking a boolean result. */
  materialDescriptor(blasIndex: number, mi: number): {
    name: string;
    color: [number, number, number];
    metalness: number;
    roughness: number;
    tex: Uint8Array<ArrayBuffer> | null;
    normal: Uint8Array<ArrayBuffer> | null;
  } {
    const gi = (this.blasMatBase[blasIndex] ?? 0) + mi;
    const o = gi * MAT_FLOATS;
    const m = this.meshMaterials;
    const texIdx = m[o + 10];
    const nrmIdx = m[o + 11];
    return {
      name: this.meshMaterialNames[gi] ?? `material ${mi + 1}`,
      color: [m[o], m[o + 1], m[o + 2]],
      metalness: m[o + 8],
      roughness: m[o + 9],
      tex: texIdx >= 0 ? this.meshTexLayers[texIdx] : null,
      normal: nrmIdx >= 0 ? this.meshNormalLayers[nrmIdx] : null,
    };
  }

  /** Append a base-color texture to the mesh pool; returns its layer index.
   *  The layer must be `meshTexSize`² RGBA bytes. */
  addMeshTexture(layer: Uint8Array<ArrayBuffer>): number {
    const index = this.meshTexLayers.length;
    this.meshTexLayers.push(layer);
    this.meshStructVersion++; // rebuilds the texture array (+ materials)
    this.touch();
    return index;
  }

  /** Copy metalness/roughness across every material of a model. */
  applyModelPBR(blasIndex: number, metalness: number, roughness: number): void {
    const count = this.materialCount(blasIndex);
    for (let m = 0; m < count; m++) this.setMaterial(blasIndex, m, { metalness, roughness });
  }

  /** Swap in a freshly baked BLAS (e.g. a new animation pose). */
  replaceBlas(blasIndex: number, blas: BLAS): void {
    this.blases[blasIndex] = blas;
    this.blasFile[blasIndex] = null; // edited/baked → no longer matches the source file
    this.meshStructVersion++;
    this.touch();
  }

  /** Add an imported model: append it to the pools and place one instance. */
  addModel(model: RawModel, spawn: Vector3): MeshInstance {
    const matBase = this.meshMaterials.length / MAT_FLOATS;
    const layerBase = this.meshTexLayers.length;
    const normalBase = this.meshNormalLayers.length;

    // Rebase the model's local material ids and texture layers into the pools.
    const tris = model.blas.tris;
    for (let i = 0; i < tris.length; i += TRI_FLOATS) tris[i + TRI_MATID_OFFSET] += matBase;
    for (let m = 0; m < model.materials.length; m += MAT_FLOATS) {
      if (model.materials[m + 10] >= 0) model.materials[m + 10] += layerBase; // base-color layer
      if (model.materials[m + 11] >= 0) model.materials[m + 11] += normalBase; // normal layer
    }
    this.meshMaterials.push(...model.materials);
    for (const nm of model.materialNames) this.meshMaterialNames.push(nm);
    for (const layer of model.texLayers) this.meshTexLayers.push(layer);
    for (const layer of model.normalLayers) this.meshNormalLayers.push(layer);
    this.meshTexSize = model.texSize;

    const blasIndex = this.blases.length;
    this.blases.push(model.blas);
    this.blasMatBase.push(matBase);
    this.blasMatCount.push(model.materials.length / MAT_FLOATS);
    this.blasFile.push(null); // set by the caller for re-importable file imports

    const inst = new MeshInstance(blasIndex, model.name);
    inst.position.copy(spawn);
    inst.position.y += model.halfHeight; // geometry is centered → lift so base rests at spawn
    this.instances.push(inst);

    this.meshStructVersion++;
    this.instanceVersion++;
    this.touch();
    return inst;
  }

  addPrim(type: PrimType, at?: Vector3): Primitive {
    const p = new Primitive(type);
    if (at) p.position.copy(at);
    p.position.y = Math.max(p.position.y, p.a); // sit roughly above the water
    this.prims.push(p);
    this.touch();
    return p;
  }

  addLight(type: LightType, at?: Vector3): Light {
    const l = new Light(type);
    if (at && type === LightType.Point) l.position.copy(at).setY(at.y + 20);
    this.lights.push(l);
    this.touch();
    return l;
  }

  remove(item: Selectable): void {
    if (item instanceof Primitive) {
      const i = this.prims.indexOf(item);
      if (i >= 0) this.prims.splice(i, 1);
    } else if (item instanceof MeshInstance) {
      const i = this.instances.indexOf(item);
      if (i >= 0) {
        this.instances.splice(i, 1);
        this.instanceVersion++; // pools kept; just drop the instance
      }
    } else {
      const i = this.lights.indexOf(item);
      if (i >= 0) this.lights.splice(i, 1);
    }
    this.touch();
  }

  /** Every pickable/movable item, in list order. */
  get selectables(): Selectable[] {
    return [...this.prims, ...this.instances, ...this.lights];
  }

  /** Append an independent copy of an item (fresh id; mesh copies share geometry). */
  addDuplicate(item: Selectable): Selectable {
    if (item instanceof Primitive) {
      const p = item.clone(); (p as { id: number }).id = nextId++;
      this.prims.push(p); this.touch(); return p;
    }
    if (item instanceof Light) {
      const l = item.clone(); (l as { id: number }).id = nextId++;
      this.lights.push(l); this.touch(); return l;
    }
    const m = item.clone(); (m as { id: number }).id = nextId++;
    this.instances.push(m); this.touchInstances(); return m;
  }

  /**
   * Point-in-time snapshot for undo/redo. Logical objects (prims/lights/
   * instances/world) are deep-copied; immutable mesh resources (BLAS geometry,
   * textures, file blobs) are shared by reference, so transform/param snapshots
   * stay cheap. meshMaterials is copied because it is edited in place.
   */
  captureState(): SceneState {
    return {
      prims: this.prims.map((p) => p.clone()),
      lights: this.lights.map((l) => l.clone()),
      instances: this.instances.map((m) => m.clone()),
      world: structuredClone(this.world),
      blases: this.blases.slice(),
      blasMatBase: this.blasMatBase.slice(),
      blasMatCount: this.blasMatCount.slice(),
      blasFile: this.blasFile.slice(),
      meshMaterials: this.meshMaterials.slice(),
      meshMaterialNames: this.meshMaterialNames.slice(),
      meshTexLayers: this.meshTexLayers.slice(),
      meshNormalLayers: this.meshNormalLayers.slice(),
      meshTexSize: this.meshTexSize,
      primImageLayers: this.primImageLayers.slice(),
      primImageSize: this.primImageSize,
    };
  }

  /** Restore a snapshot from captureState() and mark everything dirty. */
  restoreState(s: SceneState): void {
    const replace = <T>(arr: T[], items: T[]): void => { arr.length = 0; arr.push(...items); };
    replace(this.prims, s.prims.map((p) => p.clone()));
    replace(this.lights, s.lights.map((l) => l.clone()));
    replace(this.instances, s.instances.map((m) => m.clone()));
    Object.assign(this.world, structuredClone(s.world));
    replace(this.blases, s.blases.slice());
    replace(this.blasMatBase, s.blasMatBase.slice());
    replace(this.blasMatCount, s.blasMatCount.slice());
    replace(this.blasFile, s.blasFile.slice());
    replace(this.meshMaterials, s.meshMaterials.slice());
    replace(this.meshMaterialNames, s.meshMaterialNames.slice());
    replace(this.meshTexLayers, s.meshTexLayers.slice());
    replace(this.meshNormalLayers, s.meshNormalLayers.slice());
    this.meshTexSize = s.meshTexSize;
    replace(this.primImageLayers, s.primImageLayers.slice());
    this.primImageSize = s.primImageSize;
    this.meshStructVersion++;
    this.instanceVersion++;
    this.worldVersion++;
    this.primTexVersion++;
    this.meshMatVersion++;
    this.touch();
  }

  /** Flatten the scene into the GPU-bound typed arrays. */
  pack(): void {
    for (let i = 0; i < this.prims.length; i++) {
      this.prims[i].write(this.primData, i * PRIM_FLOATS);
    }
    for (let i = 0; i < this.lights.length; i++) {
      this.lights[i].write(this.lightData, i * LIGHT_FLOATS);
    }
  }

  /** Direction of the first directional light (drives the sky's sun disk). */
  get sunDir(): Vector3 {
    const sun = this.lights.find((l) => l.type === LightType.Directional);
    return sun ? sun.direction : new Vector3(0.5, 0.55, 0.3).normalize();
  }

  /** Color of the first directional light (drives the sun disk + glow). */
  get sunColor(): [number, number, number] {
    const sun = this.lights.find((l) => l.type === LightType.Directional);
    return sun ? sun.color : [1, 0.96, 0.9];
  }
}

/** Opening scene: keeps the familiar chrome sphere, plus a couple of demo objects. */
export function defaultScene(): Scene {
  const s = new Scene();

  const chrome = s.addPrim(PrimType.Sphere);
  chrome.position.set(38, 6, 18);
  chrome.a = 6;
  chrome.color = [0.95, 0.96, 0.98];
  chrome.reflectivity = 0.9;
  chrome.name = "Chrome Sphere";

  const torus = s.addPrim(PrimType.Torus);
  torus.position.set(14, 7, 6);
  torus.a = 6;
  torus.b = 2;
  torus.color = [0.75, 0.2, 0.18];
  torus.rotation.x = 0.6;

  const box = s.addPrim(PrimType.Box);
  box.position.set(2, 5, 22);
  box.a = box.b = box.c = 5;
  box.color = [0.2, 0.35, 0.7];
  box.rotation.y = 0.5;

  s.addLight(LightType.Directional);
  s.touch();
  return s;
}
