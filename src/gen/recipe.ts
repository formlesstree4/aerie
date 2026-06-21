import type { NoiseBasis, FractalMode, MaskKind } from "./noise";

// Numeric ids for packing a recipe into the GPU world uniform. The WGSL port in
// raytrace.wgsl uses the same constants.
export const BASIS_ID: Record<NoiseBasis, number> = { value: 0, gradient: 1, worley: 2 };
export const FRACTAL_ID: Record<FractalMode, number> = { fbm: 0, ridged: 1, billow: 2 };

// A TerrainRecipe is a declarative description of a landform: a stack of noise
// layers (summed), an optional domain warp, terracing, a radial shaping mask,
// and a material palette. Both the CPU mesh baker (evaluate.ts) and — for the
// active world recipe — the GPU ray tracer interpret this same structure, so a
// new landform is *data*, not new code.

export interface NoiseLayer {
  basis: NoiseBasis;
  fractal: FractalMode;
  freq: number;        // base frequency in recipe space (x,z ∈ [-1,1] → freq)
  octaves: number;
  lacunarity: number;  // frequency multiplier per octave
  gain: number;        // amplitude multiplier per octave
  weight: number;      // contribution of this layer to the summed height
}

/** Vertex-color palette + thresholds for the height/slope shader. */
export interface Palette {
  low: [number, number, number];   // valley / ground
  rock: [number, number, number];  // steep slopes
  high: [number, number, number];  // peaks
  sand: [number, number, number];  // shoreline / base
  snowLine: number;                // normalized height where snow begins (0..1)
  slopeRock: number;               // slope where rock takes over (0..1)
}

export interface TerrainRecipe {
  name: string;
  layers: NoiseLayer[];
  warpFreq: number;        // domain-warp frequency (0 = off)
  warpStrength: number;    // domain-warp displacement
  terraceSteps: number;    // 0 = off
  terraceSharpness: number;
  mask: MaskKind;
  maskRadius: number;
  amplitude: number;       // world-units height scale
  palette: Palette;
}

// ---- palettes ---------------------------------------------------------------

const ALPINE: Palette = {
  low: [0.2, 0.32, 0.14], rock: [0.36, 0.31, 0.27], high: [0.92, 0.94, 0.97],
  sand: [0.55, 0.5, 0.36], snowLine: 0.62, slopeRock: 0.4,
};
const DESERT: Palette = {
  low: [0.78, 0.66, 0.42], rock: [0.62, 0.44, 0.28], high: [0.88, 0.79, 0.58],
  sand: [0.82, 0.71, 0.48], snowLine: 1.1, slopeRock: 0.45,
};
const CANYON: Palette = {
  low: [0.56, 0.31, 0.2], rock: [0.4, 0.22, 0.16], high: [0.72, 0.56, 0.4],
  sand: [0.6, 0.42, 0.3], snowLine: 1.1, slopeRock: 0.3,
};
const TAIGA: Palette = {
  low: [0.16, 0.26, 0.16], rock: [0.3, 0.3, 0.3], high: [0.9, 0.93, 0.96],
  sand: [0.4, 0.42, 0.34], snowLine: 0.45, slopeRock: 0.5,
};

const layer = (p: Partial<NoiseLayer> & Pick<NoiseLayer, "basis" | "fractal" | "freq">): NoiseLayer => ({
  octaves: 6, lacunarity: 2.03, gain: 0.5, weight: 1, ...p,
});

// ---- catalog ----------------------------------------------------------------

export type TerrainKind =
  | "mountain" | "hills" | "mesa" | "cliff" | "canyon" | "dunes" | "taiga";

export const TERRAIN_CATALOG: Record<TerrainKind, TerrainRecipe> = {
  mountain: {
    name: "Mountain",
    layers: [layer({ basis: "gradient", fractal: "ridged", freq: 1.6, octaves: 7 })],
    warpFreq: 0.6, warpStrength: 0.18,
    terraceSteps: 0, terraceSharpness: 0,
    mask: "peak", maskRadius: 1.0, amplitude: 28, palette: ALPINE,
  },
  hills: {
    name: "Hills",
    layers: [layer({ basis: "gradient", fractal: "fbm", freq: 1.4, octaves: 5 })],
    warpFreq: 0, warpStrength: 0,
    terraceSteps: 0, terraceSharpness: 0,
    mask: "none", maskRadius: 1, amplitude: 11, palette: ALPINE,
  },
  mesa: {
    name: "Mesa",
    layers: [layer({ basis: "gradient", fractal: "fbm", freq: 2.2, octaves: 4 })],
    warpFreq: 0, warpStrength: 0,
    terraceSteps: 4, terraceSharpness: 0.85,
    mask: "plateau", maskRadius: 0.62, amplitude: 16, palette: DESERT,
  },
  cliff: {
    name: "Cliff",
    layers: [
      layer({ basis: "gradient", fractal: "ridged", freq: 1.2, octaves: 6, weight: 0.7 }),
      layer({ basis: "worley", fractal: "fbm", freq: 2.0, octaves: 3, weight: 0.3 }),
    ],
    warpFreq: 0.5, warpStrength: 0.12,
    terraceSteps: 6, terraceSharpness: 0.95,
    mask: "none", maskRadius: 1, amplitude: 24, palette: CANYON,
  },
  canyon: {
    name: "Canyon",
    layers: [layer({ basis: "worley", fractal: "ridged", freq: 1.5, octaves: 5 })],
    warpFreq: 0.8, warpStrength: 0.25,
    terraceSteps: 5, terraceSharpness: 0.7,
    mask: "none", maskRadius: 1, amplitude: 20, palette: CANYON,
  },
  dunes: {
    name: "Dunes",
    layers: [layer({ basis: "gradient", fractal: "billow", freq: 1.1, octaves: 4 })],
    warpFreq: 0.4, warpStrength: 0.35,
    terraceSteps: 0, terraceSharpness: 0,
    mask: "none", maskRadius: 1, amplitude: 9, palette: DESERT,
  },
  taiga: {
    name: "Taiga",
    layers: [
      layer({ basis: "gradient", fractal: "fbm", freq: 1.3, octaves: 6, weight: 0.8 }),
      layer({ basis: "gradient", fractal: "ridged", freq: 3.0, octaves: 3, weight: 0.2 }),
    ],
    warpFreq: 0.3, warpStrength: 0.1,
    terraceSteps: 0, terraceSharpness: 0,
    mask: "none", maskRadius: 1, amplitude: 14, palette: TAIGA,
  },
};

export const TERRAIN_KINDS = Object.keys(TERRAIN_CATALOG) as TerrainKind[];

/** A placed landform: its source kind, an editable recipe, and a seed. Stored on
 *  the MeshInstance so a baked landform can be re-rolled / re-authored in place. */
export interface LandformSpec {
  kind: TerrainKind;
  recipe: TerrainRecipe;
  seed: number;
}

export const cloneRecipe = (r: TerrainRecipe): TerrainRecipe => structuredClone(r);
