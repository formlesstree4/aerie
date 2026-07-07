// Numeric ids for packing a recipe into the GPU world uniform. The WGSL port in
// raytrace.wgsl uses the same constants.
export const BASIS_ID = { value: 0, gradient: 1, worley: 2 };
export const FRACTAL_ID = { fbm: 0, ridged: 1, billow: 2 };
// ---- palettes ---------------------------------------------------------------
const ALPINE = {
    low: [0.2, 0.32, 0.14], rock: [0.36, 0.31, 0.27], high: [0.92, 0.94, 0.97],
    sand: [0.55, 0.5, 0.36], snowLine: 0.62, slopeRock: 0.4,
};
const DESERT = {
    low: [0.78, 0.66, 0.42], rock: [0.62, 0.44, 0.28], high: [0.88, 0.79, 0.58],
    sand: [0.82, 0.71, 0.48], snowLine: 1.1, slopeRock: 0.45,
};
const CANYON = {
    low: [0.56, 0.31, 0.2], rock: [0.4, 0.22, 0.16], high: [0.72, 0.56, 0.4],
    sand: [0.6, 0.42, 0.3], snowLine: 1.1, slopeRock: 0.3,
};
const TAIGA = {
    low: [0.16, 0.26, 0.16], rock: [0.3, 0.3, 0.3], high: [0.9, 0.93, 0.96],
    sand: [0.4, 0.42, 0.34], snowLine: 0.45, slopeRock: 0.5,
};
const layer = (p) => ({
    octaves: 6, lacunarity: 2.03, gain: 0.5, weight: 1, ...p,
});
export const TERRAIN_CATALOG = {
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
export const TERRAIN_KINDS = Object.keys(TERRAIN_CATALOG);
export const cloneRecipe = (r) => structuredClone(r);
