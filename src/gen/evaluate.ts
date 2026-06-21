import { fractal, terrace, radialMask, valueNoise } from "./noise";
import type { TerrainRecipe } from "./recipe";

// CPU evaluation of a TerrainRecipe. Mirrors the WGSL port used by the ray
// tracer for the active world recipe.

/** Domain-warp the sample point using low-frequency noise. */
function warp(recipe: TerrainRecipe, x: number, z: number, seed: number): [number, number] {
  if (recipe.warpStrength <= 0 || recipe.warpFreq <= 0) return [x, z];
  const wf = recipe.warpFreq;
  const wx = valueNoise(x * wf + 2.1, z * wf + 7.3, seed);
  const wz = valueNoise(x * wf + 8.7, z * wf + 1.9, seed + 17);
  return [x + (wx * 2 - 1) * recipe.warpStrength, z + (wz * 2 - 1) * recipe.warpStrength];
}

/**
 * Normalized height in [0,1] for a recipe at recipe-space coords x,z ∈ [-1,1].
 * Multiply by recipe.amplitude for world units.
 */
export function evaluateHeight(recipe: TerrainRecipe, x: number, z: number, seed: number): number {
  const [wx, wz] = warp(recipe, x, z, seed);

  let h = 0, total = 0;
  for (const l of recipe.layers) {
    const n = fractal(l.basis, l.fractal, wx * l.freq, wz * l.freq, seed, l.octaves, l.lacunarity, l.gain);
    h += n * l.weight;
    total += l.weight;
  }
  h = total > 0 ? h / total : h;

  h = terrace(h, recipe.terraceSteps, recipe.terraceSharpness);
  h *= radialMask(recipe.mask, x, z, recipe.maskRadius);
  return Math.min(1, Math.max(0, h));
}
