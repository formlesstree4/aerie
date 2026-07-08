// Faithful CPU port of the world-terrain height field in raytrace.wgsl
// (terrainHeight and its noise chain). Kept bit-for-bit in step with the shader
// so the raster preview's terrain mesh matches the ray-traced surface.
//
// NOTE: this is a DIFFERENT hash lineage from src/gen/noise.ts. The shader's
// terrain uses the fract-based `hash21`, while noise.ts uses a sin-based hash for
// placeable landform recipes. They are not interchangeable — sampling the wrong
// one yields a preview that is subtly, confusingly wrong rather than obviously
// flat. If you change terrainHeight in raytrace.wgsl, mirror it here.
const fract = (x) => x - Math.floor(x);
const mix = (a, b, t) => a + (b - a) * t;
const clampi = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const smoothstep = (e0, e1, x) => {
    const t = clampi((x - e0) / (e1 - e0), 0, 1);
    return t * t * (3 - 2 * t);
};
// --- hashing (mirrors WGSL hash21) ------------------------------------------
// p3 = fract(vec3(p.x, p.y, p.x) * 0.1031); p3 += dot(p3, p3.yzx + 33.33);
// return fract((p3.x + p3.y) * p3.z)
function hash21(px, py) {
    const a = fract(px * 0.1031);
    const b = fract(py * 0.1031);
    const c = a; // vec3(p.xyx) → third lane is p.x again
    const d = a * (b + 33.33) + b * (c + 33.33) + c * (a + 33.33);
    return fract((a + d + (b + d)) * (c + d));
}
// --- bases -------------------------------------------------------------------
function noise2(px, py) {
    const ix = Math.floor(px), iy = Math.floor(py);
    const fx = px - ix, fy = py - iy;
    const ux = fx * fx * (3 - 2 * fx);
    const uy = fy * fy * (3 - 2 * fy);
    const a = hash21(ix, iy);
    const b = hash21(ix + 1, iy);
    const c = hash21(ix, iy + 1);
    const d = hash21(ix + 1, iy + 1);
    return mix(mix(a, b, ux), mix(c, d, ux), uy);
}
function gradNoise2(px, py) {
    const ix = Math.floor(px), iy = Math.floor(py);
    const fx = px - ix, fy = py - iy;
    const ux = fx * fx * fx * (fx * (fx * 6 - 15) + 10);
    const uy = fy * fy * fy * (fy * (fy * 6 - 15) + 10);
    const dotg = (cx, cy, dx, dy) => {
        const ang = hash21(cx, cy) * 6.28318530718;
        return Math.cos(ang) * dx + Math.sin(ang) * dy;
    };
    const n00 = dotg(ix, iy, fx, fy);
    const n10 = dotg(ix + 1, iy, fx - 1, fy);
    const n01 = dotg(ix, iy + 1, fx, fy - 1);
    const n11 = dotg(ix + 1, iy + 1, fx - 1, fy - 1);
    return mix(mix(n00, n10, ux), mix(n01, n11, ux), uy) * 0.5 + 0.5;
}
function worleyNoise2(px, py) {
    const ix = Math.floor(px), iy = Math.floor(py);
    let minD = 1e9;
    for (let di = -1; di <= 1; di++) {
        for (let dj = -1; dj <= 1; dj++) {
            const cx = ix + di, cy = iy + dj;
            const ptx = cx + hash21(cx, cy);
            const pty = cy + hash21(cx + 13.7, cy + 13.7); // WGSL: hash21(c + 13.7) → both lanes
            const dx = px - ptx, dy = py - pty;
            minD = Math.min(minD, dx * dx + dy * dy);
        }
    }
    return Math.min(1, Math.sqrt(minD));
}
function basisSample(basis, px, py) {
    if (basis === 1)
        return gradNoise2(px, py);
    if (basis === 2)
        return worleyNoise2(px, py);
    return noise2(px, py);
}
// --- fbm (value noise + the WGSL rotation matrix, used only for domain warp) --
function fbm(px, py, octaves) {
    let x = px, y = py, amp = 0.5, sum = 0;
    for (let i = 0; i < octaves; i++) {
        sum += amp * noise2(x, y);
        const nx = 1.6 * x - 1.2 * y; // mat2x2(1.6,1.2,-1.2,1.6) * p (column-major)
        const ny = 1.2 * x + 1.6 * y;
        x = nx;
        y = ny;
        amp *= 0.5;
    }
    return sum;
}
// --- fractal + terrace -------------------------------------------------------
function fractalMode(basis, mode, px, py, octaves) {
    let x = px, y = py, amp = 0.5, sum = 0, norm = 0;
    for (let i = 0; i < octaves; i++) {
        let n = basisSample(basis, x, y);
        if (mode === 1)
            n = 1 - Math.abs(n * 2 - 1);
        else if (mode === 2)
            n = Math.abs(n * 2 - 1);
        sum += amp * n;
        norm += amp;
        x *= 2;
        y *= 2;
        amp *= 0.5;
    }
    return sum / norm;
}
function terraceFn(h, steps, sharp) {
    if (steps <= 0)
        return h;
    const scaled = h * steps;
    const base = Math.floor(scaled);
    const frac = scaled - base;
    const hard = frac >= 0.5 ? 1 : 0;
    const eased = mix(frac, hard, clampi(sharp, 0, 1));
    return (base + eased) / steps;
}
/** World-space terrain height at (x, z). Matches terrainHeight() in raytrace.wgsl. */
export function worldTerrainHeight(x, z, w) {
    let px = x, pz = z;
    const warp = w.terrainWarp;
    if (warp > 0.01) {
        const wf = w.terrainFreq * w.terrainWarpFreq;
        const wx = fbm(x * wf + 2.1, z * wf + 7.3, 4);
        const wz = fbm(x * wf + 8.7, z * wf + 1.9, 4);
        px += (wx * 2 - 1) * warp;
        pz += (wz * 2 - 1) * warp;
    }
    px += w.terrainSeed * 17.13;
    pz += w.terrainSeed * 31.71;
    const f = w.terrainFreq;
    const oct0 = Math.trunc(clampi(w.terrainOctaves, 1, 8));
    let h = fractalMode(w.terrainBasis, w.terrainFractal, px * f, pz * f, oct0);
    if (w.terrainWeight2 > 0.001) {
        const oct1 = Math.trunc(clampi(w.terrainOctaves2, 1, 8));
        const h2 = fractalMode(w.terrainBasis2, w.terrainFractal2, px * f * w.terrainFreq2, pz * f * w.terrainFreq2, oct1);
        h = mix(h, h2, w.terrainWeight2);
    }
    h = terraceFn(h, w.terrainTerraceSteps, w.terrainTerraceSharp);
    return Math.pow(h, w.terrainRidge) * w.terrainAmp + w.terrainOffset;
}
/** Terrain surface color at world (x, y, z) with world-up normal component `ny`.
 *  Mirrors terrainColor() in raytrace.wgsl: slope drives ground→rock, altitude
 *  (gated by non-steepness) drives →snow, a shoreline lift near y=0, and a subtle
 *  fbm mottle. Writes linear RGB into `out`. */
export function worldTerrainColor(x, y, z, ny, w, out) {
    const slope = 1 - ny;
    const amp = Math.abs(w.terrainAmp) > 1e-6 ? w.terrainAmp : 1e-6;
    const altitude = clampi((y - w.terrainOffset) / amp, 0, 1);
    const low = w.terrainLow, rock = w.terrainRock, high = w.terrainHigh;
    const sr = w.terrainSlopeRock, sl = w.terrainSnowLine;
    const t1 = smoothstep(sr - 0.15, sr + 0.15, slope);
    let r = mix(low[0], rock[0], t1), g = mix(low[1], rock[1], t1), b = mix(low[2], rock[2], t1);
    const t2 = smoothstep(sl, sl + 0.2, altitude) * (1 - smoothstep(sr, sr + 0.3, slope));
    r = mix(r, high[0], t2);
    g = mix(g, high[1], t2);
    b = mix(b, high[2], t2);
    const t3 = smoothstep(0, 0.06, y); // beach/flats lift at the shoreline
    r = mix(low[0] * 1.15, r, t3);
    g = mix(low[1] * 1.15, g, t3);
    b = mix(low[2] * 1.15, b, t3);
    const v = 0.85 + 0.3 * fbm(x * 0.4, z * 0.4, 3);
    out[0] = r * v;
    out[1] = g * v;
    out[2] = b * v;
}
