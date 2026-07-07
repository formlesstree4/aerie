// Composable noise core shared (in spirit) by the CPU mesh baker and the GPU
// ray tracer. The WGSL port in raytrace.wgsl mirrors these algorithms so a
// recipe evaluated on the CPU matches the one evaluated on the GPU.
//
// All bases return values normalized to [0,1] unless noted. x,z are in the
// recipe's frequency space (already multiplied by layer frequency).
const TAU = Math.PI * 2;
export const lerp = (a, b, t) => a + (b - a) * t;
const fade = (t) => t * t * t * (t * (t * 6 - 15) + 10); // quintic
/** Deterministic 2D hash → [0,1). Seed offsets the lattice. */
export function hash2(x, z) {
    const s = Math.sin(x * 127.1 + z * 311.7) * 43758.5453;
    return s - Math.floor(s);
}
// ---- bases ------------------------------------------------------------------
/** Value noise — cheap, slightly blocky. Range [0,1]. */
export function valueNoise(x, z, seed) {
    const ix = Math.floor(x), iz = Math.floor(z);
    const fx = x - ix, fz = z - iz;
    const ux = fx * fx * (3 - 2 * fx), uz = fz * fz * (3 - 2 * fz);
    const a = hash2(ix + seed, iz), b = hash2(ix + 1 + seed, iz);
    const c = hash2(ix + seed, iz + 1), d = hash2(ix + 1 + seed, iz + 1);
    return a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz;
}
/** Gradient (Perlin) noise — smooth, no axis bias. Range [0,1]. */
export function gradientNoise(x, z, seed) {
    const ix = Math.floor(x), iz = Math.floor(z);
    const fx = x - ix, fz = z - iz;
    const dot = (cx, cz, dx, dz) => {
        const ang = hash2(cx + seed * 0.137, cz - seed * 0.179) * TAU;
        return Math.cos(ang) * dx + Math.sin(ang) * dz;
    };
    const u = fade(fx), v = fade(fz);
    const nx0 = lerp(dot(ix, iz, fx, fz), dot(ix + 1, iz, fx - 1, fz), u);
    const nx1 = lerp(dot(ix, iz + 1, fx, fz - 1), dot(ix + 1, iz + 1, fx - 1, fz - 1), u);
    return lerp(nx0, nx1, v) * 0.5 + 0.5;
}
/** Worley / cellular F1 distance. Range ~[0,1] (clamped). 0 at cell centers. */
export function worleyNoise(x, z, seed) {
    const ix = Math.floor(x), iz = Math.floor(z);
    let minD = 1e9;
    for (let di = -1; di <= 1; di++) {
        for (let dj = -1; dj <= 1; dj++) {
            const cx = ix + di, cz = iz + dj;
            const px = cx + hash2(cx + seed, cz);
            const pz = cz + hash2(cx, cz + seed);
            const dx = x - px, dz = z - pz;
            minD = Math.min(minD, dx * dx + dz * dz);
        }
    }
    return Math.min(1, Math.sqrt(minD));
}
export function sampleBasis(basis, x, z, seed) {
    if (basis === "gradient")
        return gradientNoise(x, z, seed);
    if (basis === "worley")
        return worleyNoise(x, z, seed);
    return valueNoise(x, z, seed);
}
/** Fractal sum of a basis. Returns ~[0,1]. */
export function fractal(basis, mode, x, z, seed, octaves, lacunarity, gain) {
    let sum = 0, amp = 0.5, norm = 0, fx = x, fz = z, f = 1;
    for (let i = 0; i < octaves; i++) {
        let n = sampleBasis(basis, fx * f, fz * f, seed + i * 31);
        if (mode === "ridged")
            n = 1 - Math.abs(n * 2 - 1); // sharp crests
        else if (mode === "billow")
            n = Math.abs(n * 2 - 1); // rounded lobes
        sum += amp * n;
        norm += amp;
        f *= lacunarity;
        amp *= gain;
    }
    return norm > 0 ? sum / norm : sum;
}
// ---- operators --------------------------------------------------------------
/** Quantize height into terraces with a smooth-ish riser. step in (0,1]. */
export function terrace(h, steps, sharpness) {
    if (steps <= 0)
        return h;
    const scaled = h * steps;
    const base = Math.floor(scaled);
    const frac = scaled - base;
    // sharpness 0 → linear (no terracing); 1 → hard steps
    const eased = lerp(frac, frac < 0.5 ? 0 : 1, Math.min(1, Math.max(0, sharpness)));
    return (base + eased) / steps;
}
export const smoothstep = (e0, e1, x) => {
    const t = Math.min(1, Math.max(0, (x - e0) / (e1 - e0)));
    return t * t * (3 - 2 * t);
};
/** Radial shaping mask for placeable landforms. x,z ∈ [-1,1]; r = radius cutoff. */
export function radialMask(kind, x, z, radius) {
    if (kind === "none")
        return 1;
    const r = Math.hypot(x, z);
    if (kind === "plateau") {
        const top = 1 - Math.min(1, r / radius);
        return top * top * (3 - 2 * top); // smoothstep dome → flat top
    }
    // peak: quadratic falloff
    const peak = Math.max(0, 1 - r / radius);
    return peak * peak;
}
