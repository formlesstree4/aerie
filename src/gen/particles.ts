// Stateless, time-parameterized particle model — the single source of truth both
// renderers consume. Every particle's state is a closed-form function of the
// scene clock `t`: given the same emitter + t you always get the same result, so
// the cutscene timeline can scrub/seek freely and the off-screen WebM export
// reproduces exactly what the preview showed. No frame-to-frame state is carried.
//
// The path tracer draws these as additive emissive blobs (see particleEmission in
// raytrace.wgsl); the Three.js preview draws them as additive points. Both read
// the same packed buffer this module produces.

/** The packed per-particle layout the GPU/preview consume (3 × vec4 = 12 floats):
 *   [0..3]  position.xyz, radius
 *   [4..7]  color.rgb (HDR/emissive), opacity
 *   [8..11] velocity.xyz, _pad   (velocity drives sub-frame motion blur)  */
export const PARTICLE_FLOATS = 12;
/** Hard cap on live particles across all emitters (keeps the per-ray loop cheap). */
export const MAX_PARTICLES = 1024;

export enum EmitterType {
  Campfire = 0,
  Explosion = 1,
  Smoke = 2,
  Sparks = 3,
  Fireworks = 4,
  Magic = 5,
}

export const EMITTER_LABELS: Record<EmitterType, string> = {
  [EmitterType.Campfire]: "Campfire",
  [EmitterType.Explosion]: "Explosion",
  [EmitterType.Smoke]: "Smoke Plume",
  [EmitterType.Sparks]: "Sparks / Fountain",
  [EmitterType.Fireworks]: "Fireworks",
  [EmitterType.Magic]: "Magic Wisp",
};

/** Structural view of an emitter the evaluator needs. The scene's Emitter class
 *  implements this (its Vector3 position satisfies the {x,y,z} shape). */
export interface EmitterSpec {
  type: EmitterType;
  position: { x: number; y: number; z: number };
  count: number; // particles alive per emitter
  size: number; // base particle radius (world units)
  speed: number; // initial velocity magnitude
  spread: number; // 0..1 directional/positional spread
  gravity: number; // downward acceleration (units/s²); negative = buoyant rise
  lifetime: number; // seconds a particle lives before it recycles
  intensity: number; // emission multiplier
  colorA: [number, number, number]; // young color (HDR ok — >1 glows through tonemap)
  colorB: [number, number, number]; // aged color
  seed: number; // re-roll to shuffle the particle field
  loop: boolean; // true = continuous stream; false = one-shot burst at burstTime
  burstTime: number; // seconds (one-shot archetypes)
}

// --- tiny deterministic hash rng (per-particle, stateless) ------------------
const fract = (x: number) => x - Math.floor(x);
const clamp01 = (x: number) => (x < 0 ? 0 : x > 1 ? 1 : x);
const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp01((x - e0) / (e1 - e0));
  return t * t * (3 - 2 * t);
};
const mix = (a: number, b: number, t: number) => a + (b - a) * t;
function hash(n: number): number {
  return fract(Math.sin(n) * 43758.5453);
}

/** Per-type starting parameters. A newly-added emitter is seeded from these, then
 *  the user tweaks them in the inspector. Kept here so archetypes stay one place. */
export function defaultEmitter(type: EmitterType): EmitterSpec {
  const base: EmitterSpec = {
    type,
    position: { x: 0, y: 0, z: 0 },
    count: 120,
    size: 0.5,
    speed: 6,
    spread: 0.5,
    gravity: 9,
    lifetime: 2,
    intensity: 1,
    colorA: [1.4, 1.1, 0.5],
    colorB: [0.6, 0.1, 0.03],
    seed: 1,
    loop: true,
    burstTime: 0,
  };
  switch (type) {
    case EmitterType.Campfire:
      return { ...base, count: 110, size: 0.7, speed: 7, spread: 0.45, gravity: 1.5,
        lifetime: 1.6, intensity: 1.4, colorA: [1.6, 1.2, 0.5], colorB: [0.5, 0.06, 0.02] };
    case EmitterType.Explosion:
      return { ...base, count: 220, size: 0.9, speed: 20, spread: 1, gravity: 11,
        lifetime: 1.4, intensity: 2.2, colorA: [2.2, 1.4, 0.5], colorB: [0.35, 0.06, 0.03],
        loop: false, burstTime: 0.5 };
    case EmitterType.Smoke:
      return { ...base, count: 90, size: 1.8, speed: 2.4, spread: 0.5, gravity: -1.2,
        lifetime: 4.5, intensity: 0.35, colorA: [0.5, 0.5, 0.52], colorB: [0.12, 0.12, 0.13] };
    case EmitterType.Sparks:
      return { ...base, count: 160, size: 0.14, speed: 14, spread: 0.7, gravity: 16,
        lifetime: 1.8, intensity: 2.6, colorA: [2.4, 1.6, 0.6], colorB: [1.0, 0.2, 0.03] };
    case EmitterType.Fireworks:
      return { ...base, count: 260, size: 0.22, speed: 22, spread: 1, gravity: 7,
        lifetime: 2.4, intensity: 3, colorA: [2.4, 1.8, 2.2], colorB: [0.3, 0.05, 0.5],
        loop: false, burstTime: 0.6 };
    case EmitterType.Magic:
      return { ...base, count: 80, size: 0.28, speed: 1.6, spread: 2.2, gravity: 0,
        lifetime: 3.2, intensity: 1.8, colorA: [0.5, 1.4, 2.0], colorB: [1.4, 0.4, 1.6] };
  }
}

// --- per-particle evaluation -------------------------------------------------
// Scratch vectors reused across particles to avoid per-call allocation.
const _p = { x: 0, y: 0, z: 0 };
const _v = { x: 0, y: 0, z: 0 };

/** Evaluate one particle `j` of emitter `e` at time `t` into (_p, _v) plus color
 *  and returns [radius, opacity] via the out2 array. Writes nothing when dead. */
function evalOne(
  e: EmitterSpec, j: number, t: number, col: [number, number, number], out2: [number, number],
): void {
  const s = e.seed * 127.1 + j * 311.7;
  const r1 = hash(s), r2 = hash(s + 1.7), r3 = hash(s + 3.3), r4 = hash(s + 5.9), r5 = hash(s + 8.1);

  // Age in [0,1). Looping streams stagger births by a per-particle phase; one-shot
  // archetypes clamp to a single burst window and die outside it.
  let age: number;
  if (e.loop) {
    age = fract(t / e.lifetime + r1);
  } else {
    const a = (t - e.burstTime) / e.lifetime;
    if (a < 0 || a >= 1) { out2[0] = 0; out2[1] = 0; return; }
    age = a;
  }
  const tau = age * e.lifetime;
  const px = e.position.x, py = e.position.y, pz = e.position.z;
  const g = e.gravity;

  // Random azimuth + a spread-controlled radial component shared by most types.
  const ang = r2 * 6.2831853;
  const ca = Math.cos(ang), sa = Math.sin(ang);
  let alpha = 1;
  let rad = e.size;

  switch (e.type) {
    case EmitterType.Campfire: {
      const up = e.speed * (0.7 + 0.6 * r3);
      const out = e.speed * 0.22 * e.spread * (0.3 + 0.7 * r4);
      // Flames lick: a lateral sway that grows as the particle rises.
      const sway = Math.sin(tau * 3.2 + r1 * 6.28) * e.spread * 0.6 * tau;
      _p.x = px + ca * out * tau + ca * sway;
      _p.y = py + up * tau - 0.5 * g * tau * tau;
      _p.z = pz + sa * out * tau + sa * sway;
      _v.x = ca * out; _v.y = up - g * tau; _v.z = sa * out;
      rad = e.size * (0.5 + 0.9 * (1 - age));
      alpha = smoothstep(0, 0.08, age) * (1 - smoothstep(0.55, 1, age));
      break;
    }
    case EmitterType.Explosion:
    case EmitterType.Fireworks: {
      // Radial burst on a unit sphere (r5 gives the vertical component).
      const cz = r5 * 2 - 1, sr = Math.sqrt(Math.max(0, 1 - cz * cz));
      const spd = e.speed * (0.35 + 0.65 * r3);
      const vx = ca * sr * spd, vy = cz * spd, vz = sa * sr * spd;
      _p.x = px + vx * tau;
      _p.y = py + vy * tau - 0.5 * g * tau * tau;
      _p.z = pz + vz * tau;
      _v.x = vx; _v.y = vy - g * tau; _v.z = vz;
      rad = e.size * (0.6 + 0.7 * (1 - age));
      alpha = (1 - age) * (1 - age); // trailing fade
      break;
    }
    case EmitterType.Smoke: {
      const up = e.speed * (0.6 + 0.8 * r3);
      const drift = e.spread * 1.2; // steady wind push on x
      const wobble = Math.sin(tau * 0.8 + r1 * 6.28) * e.spread;
      _p.x = px + drift * tau + wobble;
      _p.y = py + up * tau - 0.5 * g * tau * tau;
      _p.z = pz + ca * e.spread * (0.4 + r4) + Math.cos(tau * 0.7 + r2 * 6.28) * e.spread;
      _v.x = drift; _v.y = up - g * tau; _v.z = 0;
      rad = e.size * (0.5 + 1.6 * age); // puffs grow as they rise
      alpha = smoothstep(0, 0.15, age) * (1 - smoothstep(0.5, 1, age));
      break;
    }
    case EmitterType.Sparks: {
      const up = e.speed * (0.7 + 0.5 * r3);
      const out = e.speed * e.spread * (0.2 + 0.8 * r4);
      const vx = ca * out, vy = up, vz = sa * out;
      _p.x = px + vx * tau;
      _p.y = py + vy * tau - 0.5 * g * tau * tau;
      _p.z = pz + vz * tau;
      _v.x = vx; _v.y = vy - g * tau; _v.z = vz;
      rad = e.size * (0.7 + 0.6 * (1 - age));
      alpha = 1 - smoothstep(0.6, 1, age);
      break;
    }
    case EmitterType.Magic: {
      // Motes orbit the emitter on a slowly-drifting helix, bobbing vertically.
      const orbit = e.spread * (0.5 + 0.9 * r3);
      const w = e.speed * (0.6 + 0.5 * r4); // angular speed
      const phi = r1 * 6.2831853 + tau * w;
      const cw = Math.cos(phi), sw = Math.sin(phi);
      const bob = Math.sin(tau * 1.5 + r2 * 6.28) * e.spread * 0.4;
      _p.x = px + cw * orbit;
      _p.y = py + bob + tau * 0.15; // gentle overall rise
      _p.z = pz + sw * orbit;
      _v.x = -sw * orbit * w; _v.y = 0; _v.z = cw * orbit * w; // tangential
      rad = e.size * (0.6 + 0.4 * Math.sin(tau * 2 + r1 * 6.28));
      alpha = smoothstep(0, 0.12, age) * (1 - smoothstep(0.7, 1, age));
      break;
    }
  }

  const k = age; // color ages young→old
  col[0] = mix(e.colorA[0], e.colorB[0], k) * e.intensity;
  col[1] = mix(e.colorA[1], e.colorB[1], k) * e.intensity;
  col[2] = mix(e.colorA[2], e.colorB[2], k) * e.intensity;
  out2[0] = rad;
  out2[1] = alpha;
}

/** Pack every particle of emitter `e` at time `t` into `out` starting at float
 *  offset `off`. Returns the number of particles written (== e.count, clamped to
 *  the buffer). Dead one-shot particles are written with opacity 0 (skipped by
 *  the renderers) so the layout stays stable. */
export function evaluateEmitter(e: EmitterSpec, t: number, out: Float32Array, off: number): number {
  const col: [number, number, number] = [0, 0, 0];
  const out2: [number, number] = [0, 0];
  const maxJ = Math.min(e.count, (out.length - off) / PARTICLE_FLOATS);
  let n = 0;
  for (let j = 0; j < maxJ; j++) {
    evalOne(e, j, t, col, out2);
    const o = off + j * PARTICLE_FLOATS;
    out[o] = _p.x; out[o + 1] = _p.y; out[o + 2] = _p.z; out[o + 3] = out2[0];
    out[o + 4] = col[0]; out[o + 5] = col[1]; out[o + 6] = col[2]; out[o + 7] = out2[1];
    out[o + 8] = _v.x; out[o + 9] = _v.y; out[o + 10] = _v.z; out[o + 11] = 0;
    n++;
  }
  return n;
}
