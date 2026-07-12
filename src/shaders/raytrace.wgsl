// CSBryce — slice 1 ray tracer (now with an editable scene).
//
// A progressive compute ray tracer rendering a Bryce-style world: an fbm
// heightfield terrain, a volumetric sky (sun + clouds), a reflective Fresnel
// water plane, plus a dynamic list of SDF primitives and lights uploaded from
// the CPU. Each dispatch traces one new sample per pixel and accumulates it.

struct Uniforms {
  camPos    : vec3f, fovScale    : f32,  // fovScale = tan(fovY/2)
  camFwd    : vec3f, aspect      : f32,
  camRight  : vec3f, frame       : f32,  // 0-based accumulated sample index
  camUp     : vec3f, seed        : f32,
  sunDir    : vec3f, time        : f32,
  res       : vec2f, sampleCount : f32, sunBright : f32,  // sunBright = sun-disk brightness
  primCount : f32,  lightCount  : f32, instCount : f32, sunFlag : f32,
  pickX     : f32,  pickY       : f32, _pad3 : f32, _pad4 : f32,
  sunColor  : vec3f, starsFlag  : f32,
  aperture  : f32,  focusDist   : f32, giBounces : f32, shaftStr : f32,  // thin-lens DoF; giBounces = indirect diffuse bounces; shaftStr = god-ray strength
  partCount : f32,  shutter     : f32, _p0 : f32, _p1 : f32,  // particle count; shutter = motion-blur time (1/fps, 0 = none)
  partBound : vec3f, partRadius : f32,  // bounding sphere over all live particles (whole-cloud ray reject)
};

struct Tri {
  p0 : vec4f, p1 : vec4f, p2 : vec4f,   // positions (xyz)
  n0 : vec4f, n1 : vec4f, n2 : vec4f,   // normals (xyz); n0.w handedness, n1.w/n2.w packed vcolor 0/1
  uv01 : vec4f,                         // uv0.xy, uv1.xy
  uv2m : vec4f,                         // uv2.xy, materialId, packed vcolor 2
};

struct Node {
  lo : vec4f,  // xyz aabb min, w = leftFirst
  hi : vec4f,  // xyz aabb max, w = triangle count (0 => internal node)
};

struct Mtl {
  base   : vec4f,  // rgb albedo, a opacity
  emis   : vec4f,  // rgb emissive, w = hidden flag (1 → invisible)
  params : vec4f,  // metallic, roughness, texLayer, normalLayer
};

struct Inst {
  invModel : mat4x4f,  // world -> local
  info     : vec4f,    // nodeBase, triBase, _, _
};

struct Prim {
  invModel : mat4x4f,  // world -> local (rigid: rotation + translation only)
  shape    : vec4f,    // x = type, yzw = dimension params
  mat0     : vec4f,    // colorA.rgb, reflectivity
  mat1     : vec4f,    // colorB.rgb, pattern type
  mat2     : vec4f,    // patternScale, bump, bumpScale, imageLayer
  flags    : vec4f,    // x = negative (1), y = corner radius, z = boolean group
};

struct Light {
  data  : vec4f,  // x=type(0 dir,1 point), y=intensity, z=softness, w=range
  color : vec4f,  // rgb, w = body radius (planet; 0 = invisible)
  vec   : vec4f,  // dir toward light (directional) OR world position (point); w = sky flag
  ring0 : vec4f,  // ringInner, ringOuter (× radius), ringOpacity, ringTilt
  ring1 : vec4f,  // ring color rgb, _
};

// "Sky" planets are pushed this far away so nothing in the scene can collide
// with them; their body radius is scaled by SKY_BODY so they keep a usable
// on-screen size at that distance.
const SKY_DIST = 3000.0;
const SKY_BODY = 30.0; // radius multiplier (SKY_DIST / 100)

fn lightInSky(lt: Light) -> bool { return lt.vec.w > 0.5; }

// World-space center of a planet's visible body (far away when it's a sky body;
// for a sky body `vec.xyz` is interpreted as a direction).
fn lightCenter(lt: Light) -> vec3f {
  if (lightInSky(lt)) { return normalize(lt.vec.xyz) * SKY_DIST; }
  return lt.vec.xyz;
}

// World-space radius of a planet's visible body (and ring scale base).
fn lightRadius(lt: Light) -> f32 {
  if (lightInSky(lt)) { return lt.color.w * SKY_BODY; }
  return lt.color.w;
}

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var<storage, read_write> accum : array<vec4f>;
@group(0) @binding(2) var<storage, read> prims : array<Prim>;
@group(0) @binding(3) var<storage, read> lights : array<Light>;
@group(0) @binding(4) var<storage, read> triangles : array<Tri>;
@group(0) @binding(5) var<storage, read> nodes : array<Node>;
@group(0) @binding(6) var<storage, read> materials : array<Mtl>;
@group(0) @binding(7) var meshTex : texture_2d_array<f32>;
@group(0) @binding(8) var meshSamp : sampler;
@group(0) @binding(9) var<storage, read_write> pickResult : array<i32>;
@group(0) @binding(10) var<storage, read> instances : array<Inst>;
@group(0) @binding(11) var meshNormalTex : texture_2d_array<f32>;
@group(0) @binding(13) var primTex : texture_2d_array<f32>;
// Top-level acceleration structure: a BVH over instance world AABBs. Same node
// layout as a BLAS, but a leaf's `lo.w` (first) indexes `tlasOrder` (instance
// indices) rather than the triangle pool.
@group(0) @binding(14) var<storage, read> tlasNodes : array<Node>;
@group(0) @binding(15) var<storage, read> tlasOrder : array<u32>;
// Emissive particle field (campfires, explosions, …). Each particle is an
// additive glowing blob; see particleEmission(). p0 = pos.xyz + radius,
// p1 = color.rgb (HDR) + opacity, p2 = velocity.xyz + pad (for motion blur).
struct Particle { p0 : vec4f, p1 : vec4f, p2 : vec4f };
@group(0) @binding(16) var<storage, read> particles : array<Particle>;

struct CloudLayer {
  a : vec4f,  // type, coverage, density, scale
  b : vec4f,  // altitude, seed, thickness, darkness
  c : vec4f,  // color.rgb, _
};
struct World {
  terrain : vec4f,  // amp, freq, ridgeExp, heightOffset
  flags   : vec4f,  // terrainEnabled, waterEnabled, cloudsEnabled, cloudLayerCount
  zenith  : vec4f,  // rgb, sunExponent
  horizon : vec4f,  // rgb, _
  haze    : vec4f,  // density, terrainOctaves, terrainWarp, _
  water   : vec4f,  // rgb deep color, F0
  star0   : vec4f,  // density, brightness, size, twinkle
  star1   : vec4f,  // color.rgb, variation
  star2   : vec4f,  // clustering, seed, _, _
  tlow    : vec4f,  // terrain low (ground) color
  trock   : vec4f,  // terrain slope color
  thigh   : vec4f,  // terrain peak color
  rec0    : vec4f,  // basis0, fractal0, basis1, fractal1
  rec1    : vec4f,  // freq2 (relative), octaves1, weight2, terraceSteps
  rec2    : vec4f,  // terraceSharp, warpFreq, snowLine, slopeRock
  clouds  : array<CloudLayer, 4>,
};
@group(0) @binding(12) var<uniform> world : World;

const PI = 3.14159265359;
const MAX_BOUNCE = 5;
const WATER_Y    = 0.0;
const SHAFT_STEPS = 14;      // ray-march samples for volumetric light shafts
const SHAFT_MAX   = 420.0;   // how far into the air shafts are integrated
const SHAFT_G     = 0.72;    // forward-scatter anisotropy → beams glow toward the sun

// ---------------------------------------------------------------- hashing / rng
fn hash21(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.xyx) * 0.1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

fn hash31(p: vec3f) -> f32 {
  var p3 = fract(p * 0.1031);
  p3 += dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}

// Stellar-class palette: blue → white → yellow → orange → red.
fn starTint(t: f32) -> vec3f {
  if (t < 0.25) { return mix(vec3f(0.5, 0.66, 1.0), vec3f(1.0, 1.0, 1.0), t * 4.0); }
  if (t < 0.5)  { return mix(vec3f(1.0, 1.0, 1.0), vec3f(1.0, 0.95, 0.7), (t - 0.25) * 4.0); }
  if (t < 0.75) { return mix(vec3f(1.0, 0.95, 0.7), vec3f(1.0, 0.7, 0.4), (t - 0.5) * 4.0); }
  return mix(vec3f(1.0, 0.7, 0.4), vec3f(1.0, 0.46, 0.38), (t - 0.75) * 4.0);
}

// Round, parametric stars: one per sparse cell with soft radial falloff. Density,
// brightness, size, twinkle, base color, color variation, clustering and seed are
// all driven from world.star0/star1/star2.
fn stars(dir: vec3f) -> vec3f {
  if (dir.y <= 0.0) { return vec3f(0.0); }
  let density = world.star0.x;
  let brightness = world.star0.y;
  let sizeAmt = world.star0.z;
  let twinkleAmt = world.star0.w;
  let baseColor = world.star1.rgb;
  let variation = world.star1.w;
  let clustering = world.star2.x;
  let seed = world.star2.y;

  let scale = 90.0;
  let uv = dir * scale + seed * 19.7;
  let cell = floor(uv);
  let h = hash31(cell + seed);

  // Clustering: a low-freq field that strongly concentrates stars into clumps.
  let cf = fbm(vec2f(dir.x, dir.z) * 2.4 + vec2f(dir.y * 5.0 + seed, 1.7), 4);
  let clump = mix(1.0, smoothstep(0.34, 0.66, cf) * 2.4, clustering);
  let densEff = clamp(density * clump, 0.0, 1.3);
  let thr = clamp(mix(0.994, 0.78, densEff), 0.5, 0.999);
  if (h <= thr) { return vec3f(0.0); }

  let sub = vec3f(hash31(cell + 1.7), hash31(cell + 4.3), hash31(cell + 9.1));
  let d = length(fract(uv) - sub);
  let bright = (h - thr) / max(1.0 - thr, 1e-3);
  let radius = mix(0.02, 0.12, hash31(cell + 6.0)) * (0.6 + 0.4 * bright) * (0.45 + sizeAmt);
  let core = smoothstep(radius, 0.0, d);

  // Per-star twinkle (independent phase + speed) — dips strongly at full amount.
  let phase = hash31(cell + 12.0) * 6.2832;
  let speed = 1.0 + 4.0 * hash31(cell + 15.0);
  let flick = 0.5 + 0.5 * sin(u.time * speed + phase);
  let tw = mix(1.0, 0.12 + 0.88 * flick, twinkleAmt);

  // Color: base tint blended toward the per-star stellar color by `variation`.
  let tint = mix(baseColor, starTint(hash31(cell + 5.0)), variation);

  return tint * (core * core) * (0.4 + 0.6 * bright) * tw * brightness * smoothstep(0.0, 0.1, dir.y);
}

var<private> rngState : u32;
fn seedRng(px: vec2u) {
  rngState = (px.x * 1973u + px.y * 9277u + u32(u.frame) * 26699u) | 1u;
}
fn rand() -> f32 {
  rngState = rngState * 747796405u + 2891336453u;
  var w = ((rngState >> ((rngState >> 28u) + 4u)) ^ rngState) * 277803737u;
  w = (w >> 22u) ^ w;
  return f32(w) / 4294967295.0;
}

// ---------------------------------------------------------------- noise / fbm
fn noise2(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u2 = f * f * (3.0 - 2.0 * f);
  let a = hash21(i + vec2f(0.0, 0.0));
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u2.x), mix(c, d, u2.x), u2.y);
}

fn fbm(p0: vec2f, octaves: i32) -> f32 {
  var p = p0;
  var amp = 0.5;
  var sum = 0.0;
  let m = mat2x2f(1.6, 1.2, -1.2, 1.6);
  for (var i = 0; i < octaves; i = i + 1) {
    sum += amp * noise2(p);
    p = m * p;
    amp *= 0.5;
  }
  return sum;
}

// ---- composable recipe noise (mirrors src/gen/noise.ts) ----
fn grad2(c: vec2f) -> vec2f {
  let a = hash21(c) * 6.28318530718;
  return vec2f(cos(a), sin(a));
}
// Gradient (Perlin) noise → [0,1].
fn gradNoise2(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let n00 = dot(grad2(i + vec2f(0.0, 0.0)), f - vec2f(0.0, 0.0));
  let n10 = dot(grad2(i + vec2f(1.0, 0.0)), f - vec2f(1.0, 0.0));
  let n01 = dot(grad2(i + vec2f(0.0, 1.0)), f - vec2f(0.0, 1.0));
  let n11 = dot(grad2(i + vec2f(1.0, 1.0)), f - vec2f(1.0, 1.0));
  return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y) * 0.5 + 0.5;
}
// Worley / cellular F1 distance → ~[0,1].
fn worleyNoise2(p: vec2f) -> f32 {
  let ip = floor(p);
  var minD = 1e9;
  for (var di = -1; di <= 1; di = di + 1) {
    for (var dj = -1; dj <= 1; dj = dj + 1) {
      let c = ip + vec2f(f32(di), f32(dj));
      let pt = c + vec2f(hash21(c), hash21(c + 13.7));
      let d = p - pt;
      minD = min(minD, dot(d, d));
    }
  }
  return min(1.0, sqrt(minD));
}
fn basisSample(basis: i32, p: vec2f) -> f32 {
  if (basis == 1) { return gradNoise2(p); }
  if (basis == 2) { return worleyNoise2(p); }
  return noise2(p);
}
// Fractal sum with selectable basis + mode (0 fbm, 1 ridged, 2 billow) → ~[0,1].
fn fractalMode(basis: i32, mode: i32, p0: vec2f, octaves: i32) -> f32 {
  var p = p0;
  var amp = 0.5;
  var sum = 0.0;
  var norm = 0.0;
  for (var i = 0; i < octaves; i = i + 1) {
    var n = basisSample(basis, p);
    if (mode == 1) { n = 1.0 - abs(n * 2.0 - 1.0); }
    else if (mode == 2) { n = abs(n * 2.0 - 1.0); }
    sum += amp * n;
    norm += amp;
    p *= 2.0;
    amp *= 0.5;
  }
  return sum / norm;
}
fn terraceFn(h: f32, steps: f32, sharp: f32) -> f32 {
  if (steps <= 0.0) { return h; }
  let scaled = h * steps;
  let base = floor(scaled);
  let frac = scaled - base;
  let hard = select(0.0, 1.0, frac >= 0.5);
  let eased = mix(frac, hard, clamp(sharp, 0.0, 1.0));
  return (base + eased) / steps;
}

fn noise3(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let w = f * f * (3.0 - 2.0 * f);
  let c000 = hash31(i + vec3f(0.0, 0.0, 0.0));
  let c100 = hash31(i + vec3f(1.0, 0.0, 0.0));
  let c010 = hash31(i + vec3f(0.0, 1.0, 0.0));
  let c110 = hash31(i + vec3f(1.0, 1.0, 0.0));
  let c001 = hash31(i + vec3f(0.0, 0.0, 1.0));
  let c101 = hash31(i + vec3f(1.0, 0.0, 1.0));
  let c011 = hash31(i + vec3f(0.0, 1.0, 1.0));
  let c111 = hash31(i + vec3f(1.0, 1.0, 1.0));
  let x00 = mix(c000, c100, w.x);
  let x10 = mix(c010, c110, w.x);
  let x01 = mix(c001, c101, w.x);
  let x11 = mix(c011, c111, w.x);
  return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

fn fbm3(p0: vec3f, octaves: i32) -> f32 {
  var p = p0;
  var amp = 0.5;
  var sum = 0.0;
  for (var i = 0; i < octaves; i = i + 1) {
    sum += amp * noise3(p);
    p *= 2.03;
    amp *= 0.5;
  }
  return sum;
}

// ---------------------------------------------------------------- terrain
fn terrainHeight(xz: vec2f) -> f32 {
  var p = xz;
  let warp = world.haze.z;
  if (warp > 0.01) {
    // Domain warp: nudge the sample point by low-frequency noise for sinuous ridges.
    let wf = world.terrain.y * world.rec2.y;
    let wx = fbm(xz * wf + vec2f(2.1, 7.3), 4);
    let wz = fbm(xz * wf + vec2f(8.7, 1.9), 4);
    p += (vec2f(wx, wz) * 2.0 - 1.0) * warp;
  }
  p += vec2f(world.haze.w * 17.13, world.haze.w * 31.71); // seed: pan into a different region
  let f = world.terrain.y;
  let oct0 = i32(clamp(world.haze.y, 1.0, 8.0));
  var h = fractalMode(i32(world.rec0.x), i32(world.rec0.y), p * f, oct0);
  let weight2 = world.rec1.z;
  if (weight2 > 0.001) {
    let oct1 = i32(clamp(world.rec1.y, 1.0, 8.0));
    let h2 = fractalMode(i32(world.rec0.z), i32(world.rec0.w), p * f * world.rec1.x, oct1);
    h = mix(h, h2, weight2);
  }
  h = terraceFn(h, world.rec1.w, world.rec2.x);
  let ridged = pow(h, world.terrain.z); // extra crest-sharpening slider
  return ridged * world.terrain.x + world.terrain.w;
}

fn terrainNormal(xz: vec2f) -> vec3f {
  let e = 0.75;
  let hL = terrainHeight(xz - vec2f(e, 0.0));
  let hR = terrainHeight(xz + vec2f(e, 0.0));
  let hD = terrainHeight(xz - vec2f(0.0, e));
  let hU = terrainHeight(xz + vec2f(0.0, e));
  return normalize(vec3f(hL - hR, 2.0 * e, hD - hU));
}

// The terrain height field is normalized to [0,1] before scaling, so its surface
// lives entirely within this world-Y slab. Rays are clipped to it so the march
// skips the empty sky above and stops once it drops below — a large saving for
// primary rays (start at the slab top) and especially shadow rays toward a high
// sun (which exit the slab top after a few steps instead of marching to tMax).
fn terrainYRange() -> vec2f {
  let a = world.terrain.w;
  let b = world.terrain.w + world.terrain.x;
  return vec2f(min(a, b), max(a, b));
}

fn marchTerrain(ro: vec3f, rd: vec3f, tMax: f32) -> f32 {
  let yr = terrainYRange();
  var t0 = 0.1;
  var t1 = tMax;
  if (abs(rd.y) > 1e-5) {
    let ta = (yr.x - ro.y) / rd.y;
    let tb = (yr.y - ro.y) / rd.y;
    t0 = max(t0, min(ta, tb));
    t1 = min(t1, max(ta, tb));
  } else if (ro.y < yr.x || ro.y > yr.y) {
    return -1.0; // parallel to the slab and outside it → never crosses terrain
  }
  if (t1 <= t0) { return -1.0; }

  var t = t0;
  var lastT = t0;
  var lastH = (ro.y + rd.y * t0) - terrainHeight((ro + rd * t0).xz);
  if (lastH < 0.0) { return t0; } // entered already at/under the surface
  for (var i = 0; i < 220; i = i + 1) {
    t += max(0.08, 0.35 * lastH) + t * 0.0045;
    if (t > t1) { break; }
    let p = ro + rd * t;
    let h = p.y - terrainHeight(p.xz);
    if (h < 0.0) {
      let denom = max(lastH - h, 1e-4);
      return lastT + (t - lastT) * lastH / denom;
    }
    lastH = h;
    lastT = t;
  }
  return -1.0;
}

// ---------------------------------------------------------------- SDF primitives
fn sdSphere(p: vec3f, r: f32) -> f32 { return length(p) - r; }

fn sdBox(p: vec3f, b: vec3f) -> f32 {
  let q = abs(p) - b;
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdTorus(p: vec3f, R: f32, r: f32) -> f32 {
  let q = vec2f(length(p.xz) - R, p.y);
  return length(q) - r;
}

fn sdCylinder(p: vec3f, r: f32, h: f32) -> f32 {
  let d = vec2f(length(p.xz) - r, abs(p.y) - h);
  return min(max(d.x, d.y), 0.0) + length(max(d, vec2f(0.0)));
}

fn sdCappedCone(p: vec3f, h: f32, r1: f32, r2: f32) -> f32 {
  let q = vec2f(length(p.xz), p.y);
  let k1 = vec2f(r2, h);
  let k2 = vec2f(r2 - r1, 2.0 * h);
  let ca = vec2f(q.x - min(q.x, select(r2, r1, q.y < 0.0)), abs(q.y) - h);
  let cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
  let s = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
  return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

fn sdOctahedron(p: vec3f, s: f32) -> f32 {
  let q = abs(p);
  return (q.x + q.y + q.z - s) * 0.57735027;
}

// Vertical capsule: radius r, shaft half-height h (round caps beyond ±h).
fn sdCapsule(p: vec3f, r: f32, h: f32) -> f32 {
  var q = p;
  q.y -= clamp(q.y, -h, h);
  return length(q) - r;
}

fn sdRoundBox(p: vec3f, b: vec3f, r: f32) -> f32 {
  let q = abs(p) - (b - vec3f(r));
  return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

// Square pyramid: base half-width a (at y=-h), apex at y=+h (h = halfHeight).
fn sdPyramid(p0: vec3f, a: f32, hh: f32) -> f32 {
  let s = 2.0 * max(a, 1e-3);
  var p = vec3f(p0.x, p0.y + hh, p0.z) / s; // base → y=0, normalize to unit base
  let h = (2.0 * hh) / s;
  let m2 = h * h + 0.25;
  p = vec3f(abs(p.x), p.y, abs(p.z));
  if (p.z > p.x) { p = vec3f(p.z, p.y, p.x); }
  let pc = vec2f(p.x, p.z) - vec2f(0.5);
  let q = vec3f(pc.y, h * p.y - 0.5 * pc.x, h * pc.x + 0.5 * p.y);
  let sm = max(-q.x, 0.0);
  let t = clamp((q.y - 0.5 * pc.y) / (m2 + 0.25), 0.0, 1.0);
  let aa = m2 * (q.x + sm) * (q.x + sm) + q.y * q.y;
  let bb = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
  var d2 = min(aa, bb);
  if (min(q.y, -q.x * m2 - q.y * 0.5) > 0.0) { d2 = 0.0; }
  return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.y)) * s;
}

fn primSDF(pl: vec3f, shape: vec4f, corner: f32) -> f32 {
  let t = i32(shape.x);
  if (t == 0) { return sdSphere(pl, shape.y); }
  if (t == 1) { return sdBox(pl, shape.yzw); }
  if (t == 2) { return sdTorus(pl, shape.y, shape.z); }
  if (t == 3) { return sdCylinder(pl, shape.y, shape.z); }
  if (t == 4) { return sdCappedCone(pl, shape.z, shape.y, 0.05); }
  if (t == 5) { return sdOctahedron(pl, shape.y); }
  if (t == 6) { return sdCapsule(pl, shape.y, shape.z); }
  if (t == 7) { return sdPyramid(pl, shape.y, shape.z); }
  return sdRoundBox(pl, shape.yzw, corner);
}

// Per-group boolean CSG. flags.z is the boolean group (0 = ungrouped). Within a
// group the result is union(solids) carved by union(negatives); ungrouped solids
// just union in, ungrouped negatives do nothing. Returns (combined signed
// distance, nearest solid prim index — so cut surfaces inherit the solid's
// material and an inward-pointing gradient normal). Keep MAX in sync with the
// CPU's MAX_BOOL_GROUPS.
fn mapPrims(p: vec3f) -> vec2f {
  var dU = 1e9;    // ungrouped union
  var idxU = -1.0;
  var gPos : array<f32, 9>; // nearest solid distance per group (1..8)
  var gNeg : array<f32, 9>; // max(-d) over negatives per group
  var gIdx : array<f32, 9>; // nearest solid prim index per group
  for (var k = 1; k <= 8; k = k + 1) { gPos[k] = 1e9; gNeg[k] = -1e9; gIdx[k] = -1.0; }

  let n = i32(u.primCount);
  for (var i = 0; i < n; i = i + 1) {
    let pr = prims[i];
    let pl = (pr.invModel * vec4f(p, 1.0)).xyz;
    let d = primSDF(pl, pr.shape, pr.flags.y);
    let g = clamp(i32(pr.flags.z), 0, 8);
    let isNeg = pr.flags.x >= 0.5;
    if (g == 0) {
      if (!isNeg && d < dU) { dU = d; idxU = f32(i); }
    } else if (isNeg) {
      gNeg[g] = max(gNeg[g], -d);
    } else if (d < gPos[g]) {
      gPos[g] = d; gIdx[g] = f32(i);
    }
  }

  var best = dU;
  var bestIdx = idxU;
  for (var k = 1; k <= 8; k = k + 1) {
    if (gPos[k] < 1e8) {
      let gd = max(gPos[k], gNeg[k]);
      if (gd < best) { best = gd; bestIdx = gIdx[k]; }
    }
  }
  return vec2f(best, bestIdx);
}

fn marchPrims(ro: vec3f, rd: vec3f, tMax: f32) -> vec2f {
  if (u.primCount < 0.5) { return vec2f(-1.0, -1.0); }
  var t = 0.02;
  for (var i = 0; i < 160; i = i + 1) {
    let m = mapPrims(ro + rd * t);
    if (m.x < 0.001 * t + 0.0005) { return vec2f(t, m.y); }
    t += m.x;
    if (t > tMax) { break; }
  }
  return vec2f(-1.0, -1.0);
}

fn primNormal(p: vec3f) -> vec3f {
  let e = vec2f(0.0015, 0.0);
  return normalize(vec3f(
    mapPrims(p + e.xyy).x - mapPrims(p - e.xyy).x,
    mapPrims(p + e.yxy).x - mapPrims(p - e.yxy).x,
    mapPrims(p + e.yyx).x - mapPrims(p - e.yyx).x,
  ));
}

// ---------------------------------------------------------------- sky
fn coneSample(dir: vec3f, halfAngle: f32) -> vec3f {
  let r1 = rand();
  let r2 = rand();
  let a = r1 * 2.0 * PI;
  let z = 1.0 - r2 * (1.0 - cos(halfAngle));
  let s = sqrt(max(0.0, 1.0 - z * z));
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(dir.y) > 0.95);
  let tx = normalize(cross(up, dir));
  let ty = cross(dir, tx);
  return normalize(tx * (cos(a) * s) + ty * (sin(a) * s) + dir * z);
}

// Cloud height band (world Y) for a layer's altitude/thickness.
fn cloudBand(L: CloudLayer) -> vec2f {  // (centerY, halfHeight)
  return vec2f(60.0 + L.b.x * 240.0, mix(14.0, 130.0, L.b.z));
}

// Volumetric cloud field: density (w) + scattering color (xyz) at world point p.
fn cloudVolume(p: vec3f) -> vec4f {
  var dens = 0.0;
  var col = vec3f(0.0);
  let count = i32(world.flags.w);
  for (var i = 0; i < count; i = i + 1) {
    let L = world.clouds[i];
    let band = cloudBand(L);
    let dy = (p.y - band.x) / band.y;
    if (abs(dy) > 1.0) { continue; }
    let ctype = i32(L.a.x);
    let coverage = L.a.y;
    let density = L.a.z;
    let scale = max(L.a.w, 0.05);
    let seed = L.b.y;
    // Soft vertical falloff; cumulus/cumulonimbus billow up from a flat base.
    var vert = smoothstep(1.0, 0.55, abs(dy));
    if (ctype == 2 || ctype == 3) { vert *= smoothstep(-1.05, -0.4, dy); }

    let f = 0.0016 * scale;
    var q = vec3f(p.x * f + seed * 31.0, p.y * f, p.z * f + seed * 17.0)
          + vec3f(u.time * 0.01, 0.0, 0.0);
    var n = fbm3(q, 4);
    if (ctype == 0) { n = pow(n, 1.4); q.y *= 0.4; n = fbm3(q, 4); } // cirrus: stretched, wispy
    else if (ctype == 1) { n = smoothstep(0.4, 0.75, fbm3(q * 2.2, 4)); } // cirrocumulus: small dapples
    else if (ctype == 3) { n = clamp((fbm3(q * 0.8, 5) - 0.1) * 1.5, 0.0, 1.0); } // cumulonimbus: towering
    else if (ctype == 5) { q.y *= 0.5; n = mix(fbm3(q, 4), 0.62, 0.55); } // cirrostratus: thin uniform veil
    else if (ctype == 6) { n = smoothstep(0.42, 0.7, fbm3(q * 1.4, 4)); } // altocumulus: mid rolls
    else if (ctype == 7) { q.y *= 0.6; n = mix(fbm3(q, 4), 0.6, 0.5); } // altostratus: featureless mid sheet
    else if (ctype == 8) { n = clamp(fbm3(q * 0.7, 5) * 1.2 + 0.12, 0.0, 1.0); } // nimbostratus: thick dark layer
    else if (ctype == 9) { n = smoothstep(0.34, 0.64, fbm3(q * 1.1, 4)); } // stratocumulus: low lumpy rolls

    let thr = mix(0.66, 0.16, coverage);
    let d = smoothstep(thr, thr + 0.16, n) * vert * density;
    dens += d;
    col += L.c.rgb * d;
  }
  if (dens > 1e-4) { col /= dens; }
  return vec4f(col, dens);
}

// Henyey–Greenstein phase (forward scattering → silver lining toward the sun).
fn hg(c: f32, g: f32) -> f32 {
  let g2 = g * g;
  return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-3), 1.5));
}

// March the cloud slab from `ro` along `rd`, compositing over the background.
fn marchClouds(ro: vec3f, rd: vec3f, bg: vec3f) -> vec3f {
  if (world.flags.z < 0.5 || i32(world.flags.w) == 0) { return bg; }
  let yLo = 40.0;
  let yHi = 400.0;
  if (abs(rd.y) < 1e-3 && (ro.y < yLo || ro.y > yHi)) { return bg; }
  var t0 = 0.0;
  var t1 = 1e9;
  if (abs(rd.y) >= 1e-3) {
    let ta = (yLo - ro.y) / rd.y;
    let tb = (yHi - ro.y) / rd.y;
    t0 = max(0.0, min(ta, tb));
    t1 = max(ta, tb);
  }
  if (t1 <= t0 || t0 > 5000.0) { return bg; }
  t1 = min(t1, t0 + 2200.0);

  let steps = 44;
  let dt = (t1 - t0) / f32(steps);
  var t = t0 + dt * rand();
  var trans = 1.0;
  var scattered = vec3f(0.0);
  let sunDir = normalize(u.sunDir);
  let phase = max(hg(dot(rd, sunDir), 0.65), 0.35 * hg(dot(rd, sunDir), -0.15));
  let ambient = mix(world.horizon.rgb, world.zenith.rgb, 0.6) * skyLightAmt();

  for (var i = 0; i < steps; i = i + 1) {
    if (trans < 0.02) { break; }
    let p = ro + rd * t;
    let s = cloudVolume(p);
    let dens = s.w;
    if (dens > 0.01) {
      // Shadow: accumulate density toward the sun.
      var lacc = 0.0;
      let lstep = 24.0;
      var lp = p;
      for (var j = 0; j < 4; j = j + 1) {
        lp += sunDir * lstep;
        lacc += cloudVolume(lp).w;
      }
      let lightTrans = exp(-lacc * lstep * 0.05);
      let powder = 1.0 - exp(-dens * 2.0);
      let sunLit = u.sunColor * (u.sunFlag * lightTrans * phase * 4.0);
      let lit = (sunLit + ambient * 0.55) * s.rgb * mix(0.5, 1.0, powder);
      let stepTrans = exp(-dens * dt * 0.06);
      scattered += trans * (1.0 - stepTrans) * lit;
      trans *= stepTrans;
    }
    t += dt;
  }
  return bg * trans + scattered;
}

fn skyGradient(dir: vec3f) -> vec3f {
  let up = clamp(dir.y, 0.0, 1.0);
  return mix(world.horizon.rgb, world.zenith.rgb, pow(up, 0.55));
}

// How much daylight the sky scatters — tracks the sun's intensity. With no sun
// (all lights removed, or sun at ~0) the sky goes dark; objects then receive
// light only from explicit lights (points / planets).
fn skyLightAmt() -> f32 {
  return u.sunFlag * clamp(u.sunBright / 3.0, 0.0, 1.5);
}

// Background sky: sun-lit gradient + stars + sun disk. No clouds.
fn sky(dir: vec3f) -> vec3f {
  var col = skyGradient(dir) * skyLightAmt();

  if (u.starsFlag > 0.5) { col += stars(dir); }

  // Sun disk + glow — brightness tracks the sun light's intensity.
  if (u.sunFlag > 0.5) {
    let sd = max(dot(dir, u.sunDir), 0.0);
    let disk = pow(sd, world.zenith.w) * (2.0 + u.sunBright * 6.0);
    let glow = pow(sd, 6.0) * (0.12 * u.sunBright + 0.08);
    col += u.sunColor * (disk + glow);

    // Golden-hour halo: a broad, warm scattering glow around the sun that swells
    // as the sun nears the horizon and pools along the horizon band — the aerial
    // glow that makes low-sun scenes read as golden hour. Free (sky-only) and it
    // also warms distant geometry via the haze, which mixes in sky().
    let lowSun = 1.0 - smoothstep(0.02, 0.32, u.sunDir.y);
    let horizonBand = 1.0 - smoothstep(0.0, 0.40, max(dir.y, 0.0));
    let halo = pow(sd, 2.5) * (0.35 + 0.65 * horizonBand);
    col += u.sunColor * (halo * lowSun * 0.7 * skyLightAmt());
  }
  return col;
}

// Sky + volumetric clouds (call at a ray miss with the ray's origin).
fn skyClouds(ro: vec3f, dir: vec3f) -> vec3f {
  return marchClouds(ro, dir, sky(dir));
}

fn terrainColor(p: vec3f, n: vec3f) -> vec3f {
  let slope = 1.0 - n.y;
  let altitude = clamp((p.y - world.terrain.w) / world.terrain.x, 0.0, 1.0);
  let low = world.tlow.rgb;
  let rock = world.trock.rgb;
  let high = world.thigh.rgb;
  let snowLine = world.rec2.z;
  let slopeRock = world.rec2.w;
  var c = mix(low, rock, smoothstep(slopeRock - 0.15, slopeRock + 0.15, slope));
  c = mix(c, high, smoothstep(snowLine, snowLine + 0.2, altitude) * (1.0 - smoothstep(slopeRock, slopeRock + 0.3, slope)));
  // Lighten the very-low shoreline toward the ground color (beach/flats).
  c = mix(low * 1.15, c, smoothstep(0.0, 0.06, p.y));
  c *= 0.85 + 0.3 * fbm(p.xz * 0.4, 3);
  return c;
}

// ---------------------------------------------------------------- prim materials
fn triplanar(p: vec3f, nl: vec3f, layer: i32) -> vec3f {
  let w = abs(nl);
  let sum = w.x + w.y + w.z + 1e-4;
  let cx = textureSampleLevel(primTex, meshSamp, p.zy, layer, 0.0).rgb;
  let cy = textureSampleLevel(primTex, meshSamp, p.xz, layer, 0.0).rgb;
  let cz = textureSampleLevel(primTex, meshSamp, p.xy, layer, 0.0).rgb;
  return pow((cx * w.x + cy * w.y + cz * w.z) / sum, vec3f(2.2)); // sRGB → linear
}

// Evaluate a primitive's pattern at its local hit point.
fn patternColor(pr: Prim, pl: vec3f, nl: vec3f) -> vec3f {
  let pat = i32(pr.mat1.w);
  let a = pr.mat0.rgb;
  let b = pr.mat1.rgb;
  let s = pr.mat2.x;
  if (pat == 1) {                 // checker
    let q = floor(pl * (s * 0.25));
    return select(a, b, fract((q.x + q.y + q.z) * 0.5) > 0.25);
  } else if (pat == 2) {          // noise
    let n = fbm(pl.xz * (s * 0.1) + vec2f(pl.y * 0.4, 0.0), 5);
    return mix(a, b, clamp(n, 0.0, 1.0));
  } else if (pat == 3) {          // wood (rings)
    let r = length(pl.xz) * (s * 0.12) + fbm(pl.xz * (s * 0.05), 4) * 1.5;
    return mix(a, b, smoothstep(0.4, 0.5, abs(fract(r) - 0.5)));
  } else if (pat == 4) {          // marble (veins)
    let t = pl.x * (s * 0.1) + fbm(pl.xz * (s * 0.08), 5) * 3.0;
    return mix(a, b, 0.5 + 0.5 * sin(t * PI));
  } else if (pat == 5) {          // gradient (local Y)
    return mix(a, b, clamp(pl.y * (s * 0.05) + 0.5, 0.0, 1.0));
  } else if (pat == 6) {          // image (triplanar)
    let layer = i32(pr.mat2.w);
    if (layer >= 0) { return triplanar(pl * (s * 0.05), nl, layer); }
    return a;
  }
  return a;                       // solid
}

// Procedural surface relief: perturb the normal using tangent-plane fbm.
fn bumpNormal(p: vec3f, n: vec3f, amt: f32, scale: f32) -> vec3f {
  if (amt <= 0.001) { return n; }
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.95);
  let t = normalize(cross(up, n));
  let bt = cross(n, t);
  let uv = vec2f(dot(p, t), dot(p, bt)) * (scale * 0.15);
  let e = 0.5;
  let hC = fbm(uv, 4);
  let hU = fbm(uv + vec2f(e, 0.0), 4);
  let hV = fbm(uv + vec2f(0.0, e), 4);
  let grad = (t * (hU - hC) + bt * (hV - hC)) * (amt * 4.0);
  return normalize(n - grad);
}

// ---------------------------------------------------------------- intersection
struct Hit {
  t       : f32,
  kind    : i32,   // 0 miss, 1 terrain, 2 water, 4 primitive, 5 mesh
  primIdx : i32,   // primitive index, or instance index for meshes
  matId   : i32,   // global material index (mesh hits)
  pos     : vec3f,
  nrm     : vec3f,
  uv      : vec2f,
  color   : vec3f, // interpolated vertex color (mesh hits; white if none)
};

// Decode a per-vertex color packed as r + g·256 + b·65536 (see gltf.ts). The
// stored float is an exact integer (≤ 2^24−1), read verbatim — no rounding bias.
fn decodeVCol(f: f32) -> vec3f {
  let p = u32(f);
  return vec3f(f32(p & 0xffu), f32((p >> 8u) & 0xffu), f32((p >> 16u) & 0xffu)) * (1.0 / 255.0);
}

// Möller–Trumbore. Returns (hit?, t, u, v) packed as vec4 (x>0 means hit).
fn rayTri(ro: vec3f, rd: vec3f, p0: vec3f, p1: vec3f, p2: vec3f) -> vec4f {
  let e1 = p1 - p0;
  let e2 = p2 - p0;
  let pv = cross(rd, e2);
  let det = dot(e1, pv);
  if (abs(det) < 1e-8) { return vec4f(-1.0, 0.0, 0.0, 0.0); }
  let inv = 1.0 / det;
  let tv = ro - p0;
  let u = dot(tv, pv) * inv;
  if (u < 0.0 || u > 1.0) { return vec4f(-1.0, 0.0, 0.0, 0.0); }
  let qv = cross(tv, e1);
  let v = dot(rd, qv) * inv;
  if (v < 0.0 || u + v > 1.0) { return vec4f(-1.0, 0.0, 0.0, 0.0); }
  let t = dot(e2, qv) * inv;
  if (t <= 1e-4) { return vec4f(-1.0, 0.0, 0.0, 0.0); }
  return vec4f(1.0, t, u, v);
}

fn slabHit(ro: vec3f, invD: vec3f, lo: vec3f, hi: vec3f, tMax: f32) -> bool {
  let t0 = (lo - ro) * invD;
  let t1 = (hi - ro) * invD;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let near = max(max(tmin.x, tmin.y), tmin.z);
  let far = min(min(tmax.x, tmax.y), tmax.z);
  return far >= max(near, 0.0) && near < tMax;
}

// Like slabHit, but returns the ray's entry distance into the box (clamped to 0
// if the origin is inside), or NO_HIT if it misses / lies beyond tMax. Used to
// order child visitation front-to-back so the nearer subtree tightens bestT
// before the farther one is tested.
const NO_HIT = 1e30;
fn slabEnter(ro: vec3f, invD: vec3f, lo: vec3f, hi: vec3f, tMax: f32) -> f32 {
  let t0 = (lo - ro) * invD;
  let t1 = (hi - ro) * invD;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let near = max(max(tmin.x, tmin.y), tmin.z);
  let far = min(min(tmax.x, tmax.y), tmax.z);
  if (far >= max(near, 0.0) && near < tMax) { return max(near, 0.0); }
  return NO_HIT;
}

// Traverse one BLAS in its local space. `ro`/`rd` are already in local space
// (rd is NOT normalized, so t stays comparable to the world ray). Node child
// links and leaf triangle indices are BLAS-relative; nodeBase/triBase rebase
// them into the global pools. Returns vec4(t, globalTriIndex, u, v); t < 0 miss.
fn traceBLAS(ro: vec3f, rd: vec3f, tMax: f32, nodeBase: u32, triBase: u32) -> vec4f {
  let invD = 1.0 / rd;
  var stack: array<u32, 64>;
  var sp = 0;
  stack[0] = nodeBase; sp = 1;
  var bestT = tMax;
  var bestTri = -1.0;
  var bu = 0.0;
  var bv = 0.0;

  while (sp > 0) {
    sp = sp - 1;
    let node = nodes[stack[sp]];
    if (!slabHit(ro, invD, node.lo.xyz, node.hi.xyz, bestT)) { continue; }
    let count = i32(node.hi.w);
    if (count > 0) {
      let first = triBase + u32(node.lo.w);
      for (var k = 0; k < count; k = k + 1) {
        let tri = triangles[first + u32(k)];
        let r = rayTri(ro, rd, tri.p0.xyz, tri.p1.xyz, tri.p2.xyz);
        if (r.x > 0.0 && r.y < bestT) {
          bestT = r.y; bestTri = f32(first + u32(k)); bu = r.z; bv = r.w;
        }
      }
    } else {
      // Internal node: count < 0 encodes -(rightChild + 1). Visit the nearer
      // child first (pushed last → popped first) so bestT tightens early and the
      // farther subtree is more likely to be culled at its next slabHit.
      let li = nodeBase + u32(node.lo.w);
      let ri = nodeBase + u32(-count - 1);
      let lEnter = slabEnter(ro, invD, nodes[li].lo.xyz, nodes[li].hi.xyz, bestT);
      let rEnter = slabEnter(ro, invD, nodes[ri].lo.xyz, nodes[ri].hi.xyz, bestT);
      // Push far first, near second; skip children that already miss.
      var nearI = li; var nearE = lEnter; var farI = ri; var farE = rEnter;
      if (rEnter < lEnter) { nearI = ri; nearE = rEnter; farI = li; farE = lEnter; }
      if (farE < NO_HIT && sp < 63) { stack[sp] = farI; sp = sp + 1; }
      if (nearE < NO_HIT && sp < 63) { stack[sp] = nearI; sp = sp + 1; }
    }
  }
  return vec4f(select(-1.0, bestT, bestTri >= 0.0), bestTri, bu, bv);
}

fn blasOccluded(ro: vec3f, rd: vec3f, tMax: f32, nodeBase: u32, triBase: u32) -> bool {
  let invD = 1.0 / rd;
  var stack: array<u32, 64>;
  var sp = 0;
  stack[0] = nodeBase; sp = 1;
  while (sp > 0) {
    sp = sp - 1;
    let node = nodes[stack[sp]];
    if (!slabHit(ro, invD, node.lo.xyz, node.hi.xyz, tMax)) { continue; }
    let count = i32(node.hi.w);
    if (count > 0) {
      let first = triBase + u32(node.lo.w);
      for (var k = 0; k < count; k = k + 1) {
        let tri = triangles[first + u32(k)];
        let r = rayTri(ro, rd, tri.p0.xyz, tri.p1.xyz, tri.p2.xyz);
        if (r.x > 0.0 && r.y < tMax) { return true; }
      }
    } else {
      // Internal node: count < 0 encodes -(rightChild + 1).
      let l = nodeBase + u32(node.lo.w);
      let r = nodeBase + u32(-count - 1);
      if (sp < 63) { stack[sp] = l; sp = sp + 1; }
      if (sp < 63) { stack[sp] = r; sp = sp + 1; }
    }
  }
  return false;
}

struct MeshHit {
  hit  : bool,
  t    : f32,
  inst : i32,
  tri  : i32,  // global triangle index
  u    : f32,
  v    : f32,
};

// Top level: traverse the TLAS, descending into each candidate instance's BLAS
// in local space. Turns the old O(instances) linear scan into O(log instances).
fn traceInstances(ro: vec3f, rd: vec3f, tMax: f32) -> MeshHit {
  var best: MeshHit;
  best.hit = false;
  best.t = tMax;
  let invD = 1.0 / rd;
  var stack: array<u32, 32>;
  var sp = 0;
  stack[0] = 0u; sp = 1;

  while (sp > 0) {
    sp = sp - 1;
    let node = tlasNodes[stack[sp]];
    if (!slabHit(ro, invD, node.lo.xyz, node.hi.xyz, best.t)) { continue; }
    let count = i32(node.hi.w);
    if (count > 0) {
      let first = u32(node.lo.w);
      for (var k = 0u; k < u32(count); k = k + 1u) {
        let ii = tlasOrder[first + k];
        let inst = instances[ii];
        let rol = (inst.invModel * vec4f(ro, 1.0)).xyz;
        let rdl = (inst.invModel * vec4f(rd, 0.0)).xyz; // unnormalized → t preserved
        let r = traceBLAS(rol, rdl, best.t, u32(inst.info.x), u32(inst.info.y));
        if (r.x > 0.0 && r.x < best.t) {
          best.hit = true; best.t = r.x; best.inst = i32(ii); best.tri = i32(r.y); best.u = r.z; best.v = r.w;
        }
      }
    } else {
      // Internal node: count < 0 encodes -(rightChild + 1). Visit the nearer
      // child first so best.t tightens before the farther subtree is tested.
      let li = u32(node.lo.w);
      let ri = u32(-count - 1);
      let lEnter = slabEnter(ro, invD, tlasNodes[li].lo.xyz, tlasNodes[li].hi.xyz, best.t);
      let rEnter = slabEnter(ro, invD, tlasNodes[ri].lo.xyz, tlasNodes[ri].hi.xyz, best.t);
      var nearI = li; var nearE = lEnter; var farI = ri; var farE = rEnter;
      if (rEnter < lEnter) { nearI = ri; nearE = rEnter; farI = li; farE = lEnter; }
      if (farE < NO_HIT && sp < 31) { stack[sp] = farI; sp = sp + 1; }
      if (nearE < NO_HIT && sp < 31) { stack[sp] = nearI; sp = sp + 1; }
    }
  }
  return best;
}

fn intersect(ro: vec3f, rd: vec3f) -> Hit {
  var h: Hit;
  h.kind = 0;
  h.primIdx = -1;
  h.t = 1e9;

  if (world.flags.y > 0.5 && abs(rd.y) > 1e-4) {
    let tw = (WATER_Y - ro.y) / rd.y;
    if (tw > 1e-3 && tw < h.t) { h.t = tw; h.kind = 2; }
  }
  let mp = marchPrims(ro, rd, min(h.t, 1000.0));
  if (mp.x > 0.0 && mp.x < h.t) { h.t = mp.x; h.kind = 4; h.primIdx = i32(mp.y); }

  if (world.flags.x > 0.5) {
    let tt = marchTerrain(ro, rd, min(h.t, 1200.0));
    if (tt > 0.0 && tt < h.t) { h.t = tt; h.kind = 1; h.primIdx = -1; }
  }

  // Planet bodies + rings — point lights with a visible glowing sphere (color.w
  // = radius) and an optional ring disc (ring0/ring1).
  let nLights = i32(u.lightCount);
  for (var i = 0; i < nLights; i = i + 1) {
    let lt = lights[i];
    if (lt.data.x < 0.5 || lt.color.w <= 0.0) { continue; }
    let center = lightCenter(lt);
    let radius = lightRadius(lt);
    let oc = ro - center;
    let b = dot(oc, rd);
    let cc = dot(oc, oc) - radius * radius;
    let disc = b * b - cc;
    if (disc > 0.0) {
      let tt = -b - sqrt(disc);
      if (tt > 1e-3 && tt < h.t) { h.t = tt; h.kind = 6; h.primIdx = i; }
    }
    if (lt.ring0.z > 0.0) {
      let tilt = lt.ring0.w;
      let rn = normalize(vec3f(sin(tilt), cos(tilt), 0.0));
      let denom = dot(rd, rn);
      if (abs(denom) > 1e-4) {
        let tr = dot(center - ro, rn) / denom;
        if (tr > 1e-3 && tr < h.t) {
          let rad = length(ro + rd * tr - center);
          if (rad >= lt.ring0.x * radius && rad <= lt.ring0.y * radius) {
            h.t = tr; h.kind = 7; h.primIdx = i;
          }
        }
      }
    }
  }

  if (u.instCount > 0.5) {
    let m = traceInstances(ro, rd, h.t);
    if (m.hit) {
      let tri = triangles[u32(m.tri)];
      let w = 1.0 - m.u - m.v;
      let ln = tri.n0.xyz * w + tri.n1.xyz * m.u + tri.n2.xyz * m.v;
      // World normal = inverse-transpose(model) · localN = transpose(invModel) · localN.
      let inv3 = mat3x3f(
        instances[m.inst].invModel[0].xyz,
        instances[m.inst].invModel[1].xyz,
        instances[m.inst].invModel[2].xyz,
      );
      h.t = m.t; h.kind = 5; h.primIdx = m.inst;
      h.matId = i32(tri.uv2m.z);
      h.uv = tri.uv01.xy * w + tri.uv01.zw * m.u + tri.uv2m.xy * m.v;
      h.color = decodeVCol(tri.n1.w) * w + decodeVCol(tri.n2.w) * m.u + decodeVCol(tri.uv2m.w) * m.v;
      h.nrm = normalize(transpose(inv3) * ln);

      // Normal mapping: perturb the geometric normal in tangent space.
      let nLayer = i32(materials[h.matId].params.w);
      if (nLayer >= 0) {
        let tl = vec3f(tri.p0.w, tri.p1.w, tri.p2.w); // face tangent (local)
        var T = normalize(transpose(inv3) * tl);
        let N = h.nrm;
        T = normalize(T - N * dot(N, T)); // Gram–Schmidt
        let B = cross(N, T) * tri.n0.w;   // handedness
        let nm = textureSampleLevel(meshNormalTex, meshSamp, h.uv, nLayer, 0.0).xyz * 2.0 - 1.0;
        h.nrm = normalize(T * nm.x + B * nm.y + N * nm.z);
      }
      h.pos = ro + rd * h.t;
      return h;
    }
  }

  if (h.kind == 0) { return h; }
  h.pos = ro + rd * h.t;
  if (h.kind == 1) {
    h.nrm = terrainNormal(h.pos.xz);
  } else if (h.kind == 2) {
    let e = 0.6;
    let s = 0.25;
    let nx = fbm((h.pos.xz + vec2f(e, 0.0)) * s + u.time * 0.05, 4)
           - fbm((h.pos.xz - vec2f(e, 0.0)) * s + u.time * 0.05, 4);
    let nz = fbm((h.pos.xz + vec2f(0.0, e)) * s + u.time * 0.05, 4)
           - fbm((h.pos.xz - vec2f(0.0, e)) * s + u.time * 0.05, 4);
    h.nrm = normalize(vec3f(-nx * 1.5, 1.0, -nz * 1.5));
  } else if (h.kind == 6) {
    h.nrm = normalize(h.pos - lightCenter(lights[h.primIdx]));
  } else if (h.kind == 7) {
    h.nrm = vec3f(0.0, 1.0, 0.0); // rings shaded radially in trace, no surface normal
  } else {
    h.nrm = primNormal(h.pos);
  }
  return h;
}

// Any-hit TLAS traversal for shadow rays: stops at the first blocking triangle.
fn tlasOccluded(o: vec3f, ldir: vec3f, maxDist: f32) -> bool {
  let invD = 1.0 / ldir;
  var stack: array<u32, 32>;
  var sp = 0;
  stack[0] = 0u; sp = 1;

  while (sp > 0) {
    sp = sp - 1;
    let node = tlasNodes[stack[sp]];
    if (!slabHit(o, invD, node.lo.xyz, node.hi.xyz, maxDist)) { continue; }
    let count = i32(node.hi.w);
    if (count > 0) {
      let first = u32(node.lo.w);
      for (var k = 0u; k < u32(count); k = k + 1u) {
        let ii = tlasOrder[first + k];
        let inst = instances[ii];
        let rol = (inst.invModel * vec4f(o, 1.0)).xyz;
        let rdl = (inst.invModel * vec4f(ldir, 0.0)).xyz;
        if (blasOccluded(rol, rdl, maxDist, u32(inst.info.x), u32(inst.info.y))) { return true; }
      }
    } else {
      let l = u32(node.lo.w);
      let r = u32(-count - 1);
      if (sp < 31) { stack[sp] = l; sp = sp + 1; }
      if (sp < 31) { stack[sp] = r; sp = sp + 1; }
    }
  }
  return false;
}

fn occluded(p: vec3f, ldir: vec3f, maxDist: f32) -> bool {
  let o = p + ldir * 0.06;
  let mp = marchPrims(o, ldir, min(maxDist, 1000.0));
  if (mp.x > 0.0 && mp.x < maxDist) { return true; }
  if (u.instCount > 0.5 && tlasOccluded(o, ldir, maxDist)) { return true; }
  if (world.flags.x > 0.5) {
    let tt = marchTerrain(o, ldir, min(maxDist, 800.0));
    if (tt > 0.0 && tt < maxDist) { return true; }
  }
  return false;
}

fn fresnelSchlick(cosTheta: f32, f0: f32) -> f32 {
  return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Cosine-weighted hemisphere direction about n (for ambient-occlusion sampling).
fn cosineHemisphere(n: vec3f) -> vec3f {
  let r1 = rand();
  let r2 = rand();
  let r = sqrt(r1);
  let phi = 2.0 * PI * r2;
  let x = r * cos(phi);
  let y = r * sin(phi);
  let z = sqrt(max(0.0, 1.0 - r1));
  let up = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(n.y) > 0.95);
  let tx = normalize(cross(up, n));
  let ty = cross(n, tx);
  return normalize(tx * x + ty * y + n * z);
}

// Ambient occlusion: distance within which nearby geometry darkens the ambient,
// and the ambient floor a fully-occluded point keeps. One sky-visibility ray per
// sample averages into soft AO as frames accumulate.
const AO_DIST  = 14.0;
const AO_FLOOR = 0.35;

// Skylight ambient (sun-driven), attenuated by local sky visibility so contact
// points and crevices darken — this grounds objects instead of leaving them
// looking pasted onto the scene. This is a *stand-in* for indirect light and is
// dropped once real GI bounces are enabled (they carry the sky in for real).
fn skyAmbient(pos: vec3f, n: vec3f, albedo: vec3f) -> vec3f {
  var ambient = skyGradient(n) * skyLightAmt() * 0.30 * albedo;
  if (occluded(pos, cosineHemisphere(n), AO_DIST)) { ambient *= AO_FLOOR; }
  return ambient;
}

// The sky seen by an indirect (GI) bounce ray: the lit gradient only — no sun
// disk, glow, halo, or clouds. The sun is sampled directly (NEE) at each diffuse
// vertex, so re-counting its disk here would double the sun and spray fireflies.
fn skyDome(dir: vec3f) -> vec3f {
  return skyGradient(dir) * skyLightAmt();
}

// Direct lighting (next-event estimation) for a diffuse point: emission-free,
// ambient-free — just the explicit lights, shadow-tested.
fn directLight(pos: vec3f, n: vec3f, albedo: vec3f) -> vec3f {
  var col = vec3f(0.0);
  let n_lights = i32(u.lightCount);
  for (var i = 0; i < n_lights; i = i + 1) {
    let lt = lights[i];
    var L: vec3f;
    var atten = 1.0;
    var maxD = 1e6;
    if (lt.data.x < 0.5) {
      L = normalize(lt.vec.xyz);
      maxD = 800.0;
    } else if (lightInSky(lt)) {
      // Distant sky planet: behaves like the sun — directional, no falloff with
      // scene position. Intensity is normalized against a reference distance so
      // the same intensity slider reads similarly to a near planet.
      L = normalize(lt.vec.xyz);
      atten = 1.0 / (60.0 * 60.0);
      maxD = SKY_DIST;
    } else {
      let toL = lt.vec.xyz - pos;
      let dist = max(length(toL), 1e-3);
      L = toL / dist;
      let rng = max(lt.data.w, 1e-3);
      atten = 1.0 / (dist * dist) * (1.0 / (1.0 + dist / rng));
      maxD = dist;
    }
    let ndl = max(dot(n, L), 0.0);
    if (ndl <= 0.0) { continue; }
    // Cull lights whose (pre-shadow) contribution is negligible BEFORE paying for
    // the shadow march. Critical for local point lights — e.g. an emitter fill-
    // light — which reach only a small area but would otherwise shadow-march the
    // whole scene toward themselves on every pixel. The sun/planets have atten≈1
    // so they never trip this.
    let bright = max(lt.color.x, max(lt.color.y, lt.color.z));
    if (lt.data.y * atten * ndl * bright < 0.0015) { continue; }
    let Lj = coneSample(L, max(lt.data.z, 0.001));
    let vis = select(1.0, 0.0, occluded(pos, Lj, maxD));
    col += albedo * lt.color.xyz * (lt.data.y * ndl * atten * vis);
  }
  return col;
}

// Direct + ambient lighting for a diffuse point (legacy Whitted look).
fn shade(pos: vec3f, n: vec3f, albedo: vec3f) -> vec3f {
  return skyAmbient(pos, n, albedo) + directLight(pos, n, albedo);
}

// Is indirect GI live for THIS sample? Interaction resets accumulation every
// frame, so sample 0 (frame 0) always renders the cheap direct-lit baseline —
// keeping orbit/edit fully interactive — while the refinement samples that build
// up when the camera is still add the (expensive) bounced light. The converged
// image is essentially unchanged: the single cheap sample is <1% of the total.
fn giActive() -> bool {
  return u.giBounces > 0.5 && u.frame >= 1.0;
}

// Color deposited at a diffuse vertex. With GI live, indirect light (sky + bounces)
// arrives through the continued ray, so we keep only direct light and drop the
// sky-ambient stand-in that would otherwise double the fill. Otherwise we keep the
// ambient term so the cheap baseline sample still looks grounded.
fn litDiffuse(pos: vec3f, n: vec3f, albedo: vec3f) -> vec3f {
  if (giActive()) { return directLight(pos, n, albedo); }
  return skyAmbient(pos, n, albedo) + directLight(pos, n, albedo);
}

// Volumetric single-scatter light shafts (crepuscular rays). Marches the primary
// ray through the hazy air and, at each step, tests sun visibility (terrain +
// geometry). Lit air in-scatters sunlight toward the eye; shadowed air stays
// dark — so beams rake out from behind ridgelines and objects. A jittered start
// offset trades banding for noise that progressive accumulation averages away.
// Sun-only and opt-in via the strength slider; costs nothing when off or at night.
fn godRays(ro: vec3f, rd: vec3f, hitDist: f32) -> vec3f {
  // Off, sunless, or the cheap interaction sample (frame 0) → skip entirely. The
  // 20-step shadow march is far too costly to run while orbiting; it only refines
  // in once the camera settles (see giActive()).
  if (u.shaftStr <= 0.0 || u.sunFlag < 0.5 || u.frame < 1.0) { return vec3f(0.0); }
  let sun = normalize(u.sunDir);
  // Integrate to the first surface, but never past SHAFT_MAX (sky pixels march
  // the full bounded span so shafts read against the sky too).
  let far = min(select(SHAFT_MAX, hitDist, hitDist < 1e8), SHAFT_MAX);
  let ds = far / f32(SHAFT_STEPS);
  let sigma = 0.014 * u.shaftStr;      // scattering coefficient (also self-attenuation)
  let phase = hg(dot(rd, sun), SHAFT_G);
  var t = ds * rand();                 // jitter the first step to break up banding
  var acc = 0.0;
  for (var i = 0; i < SHAFT_STEPS; i = i + 1) {
    let p = ro + rd * t;
    if (!occluded(p, sun, 800.0)) {
      acc += exp(-sigma * t) * ds;     // transmittance-weighted in-scatter of lit air
    }
    t += ds;
  }
  return acc * sigma * phase * u.sunColor * skyLightAmt();
}

// Additive emission from the particle field along a ray. Each particle is a soft
// glowing blob: we take the ray's closest approach to the particle centre and add
// its (opacity-weighted) colour with a smooth radial falloff — no occlusion, so
// blobs blend additively like real fire/sparks. Particles past `maxT` are hidden
// behind the scene surface and skipped. Sub-frame motion blur comes for free by
// advancing each particle along its velocity by a random slice of the shutter.
fn particleEmission(ro: vec3f, rd: vec3f, maxT: f32) -> vec3f {
  let n = i32(u.partCount);
  if (n <= 0) { return vec3f(0.0); }
  // Whole-cloud bounding-sphere reject: most rays don't look anywhere near the
  // particles, so one sphere test skips the entire per-particle loop for them.
  let ocb = u.partBound - ro;
  let tcab = dot(ocb, rd);
  let d2b = dot(ocb, ocb) - tcab * tcab;
  if (d2b > u.partRadius * u.partRadius) { return vec3f(0.0); }   // ray misses the cloud
  if (tcab + u.partRadius < 0.0) { return vec3f(0.0); }           // cloud entirely behind the eye
  if (tcab - u.partRadius > maxT) { return vec3f(0.0); }          // cloud entirely behind the surface
  let dt = u.shutter * rand();
  var acc = vec3f(0.0);
  for (var i = 0; i < n; i = i + 1) {
    let pt = particles[i];
    let op = pt.p1.w;
    if (op <= 0.003) { continue; }              // dead / recycled slot
    let c = pt.p0.xyz + pt.p2.xyz * dt;         // centre at the jittered instant
    let r = pt.p0.w;
    let oc = c - ro;
    let tca = dot(oc, rd);                       // rd is unit length
    if (tca <= 0.0 || tca > maxT) { continue; } // behind the eye or occluded
    let d2 = dot(oc, oc) - tca * tca;           // squared perpendicular distance
    let r2 = r * r;
    if (d2 >= r2) { continue; }
    let falloff = 1.0 - d2 / r2;                // 1 at centre → 0 at the rim
    acc += pt.p1.xyz * (op * falloff * falloff);
  }
  return acc;
}

fn trace(ro0: vec3f, rd0: vec3f) -> vec3f {
  var ro = ro0;
  var rd = rd0;
  var throughput = vec3f(1.0);
  var radiance = vec3f(0.0);
  var primaryDist = -1.0;
  var giUsed = 0;         // indirect diffuse bounces taken so far
  var diffuseGI = false;  // path has scattered diffusely → sky misses use the dome

  for (var bounce = 0; bounce < MAX_BOUNCE; bounce = bounce + 1) {
    // Early-out low-contribution paths: once the accumulated reflection/transmission
    // throughput is negligible, the remaining bounces add < 1% and we can skip
    // their (full-scene) intersect. Diffuse hits already break; this trims deep,
    // dim specular chains (e.g. grazing water → metal). throughput is 1 at bounce
    // 0, so the primary ray always runs and primaryDist is always captured.
    if (max(throughput.r, max(throughput.g, throughput.b)) < 0.01) { break; }

    let h = intersect(ro, rd);
    if (bounce == 0) { primaryDist = select(1e9, h.t, h.kind != 0); }

    if (h.kind == 0) {
      // A specular/primary miss shows the real sky (sun disk + clouds) so mirrors
      // and water reflect it; a diffuse-GI bounce that escapes takes only the
      // sunless dome to avoid double-counting the directly-sampled sun.
      if (diffuseGI) { radiance += throughput * skyDome(rd); }
      else { radiance += throughput * skyClouds(ro, rd); }
      break;
    }

    let n = h.nrm;
    if (h.kind == 6) {
      // Planet body — its color, banded/mottled like a gas giant, lit by the sun
      // with a gentle self-glow. Brightness is independent of the cast intensity.
      let lt = lights[h.primIdx];
      let lp = (h.pos - lightCenter(lt)) / max(lightRadius(lt), 0.1); // unit-ish local coords
      let bands = 0.82 + 0.18 * sin(lp.y * 9.0 + 2.0 * fbm3(lp * 3.0, 4));
      let mottle = 0.7 + 0.6 * fbm3(lp * 5.0, 4);
      let tex = clamp(bands * mottle, 0.2, 1.4);
      let sunLit = max(dot(n, normalize(u.sunDir)), 0.0) * u.sunFlag * clamp(u.sunBright / 3.0, 0.0, 1.5);
      radiance += throughput * lt.color.rgb * tex * (0.18 + 0.9 * sunLit + 0.25 * skyLightAmt());
      break;
    } else if (h.kind == 7) {
      // Planet ring — banded annulus; stochastic transparency accumulates to a
      // soft, semi-transparent disc over samples.
      let lt = lights[h.primIdx];
      let rad = length(h.pos - lightCenter(lt));
      let inner = lt.ring0.x * lightRadius(lt);
      let outer = lt.ring0.y * lightRadius(lt);
      let frac = clamp((rad - inner) / max(outer - inner, 0.01), 0.0, 1.0);
      let gaps = 0.5 + 0.5 * fbm(vec2f(frac * 34.0, 7.3), 4);
      let edge = smoothstep(0.0, 0.05, frac) * smoothstep(1.0, 0.9, frac);
      let dens = clamp(lt.ring0.z * gaps * edge, 0.0, 1.0);
      if (rand() > dens) { ro = h.pos + rd * 0.05; continue; } // pass through
      let lit = 0.4 + 0.6 * skyLightAmt();
      radiance += throughput * lt.ring1.rgb * lit;
      break;
    } else if (h.kind == 1) {
      let tcol = terrainColor(h.pos, n);
      radiance += throughput * litDiffuse(h.pos, n, tcol);
      if (giActive() && giUsed < i32(u.giBounces)) {
        giUsed = giUsed + 1;
        diffuseGI = true;
        throughput *= tcol;
        ro = h.pos + n * 0.02;
        rd = cosineHemisphere(n);
        continue;
      }
      break;
    } else if (h.kind == 4) {
      let pr = prims[h.primIdx];
      let pl = (pr.invModel * vec4f(h.pos, 1.0)).xyz;
      let inv3 = mat3x3f(pr.invModel[0].xyz, pr.invModel[1].xyz, pr.invModel[2].xyz);
      let nl = normalize(inv3 * n);
      let albedo = patternColor(pr, pl, nl);
      let nn = bumpNormal(h.pos, n, pr.mat2.y, pr.mat2.z);
      let refl = pr.mat0.w;
      if (refl > 0.002) {
        // Reflective: deterministic split — deposit the matte fraction now, then
        // continue the mirror ray. (GI is applied only to fully-matte surfaces.)
        radiance += throughput * (1.0 - refl) * shade(h.pos, nn, albedo);
        throughput *= refl * mix(vec3f(1.0), albedo, 0.15);
        ro = h.pos + nn * 0.01;
        rd = reflect(rd, nn);
        continue;
      }
      radiance += throughput * litDiffuse(h.pos, nn, albedo);
      if (giActive() && giUsed < i32(u.giBounces)) {
        giUsed = giUsed + 1;
        diffuseGI = true;
        throughput *= albedo;
        ro = h.pos + nn * 0.02;
        rd = cosineHemisphere(nn);
        continue;
      }
      break;
    } else if (h.kind == 5) {
      // Imported mesh triangle — PBR base color (factor × vertex color × texture).
      let mtl = materials[h.matId];
      if (mtl.emis.w > 0.5) { ro = h.pos + rd * 0.02; continue; } // hidden material → pass through
      var nn = n;
      if (dot(nn, rd) > 0.0) { nn = -nn; }
      var albedo = mtl.base.rgb * h.color; // base factor × vertex color
      let layer = i32(mtl.params.z);
      if (layer >= 0) {
        let tx = textureSampleLevel(meshTex, meshSamp, h.uv, layer, 0.0);
        albedo *= pow(tx.rgb, vec3f(2.2)); // sRGB → linear
      }
      let metallic = mtl.params.x;
      let roughness = mtl.params.y;
      radiance += throughput * mtl.emis.rgb;
      if (metallic > 0.02) {
        // Colored metal: deposit the matte fraction, then tint the reflection by
        // albedo and let roughness blur it a lot at the top end so rough metals
        // read matte instead of chrome. (GI is applied only to matte surfaces.)
        radiance += throughput * (1.0 - metallic) * shade(h.pos, nn, albedo);
        throughput *= metallic * mix(vec3f(1.0), albedo, 0.65);
        let refl = reflect(rd, nn);
        rd = coneSample(refl, clamp(roughness * roughness * 1.5, 0.02, 1.2));
        ro = h.pos + nn * 0.02;
        continue;
      }
      radiance += throughput * litDiffuse(h.pos, nn, albedo);
      if (giActive() && giUsed < i32(u.giBounces)) {
        giUsed = giUsed + 1;
        diffuseGI = true;
        throughput *= albedo;
        ro = h.pos + nn * 0.02;
        rd = cosineHemisphere(nn);
        continue;
      }
      break;
    } else {
      // Water — Fresnel blend of reflection and a diffuse body term.
      let cosI = max(dot(-rd, n), 0.0);
      let F = fresnelSchlick(cosI, world.water.w);
      let deep = world.water.rgb;
      let ndl = max(dot(n, normalize(u.sunDir)), 0.0) * u.sunFlag;
      radiance += throughput * (1.0 - F) * deep * (0.3 + 0.7 * ndl) * skyLightAmt();
      throughput *= F * 0.92;
      ro = h.pos + n * 0.01;
      rd = reflect(rd, n);
      continue;
    }
  }

  if (primaryDist > 0.0 && primaryDist < 1e8) {
    let fog = 1.0 - exp(-primaryDist * world.haze.x);
    let hazeCol = mix(world.horizon.rgb * skyLightAmt(), sky(rd0), 0.4);
    radiance = mix(radiance, hazeCol, fog * 0.85);
  }
  // Volumetric shafts layer on top of the (already hazed) scene as additive
  // in-scattered sunlight.
  radiance += godRays(ro0, rd0, primaryDist);
  radiance += particleEmission(ro0, rd0, primaryDist);
  return radiance;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let px = gid.xy;
  if (f32(px.x) >= u.res.x || f32(px.y) >= u.res.y) { return; }
  seedRng(px);

  let jitter = vec2f(rand(), rand()) - 0.5;
  let p = (vec2f(px) + 0.5 + jitter);
  let ndc = vec2f(
    (p.x / u.res.x) * 2.0 - 1.0,
    1.0 - (p.y / u.res.y) * 2.0,
  );
  let dir = normalize(
    u.camFwd
    + u.camRight * (ndc.x * u.aspect * u.fovScale)
    + u.camUp * (ndc.y * u.fovScale)
  );

  // Pick pass: a single dispatch with pickX >= 0. Only the clicked pixel does
  // work; it records the primary hit's kind + index and nothing accumulates.
  if (u.pickX >= 0.0) {
    if (i32(px.x) == i32(u.pickX) && i32(px.y) == i32(u.pickY)) {
      let ph = intersect(u.camPos, dir);
      pickResult[0] = ph.kind;
      pickResult[1] = ph.primIdx;
      // Focal-plane distance (projected onto the camera forward axis) so a
      // click-to-focus lands the focal plane exactly on the hit point.
      pickResult[2] = bitcast<i32>(ph.t * dot(dir, u.camFwd));
    }
    return;
  }

  // Thin-lens depth of field: jitter the ray origin across a disk of radius
  // `aperture` and re-aim it at the focal plane, so the focal plane stays sharp
  // while everything nearer/farther blurs as samples accumulate.
  var ro = u.camPos;
  var rd = dir;
  if (u.aperture > 0.0) {
    let ft = u.focusDist / max(dot(dir, u.camFwd), 1e-4);
    let focusPoint = u.camPos + dir * ft;
    let ang = 6.2831853 * rand();
    let rad = u.aperture * sqrt(rand());
    let lens = u.camRight * (cos(ang) * rad) + u.camUp * (sin(ang) * rad);
    ro = u.camPos + lens;
    rd = normalize(focusPoint - ro);
  }
  let color = trace(ro, rd);

  let idx = px.y * u32(u.res.x) + px.x;
  if (u.frame < 0.5) {
    accum[idx] = vec4f(color, 1.0);
  } else {
    accum[idx] += vec4f(color, 1.0);
  }
}
