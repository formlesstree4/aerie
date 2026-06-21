// Resolves the accumulation buffer to the screen: average the samples, apply a
// gentle filmic tonemap + a warm lift to evoke Bryce's slightly dreamy color
// response, then gamma-encode.

struct Uniforms {
  camPos    : vec3f, fovScale    : f32,
  camFwd    : vec3f, aspect      : f32,
  camRight  : vec3f, frame       : f32,
  camUp     : vec3f, seed        : f32,
  sunDir    : vec3f, time        : f32,
  res       : vec2f, sampleCount : f32, _pad0 : f32,
  primCount : f32,  lightCount  : f32, instCount : f32, _pad2 : f32,
  pickX     : f32,  pickY       : f32, exposure  : f32, warmth : f32,
};

@group(0) @binding(0) var<uniform> u : Uniforms;
@group(0) @binding(1) var<storage, read> accum : array<vec4f>;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  // Fullscreen triangle.
  let p = array<vec2f, 3>(
    vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0),
  );
  return vec4f(p[vi], 0.0, 1.0);
}

// Filmic curve (ACES approximation by Krzysztof Narkowicz).
fn aces(x: vec3f) -> vec3f {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3f(0.0), vec3f(1.0));
}

@fragment
fn fs(@builtin(position) fragPos: vec4f) -> @location(0) vec4f {
  let px = vec2u(u32(fragPos.x), u32(fragPos.y));
  let idx = px.y * u32(u.res.x) + px.x;
  var hdr = accum[idx].rgb / max(u.sampleCount, 1.0);

  hdr *= u.exposure;                   // exposure
  var col = aces(hdr);
  col = mix(col, col * vec3f(1.04, 1.0, 0.96), u.warmth); // warm lift
  col = pow(col, vec3f(1.0 / 2.2));    // gamma
  return vec4f(col, 1.0);
}
