// Cutscene timeline: an ordered list of camera keyframes the view interpolates
// through. Pure data + evaluation here (no camera/scene refs) so it's easy to
// test and reuse for both live playback and offline (WebM) rendering.

export type Ease = "linear" | "smooth";

/** One skeleton bone's local rotation within a posed keyframe. `i` indexes the
 *  instance's skeleton bone array (stable for a given rig). */
export interface BonePose {
  i: number;
  q: [number, number, number, number]; // local quaternion (xyzw)
}

/** A snapshot of one object's transform within a keyframe (rotation as a
 *  quaternion so it can be slerped without gimbal flips). `pose`, when present,
 *  is the object's skeleton bones — so a rigged character animates at the joints
 *  across a shot, not just as a rigid body. */
export interface ObjXform {
  id: number;
  pos: [number, number, number];
  quat: [number, number, number, number];
  scale: number;
  pose?: BonePose[];
}

/** One camera keyframe. `duration` is seconds of travel from the previous key
 *  (ignored for the first key); `ease` shapes the approach into this key.
 *  `objects` optionally snapshots scene object transforms at capture time. */
export interface CamKey {
  target: [number, number, number];
  distance: number;
  yaw: number;
  pitch: number;
  aperture: number;
  focusDistance: number;
  // Atmosphere: animatable so a shot can move through the day. timeOfDay drives
  // sun + sky + exposure via applyTimeOfDay; exposure/haze layer on top.
  timeOfDay: number;
  exposure: number;
  haze: number;
  duration: number;
  ease: Ease;
  // Cubic-Bézier timing handles [x1,y1,x2,y2] (a CSS-style easing curve from
  // (0,0) to (1,1)). When present it drives the segment's timing; otherwise the
  // legacy `ease` enum is used. This is what the curve editor edits.
  bezier?: [number, number, number, number];
  objects?: ObjXform[];
}

/** The interpolated state at a point in time (camera + DoF + atmosphere + objects). */
export interface CamState {
  target: [number, number, number];
  distance: number;
  yaw: number;
  pitch: number;
  aperture: number;
  focusDistance: number;
  timeOfDay: number;
  exposure: number;
  haze: number;
  objects?: ObjXform[];
}

const clamp = (x: number, a: number, b: number) => Math.min(b, Math.max(a, x));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/** Named easing presets for the timeline (control points of a cubic-Bézier
 *  timing curve). `ease in-out` matches the old "smooth" default closely. */
export const EASE_PRESETS: { name: string; bezier: [number, number, number, number] }[] = [
  { name: "linear", bezier: [0, 0, 1, 1] },
  { name: "ease in", bezier: [0.42, 0, 1, 1] },
  { name: "ease out", bezier: [0, 0, 0.58, 1] },
  { name: "ease in-out", bezier: [0.42, 0, 0.58, 1] },
];

/** Evaluate a CSS-style cubic-Bézier easing curve at progress `x` ∈ [0,1].
 *  Given control points (x1,y1),(x2,y2) between (0,0) and (1,1), find the curve
 *  parameter t where X(t)=x, then return Y(t). Newton-Raphson with a bisection
 *  fallback — the same approach browsers use for `cubic-bezier(...)`. */
export function bezierEase(x1: number, y1: number, x2: number, y2: number, x: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const cx = 3 * x1, bx = 3 * (x2 - x1) - cx, ax = 1 - cx - bx;
  const cy = 3 * y1, by = 3 * (y2 - y1) - cy, ay = 1 - cy - by;
  const sampleX = (t: number) => ((ax * t + bx) * t + cx) * t;
  const sampleY = (t: number) => ((ay * t + by) * t + cy) * t;
  const sampleDX = (t: number) => (3 * ax * t + 2 * bx) * t + cx;

  let t = x;
  for (let i = 0; i < 8; i++) {
    const err = sampleX(t) - x;
    if (Math.abs(err) < 1e-6) return sampleY(t);
    const d = sampleDX(t);
    if (Math.abs(d) < 1e-6) break;
    t -= err / d;
  }
  let lo = 0, hi = 1;
  t = x;
  for (let i = 0; i < 30; i++) {
    const xt = sampleX(t);
    if (Math.abs(xt - x) < 1e-6) break;
    if (x > xt) lo = t; else hi = t;
    t = (lo + hi) / 2;
  }
  return sampleY(t);
}

/** Ease a segment parameter `u` for the key ending the segment: prefer the
 *  cubic-Bézier handles, else fall back to the legacy `ease` enum. */
function easeU(k: CamKey, u: number): number {
  if (k.bezier) return bezierEase(k.bezier[0], k.bezier[1], k.bezier[2], k.bezier[3], u);
  return k.ease === "smooth" ? u * u * (3 - 2 * u) : u;
}

/** Interpolate yaw the short way around the circle (no 359° spins). */
function lerpYaw(a: number, b: number, t: number): number {
  let d = (b - a) % (2 * Math.PI);
  if (d > Math.PI) d -= 2 * Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return a + d * t;
}

/** Spherical-linear interpolation of two unit quaternions (xyzw). */
function slerp(
  a: [number, number, number, number],
  b: [number, number, number, number],
  t: number,
): [number, number, number, number] {
  let [ax, ay, az, aw] = a;
  let [bx, by, bz, bw] = b;
  let cos = ax * bx + ay * by + az * bz + aw * bw;
  if (cos < 0) { bx = -bx; by = -by; bz = -bz; bw = -bw; cos = -cos; }
  if (cos > 0.9995) { // nearly parallel → normalized lerp
    const x = ax + (bx - ax) * t, y = ay + (by - ay) * t, z = az + (bz - az) * t, w = aw + (bw - aw) * t;
    const l = Math.hypot(x, y, z, w) || 1;
    return [x / l, y / l, z / l, w / l];
  }
  const ang = Math.acos(cos), s = Math.sin(ang);
  const wa = Math.sin((1 - t) * ang) / s, wb = Math.sin(t * ang) / s;
  return [ax * wa + bx * wb, ay * wa + by * wb, az * wa + bz * wb, aw * wa + bw * wb];
}

/** Interpolate two skeleton poses by bone index (bones present on only one side
 *  pass through). Returns undefined only when neither side is posed. */
function lerpPose(a: BonePose[] | undefined, b: BonePose[] | undefined, u: number): BonePose[] | undefined {
  if (!a) return b;
  if (!b) return a;
  const bm = new Map(b.map((p) => [p.i, p]));
  const out: BonePose[] = [];
  for (const pa of a) {
    const pb = bm.get(pa.i);
    out.push({ i: pa.i, q: pb ? slerp(pa.q, pb.q, u) : pa.q });
  }
  for (const pb of b) if (!out.some((p) => p.i === pb.i)) out.push(pb);
  return out;
}

/** Interpolate two object-transform sets by id (objects in only one side pass
 *  through unchanged). */
function lerpObjects(a: ObjXform[] | undefined, b: ObjXform[] | undefined, u: number): ObjXform[] | undefined {
  if (!a) return b;
  if (!b) return a;
  const bm = new Map(b.map((o) => [o.id, o]));
  const seen = new Set<number>();
  const out: ObjXform[] = [];
  for (const oa of a) {
    seen.add(oa.id);
    const ob = bm.get(oa.id);
    if (!ob) { out.push(oa); continue; }
    out.push({
      id: oa.id,
      pos: [lerp(oa.pos[0], ob.pos[0], u), lerp(oa.pos[1], ob.pos[1], u), lerp(oa.pos[2], ob.pos[2], u)],
      quat: slerp(oa.quat, ob.quat, u),
      scale: lerp(oa.scale, ob.scale, u),
      pose: lerpPose(oa.pose, ob.pose, u),
    });
  }
  for (const ob of b) if (!seen.has(ob.id)) out.push(ob);
  return out;
}

const stateOf = (k: CamKey): CamState => ({
  target: [k.target[0], k.target[1], k.target[2]],
  distance: k.distance, yaw: k.yaw, pitch: k.pitch,
  aperture: k.aperture, focusDistance: k.focusDistance,
  timeOfDay: k.timeOfDay, exposure: k.exposure, haze: k.haze,
  objects: k.objects,
});

/** Total play length in seconds (sum of per-segment durations). */
export function cutsceneDuration(keys: CamKey[]): number {
  let s = 0;
  for (let i = 1; i < keys.length; i++) s += Math.max(0, keys[i].duration);
  return s;
}

/** Absolute time (seconds) at which keyframe `idx` is reached. */
export function keyTime(keys: CamKey[], idx: number): number {
  let s = 0;
  for (let i = 1; i <= idx && i < keys.length; i++) s += Math.max(0, keys[i].duration);
  return s;
}

/** Camera state at time `t` (clamped to the timeline). Null if there are no keys. */
/**
 * Does the timeline actually animate this object's skeleton — do its keyed poses
 * differ from one key to the next?
 *
 * A keyed pose is stamped over clip playback, so a rig that carries the SAME
 * snapshot in every key would be pinned to that one frame for the whole shot
 * while its position keys carried it around: travelling without moving. When the
 * poses never vary the timeline isn't posing anything, so playback may drive the
 * joints instead. A rig keyed pose-by-pose varies, and keeps priority.
 */
export function poseVariesAcrossKeys(keys: CamKey[], id: number): boolean {
  let first: BonePose[] | undefined;
  for (const k of keys) {
    const pose = k.objects?.find((o) => o.id === id)?.pose;
    if (!pose) continue;
    if (!first) { first = pose; continue; }
    if (first.length !== pose.length) return true;
    for (let i = 0; i < pose.length; i++) {
      const a = first[i].q, b = pose[i].q;
      if (Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]) + Math.abs(a[2] - b[2]) + Math.abs(a[3] - b[3]) > 1e-6) return true;
    }
  }
  return false;
}

export function evalCutscene(keys: CamKey[], t: number): CamState | null {
  if (keys.length === 0) return null;
  if (keys.length === 1) return stateOf(keys[0]);
  const total = cutsceneDuration(keys);
  t = clamp(t, 0, total);
  let acc = 0;
  for (let i = 1; i < keys.length; i++) {
    const seg = Math.max(1e-6, keys[i].duration);
    if (t <= acc + seg || i === keys.length - 1) {
      const u = easeU(keys[i], clamp((t - acc) / seg, 0, 1));
      const a = keys[i - 1], b = keys[i];
      return {
        target: [lerp(a.target[0], b.target[0], u), lerp(a.target[1], b.target[1], u), lerp(a.target[2], b.target[2], u)],
        distance: lerp(a.distance, b.distance, u),
        yaw: lerpYaw(a.yaw, b.yaw, u),
        pitch: lerp(a.pitch, b.pitch, u),
        aperture: lerp(a.aperture, b.aperture, u),
        focusDistance: lerp(a.focusDistance, b.focusDistance, u),
        timeOfDay: lerp(a.timeOfDay, b.timeOfDay, u),
        exposure: lerp(a.exposure, b.exposure, u),
        haze: lerp(a.haze, b.haze, u),
        objects: lerpObjects(a.objects, b.objects, u),
      };
    }
    acc += seg;
  }
  return stateOf(keys[keys.length - 1]);
}
