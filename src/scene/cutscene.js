// Cutscene timeline: an ordered list of camera keyframes the view interpolates
// through. Pure data + evaluation here (no camera/scene refs) so it's easy to
// test and reuse for both live playback and offline (WebM) rendering.
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const lerp = (a, b, t) => a + (b - a) * t;
/** Interpolate yaw the short way around the circle (no 359° spins). */
function lerpYaw(a, b, t) {
    let d = (b - a) % (2 * Math.PI);
    if (d > Math.PI)
        d -= 2 * Math.PI;
    if (d < -Math.PI)
        d += 2 * Math.PI;
    return a + d * t;
}
/** Spherical-linear interpolation of two unit quaternions (xyzw). */
function slerp(a, b, t) {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    let cos = ax * bx + ay * by + az * bz + aw * bw;
    if (cos < 0) {
        bx = -bx;
        by = -by;
        bz = -bz;
        bw = -bw;
        cos = -cos;
    }
    if (cos > 0.9995) { // nearly parallel → normalized lerp
        const x = ax + (bx - ax) * t, y = ay + (by - ay) * t, z = az + (bz - az) * t, w = aw + (bw - aw) * t;
        const l = Math.hypot(x, y, z, w) || 1;
        return [x / l, y / l, z / l, w / l];
    }
    const ang = Math.acos(cos), s = Math.sin(ang);
    const wa = Math.sin((1 - t) * ang) / s, wb = Math.sin(t * ang) / s;
    return [ax * wa + bx * wb, ay * wa + by * wb, az * wa + bz * wb, aw * wa + bw * wb];
}
/** Interpolate two object-transform sets by id (objects in only one side pass
 *  through unchanged). */
function lerpObjects(a, b, u) {
    if (!a)
        return b;
    if (!b)
        return a;
    const bm = new Map(b.map((o) => [o.id, o]));
    const seen = new Set();
    const out = [];
    for (const oa of a) {
        seen.add(oa.id);
        const ob = bm.get(oa.id);
        if (!ob) {
            out.push(oa);
            continue;
        }
        out.push({
            id: oa.id,
            pos: [lerp(oa.pos[0], ob.pos[0], u), lerp(oa.pos[1], ob.pos[1], u), lerp(oa.pos[2], ob.pos[2], u)],
            quat: slerp(oa.quat, ob.quat, u),
            scale: lerp(oa.scale, ob.scale, u),
        });
    }
    for (const ob of b)
        if (!seen.has(ob.id))
            out.push(ob);
    return out;
}
const stateOf = (k) => ({
    target: [k.target[0], k.target[1], k.target[2]],
    distance: k.distance, yaw: k.yaw, pitch: k.pitch,
    aperture: k.aperture, focusDistance: k.focusDistance,
    timeOfDay: k.timeOfDay, exposure: k.exposure, haze: k.haze,
    objects: k.objects,
});
/** Total play length in seconds (sum of per-segment durations). */
export function cutsceneDuration(keys) {
    let s = 0;
    for (let i = 1; i < keys.length; i++)
        s += Math.max(0, keys[i].duration);
    return s;
}
/** Absolute time (seconds) at which keyframe `idx` is reached. */
export function keyTime(keys, idx) {
    let s = 0;
    for (let i = 1; i <= idx && i < keys.length; i++)
        s += Math.max(0, keys[i].duration);
    return s;
}
/** Camera state at time `t` (clamped to the timeline). Null if there are no keys. */
export function evalCutscene(keys, t) {
    if (keys.length === 0)
        return null;
    if (keys.length === 1)
        return stateOf(keys[0]);
    const total = cutsceneDuration(keys);
    t = clamp(t, 0, total);
    let acc = 0;
    for (let i = 1; i < keys.length; i++) {
        const seg = Math.max(1e-6, keys[i].duration);
        if (t <= acc + seg || i === keys.length - 1) {
            let u = clamp((t - acc) / seg, 0, 1);
            if (keys[i].ease === "smooth")
                u = u * u * (3 - 2 * u);
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
