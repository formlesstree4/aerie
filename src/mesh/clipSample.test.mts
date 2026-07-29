// Does absolute-time sampling actually produce distinct, deterministic, correctly
// looping poses? The WebM fix rests entirely on this.
import { Bone, AnimationMixer, AnimationClip, QuaternionKeyframeTrack, Quaternion, Vector3, Object3D } from "three";

const root = new Object3D();
const bone = new Bone(); bone.name = "Arm"; root.add(bone);
const AXIS = new Vector3(0, 0, 1);
const q = (deg: number) => new Quaternion().setFromAxisAngle(AXIS, (deg * Math.PI) / 180);
const at = [0, 1, 2];
const vals: number[] = [];
for (const d of [0, 90, 0]) { const k = q(d); vals.push(k.x, k.y, k.z, k.w); }
const clip = new AnimationClip("wave", 2, [new QuaternionKeyframeTrack("Arm.quaternion", at, vals)]);
const mixer = new AnimationMixer(root);
const action = mixer.clipAction(clip); action.play();

const angle = () => { const e = 2 * Math.acos(Math.min(1, Math.abs(bone.quaternion.w))); return (e * 180) / Math.PI; };
const sampleAt = (t: number) => { mixer.setTime(Math.max(0, t)); return angle(); };

let fail = 0;
const check = (label: string, ok: boolean, detail: string) => { if (!ok) fail++; console.log(`${ok ? "PASS" : "FAIL"}  ${label} — ${detail}`); };

const a0 = sampleAt(0), a05 = sampleAt(0.5), a1 = sampleAt(1.0), a15 = sampleAt(1.5);
// The clip is symmetric (0°→90°→0°), so 0.5s and 1.5s legitimately match; what
// matters is that consecutive frames are NOT all the same pose.
check("frames differ across the clip (not stuck on frame one)",
  new Set([a0, a05, a1, a15].map((v) => v.toFixed(3))).size >= 3, `${a0.toFixed(1)}° ${a05.toFixed(1)}° ${a1.toFixed(1)}° ${a15.toFixed(1)}°`);
check("midpoint interpolates to ~45°", Math.abs(a05 - 45) < 1.5, `${a05.toFixed(2)}°`);
check("peak at t=1 is 90°", Math.abs(a1 - 90) < 0.01, `${a1.toFixed(3)}°`);

// Deterministic: the same time must give the same pose regardless of what was
// sampled before it (frames must not depend on render order or wall-clock).
const forward = [0, 0.25, 0.5, 0.75, 1].map(sampleAt);
const shuffled = [1, 0.5, 0, 0.75, 0.25].map(sampleAt);
const again = [0, 0.25, 0.5, 0.75, 1].map(sampleAt);
check("sampling is order-independent and repeatable",
  forward.every((v, i) => Math.abs(v - again[i]) < 1e-9) && Math.abs(shuffled[2] - forward[0]) < 1e-9, "identical on re-sample");

// Looping: a 2s clip sampled at 2.5s must equal 0.5s.
check("looping wraps (t=2.5s on a 2s clip === t=0.5s)", Math.abs(sampleAt(2.5) - a05) < 1e-6,
  `${sampleAt(2.5).toFixed(3)}° vs ${a05.toFixed(3)}°`);
check("a long time still wraps correctly (t=10.5s === t=0.5s)", Math.abs(sampleAt(10.5) - a05) < 1e-6,
  `${sampleAt(10.5).toFixed(3)}°`);

// Frame pacing: 30fps over 2s must sweep the whole clip, not sit at the start.
const frames = Array.from({ length: 61 }, (_, i) => sampleAt(i / 30));
check("a 2s/30fps export sweeps the full range", Math.max(...frames) > 89 && Math.min(...frames) < 1,
  `min ${Math.min(...frames).toFixed(1)}° max ${Math.max(...frames).toFixed(1)}°`);
console.log(fail === 0 ? "\nAbsolute-time sampling is sound." : `\n${fail} FAILURE(S)`);
process.exit(fail ? 1 : 0);
