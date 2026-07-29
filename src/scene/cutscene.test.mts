/**
 * Who drives a rig's joints in a shot: the timeline's keyed poses, or a playing
 * clip? Run: `npm run test:pose`
 *
 * Cutscene keys snapshot the whole skeleton. If a rig carried the SAME snapshot
 * in every key, stamping it over clip playback pinned the rig mid-stride while
 * its position keys carried it around the shot — travelling without moving.
 */
import { poseVariesAcrossKeys, type CamKey, type BonePose } from "./cutscene.ts";

const q = (deg: number): [number, number, number, number] => {
  const r = (deg * Math.PI) / 360;
  return [0, 0, Math.sin(r), Math.cos(r)];
};
const pose = (...degs: number[]): BonePose[] => degs.map((d, i) => ({ i, q: q(d) }));
const key = (objects: CamKey["objects"]): CamKey => ({
  target: [0, 0, 0], distance: 10, yaw: 0, pitch: 0, aperture: 0, focusDistance: 10,
  timeOfDay: 12, exposure: 1, haze: 0, duration: 1, ease: "smooth", objects,
} as CamKey);

let fail = 0;
const check = (label: string, got: unknown, want: unknown) => {
  const ok = got === want; if (!ok) fail++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label} — got ${got}, want ${want}`);
};

// The reported bug: camera/position keys captured while a clip played, so every
// key holds an identical skeleton snapshot.
const pinned = [
  key([{ id: 1, pos: [0, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 30, 60) }]),
  key([{ id: 1, pos: [5, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 30, 60) }]),
  key([{ id: 1, pos: [9, 0, 2], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 30, 60) }]),
];
check("identical pose in every key → the timeline is NOT posing (clip may drive)",
  poseVariesAcrossKeys(pinned, 1), false);

// A rig the user genuinely posed key by key keeps priority.
const posed = [
  key([{ id: 1, pos: [0, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 30, 60) }]),
  key([{ id: 1, pos: [5, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 30, 61) }]),
];
check("a joint keyed differently → the timeline DOES pose it", poseVariesAcrossKeys(posed, 1), true);

// Sub-threshold jitter (float noise from a round-trip through save/load) is not
// a deliberate pose change.
const jittery = [
  key([{ id: 1, pos: [0, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: [{ i: 0, q: [0, 0, 0, 1] }] }]),
  key([{ id: 1, pos: [1, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: [{ i: 0, q: [1e-8, 0, 0, 1] }] }]),
];
check("float jitter below tolerance doesn't count as posing", poseVariesAcrossKeys(jittery, 1), false);

// Independence between rigs sharing a shot.
const twoRigs = [
  key([
    { id: 1, pos: [0, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 0) },
    { id: 2, pos: [0, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 0) },
  ]),
  key([
    { id: 1, pos: [0, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 45) },
    { id: 2, pos: [3, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 0) },
  ]),
];
check("one posed rig doesn't drag another into keyed mode (posed)", poseVariesAcrossKeys(twoRigs, 1), true);
check("one posed rig doesn't drag another into keyed mode (clip-driven)", poseVariesAcrossKeys(twoRigs, 2), false);

// Degenerate shapes must not throw or falsely claim posing.
check("no keys at all", poseVariesAcrossKeys([], 1), false);
check("keys with no objects", poseVariesAcrossKeys([key(undefined)], 1), false);
check("a single key can't vary", poseVariesAcrossKeys([pinned[0]], 1), false);
check("unknown object id", poseVariesAcrossKeys(pinned, 99), false);
// Keys captured after the fix carry no pose at all for clip-driven rigs; a shot
// that mixes those with older keyed ones must still read as un-posed.
const mixed = [
  key([{ id: 1, pos: [0, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 30) }]),
  key([{ id: 1, pos: [4, 0, 0], quat: [0, 0, 0, 1], scale: 1 }]), // captured while a clip played
];
check("a key without a pose is skipped, not treated as a change", poseVariesAcrossKeys(mixed, 1), false);
// A bone count that changes (rig re-imported mid-shot) counts as posed rather
// than silently comparing mismatched arrays.
const regrown = [
  key([{ id: 1, pos: [0, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 30) }]),
  key([{ id: 1, pos: [0, 0, 0], quat: [0, 0, 0, 1], scale: 1, pose: pose(0, 30, 60) }]),
];
check("a changed bone count counts as posed", poseVariesAcrossKeys(regrown, 1), true);

console.log(fail === 0 ? "\nKeyed-pose precedence is correct." : `\n${fail} FAILURE(S)`);
process.exit(fail ? 1 : 0);
