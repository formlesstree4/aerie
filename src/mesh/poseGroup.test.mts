// Verifies the group-drag maths: several selected joints move as a formation,
// each solving its own chain toward its own start position plus the cursor travel.
import { Bone, Vector3, Object3D } from "three";
import { ikChain } from "./poseChain.ts";
import { solveTwoBoneIK, solveCCD, autoPole } from "./ik.ts";

const bones: Bone[] = [];
const B = (name: string, parent: Bone | null, o: [number, number, number]) => {
  const b = new Bone(); b.name = name; b.position.set(...o); parent?.add(b); bones.push(b); return b;
};
const chest = B("Chest", null, [0, 1.4, 0]);
B("Neck", chest, [0, 0.12, 0]);
const arms: Record<string, Bone> = {};
for (const [side, s] of [["L", 1], ["R", -1]] as const) {
  const clav = B(`Clav.${side}`, chest, [0.06 * s, 0.09, 0]);
  const up = B(`Upper.${side}`, clav, [0.11 * s, 0, 0]);
  const fore = B(`Fore.${side}`, up, [0.28 * s, -0.02, 0]);
  const hand = B(`Hand.${side}`, fore, [0.26 * s, -0.02, 0]);
  const f1 = B(`Index1.${side}`, hand, [0.08 * s, 0, 0]);
  const f2 = B(`Index2.${side}`, f1, [0.04 * s, 0, 0]);
  B(`Index3.${side}`, f2, [0.03 * s, 0, 0]);
  Object.assign(arms, { [`up${side}`]: up, [`fore${side}`]: fore, [`hand${side}`]: hand, [`f1${side}`]: f1, [`f2${side}`]: f2 });
}
// Bend the elbows ~40° so the arms aren't at full stretch: a straight limb is
// already at its reach limit, where every drag clamps and no solver can converge.
for (const side of ["L", "R"]) {
  (arms[`fore${side}`] as Bone).rotation.z = (side === "L" ? -1 : 1) * 0.7;
}
chest.updateWorldMatrix(true, true);

const W = (b: Object3D) => b.getWorldPosition(new Vector3());
const set = new Set<Object3D>(bones);
const depth = (b: Object3D) => { let d = 0; for (let p = b.parent; p && set.has(p); p = p.parent) d++; return d; };

// The multi-drag, exactly as main.ts runs it.
function multiDrag(sel: Bone[], delta: Vector3) {
  const grabs = sel.map((bone) => {
    const chain = ikChain(bones, bones.indexOf(bone));
    return { bone, start: W(bone), chain, pole: chain.length === 3 ? autoPole(chain[0], chain[1], chain[2], new Vector3()) : null, depth: depth(bone) };
  }).filter((g) => g.chain.length >= 2);
  grabs.sort((a, b) => a.depth - b.depth);
  for (const g of grabs) {
    const target = g.start.clone().add(delta);
    if (g.chain.length === 3 && g.pole) solveTwoBoneIK(g.chain[0], g.chain[1], g.chain[2], target, { pole: g.pole });
    else solveCCD(g.chain, target, { iterations: 8, damping: 0.5 });
  }
  chest.updateWorldMatrix(true, true);
  return grabs;
}

let fail = 0;
const check = (label: string, ok: boolean, detail: string) => { if (!ok) fail++; console.log(`${ok ? "PASS" : "FAIL"}  ${label} — ${detail}`); };

// 1) Both hands, dragged forward together.
const beforeL = W(arms.handL), beforeR = W(arms.handR);
const gap = beforeL.distanceTo(beforeR);
const delta = new Vector3(0, 0.05, 0.12);
multiDrag([arms.handL as Bone, arms.handR as Bone], delta);
const afterL = W(arms.handL), afterR = W(arms.handR);
check("both hands reach their own targets", afterL.distanceTo(beforeL.clone().add(delta)) < 5e-3 && afterR.distanceTo(beforeR.clone().add(delta)) < 5e-3,
  `errL ${afterL.distanceTo(beforeL.clone().add(delta)).toExponential(1)} errR ${afterR.distanceTo(beforeR.clone().add(delta)).toExponential(1)}`);
check("the formation is preserved (hands keep their spacing)", Math.abs(afterL.distanceTo(afterR) - gap) < 5e-3,
  `gap ${gap.toFixed(4)} → ${afterL.distanceTo(afterR).toFixed(4)}`);
check("each hand moved by the drag delta", Math.abs(afterL.clone().sub(beforeL).length() - delta.length()) < 5e-3,
  `moved ${afterL.clone().sub(beforeL).length().toFixed(4)} of ${delta.length().toFixed(4)}`);

// 2) Overlapping selection: two joints in the SAME chain must stay stable, with
// the deeper joint (solved last) winning on its own target.
const e0 = W(arms.foreL), w0 = W(arms.handL);
const d2 = new Vector3(0.03, 0.04, 0.02);
multiDrag([arms.foreL as Bone, arms.handL as Bone], d2);
const e1 = W(arms.foreL), w1 = W(arms.handL);
check("overlapping selection stays finite and bounded", [e1, w1].every((v) => v.toArray().every(Number.isFinite)) && e1.distanceTo(e0) < 1 && w1.distanceTo(w0) < 1,
  `elbow moved ${e1.distanceTo(e0).toFixed(3)}, wrist ${w1.distanceTo(w0).toFixed(3)}`);
check("deeper joint (wrist, solved last) hits its target", w1.distanceTo(w0.clone().add(d2)) < 5e-3,
  `err ${w1.distanceTo(w0.clone().add(d2)).toExponential(1)}`);

// 3) A whole finger selected: curls without tearing (bone lengths preserved).
const lens = (a: Object3D, b: Object3D) => W(a).distanceTo(W(b));
const l1 = lens(arms.handL, arms.f1L), l2 = lens(arms.f1L, arms.f2L);
multiDrag([arms.f1L as Bone, arms.f2L as Bone], new Vector3(0, -0.02, 0.01));
check("bone lengths are preserved through a group solve", Math.abs(lens(arms.handL, arms.f1L) - l1) < 1e-6 && Math.abs(lens(arms.f1L, arms.f2L) - l2) < 1e-6,
  "rigid segments intact");

console.log(fail === 0 ? "\nGroup drag behaves as specified." : `\n${fail} FAILURE(S)`);
process.exit(fail ? 1 : 0);
