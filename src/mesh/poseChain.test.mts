/**
 * Chain-selection tests for the skeleton poser. Run: `npm run test:pose`
 *
 * The skeleton below is shaped like a real character rig — clavicles, twist /
 * roll helper stubs, three-bone fingers, an attachment socket, a tail — because
 * every bug this file has caught came from a rig detail a clean test skeleton
 * doesn't have.
 */
import { Bone } from "three";
import { ikChain } from "./poseChain.ts";

const bones: Bone[] = [];
const bone = (name: string, parent: Bone | null, o: [number,number,number]) => {
  const b = new Bone(); b.name = name; b.position.set(...o); parent?.add(b); bones.push(b); return b;
};
const hips = bone("Hips", null, [0, 1, 0]);
const spine = bone("Spine", hips, [0, 0.12, 0]);
const spine1 = bone("Spine1", spine, [0, 0.13, 0]);
const chest = bone("Spine2", spine1, [0, 0.13, 0]);
const neck = bone("Neck", chest, [0, 0.12, 0]);
const head = bone("Head", neck, [0, 0.10, 0]);
const clav = bone("LeftShoulder", chest, [0.06, 0.09, 0]);
const upper = bone("LeftArm", clav, [0.11, 0, 0]);
bone("LeftArmTwist", upper, [0.14, 0, 0]);            // helper stub on the upper arm
const fore = bone("LeftForeArm", upper, [0.28, 0, 0]);
bone("LeftForeArmTwist1", fore, [0.10, 0, 0]);        // helper stubs on the forearm
bone("LeftForeArmTwist2", fore, [0.18, 0, 0]);
const hand = bone("LeftHand", fore, [0.26, 0, 0]);
// Three full fingers (3 bones each) + a leaf attachment point.
const fingers = ["Index", "Middle", "Ring"].map((n) => {
  const a = bone(`Left${n}1`, hand, [0.08, 0, 0]);
  const b = bone(`Left${n}2`, a, [0.04, 0, 0]);
  return { a, b, c: bone(`Left${n}3`, b, [0.03, 0, 0]) };
});
bone("LeftWeaponSocket", hand, [0.05, 0, 0]);         // leaf attachment point
let tp: Bone = hips; const tail: Bone[] = [];
for (let i = 0; i < 7; i++) tail.push((tp = bone(`Tail${i}`, tp, [0, 0, -0.12])));
const thigh = bone("LeftUpLeg", hips, [0.09, -0.05, 0]);
bone("LeftThighTwist", thigh, [0, -0.2, 0]);          // helper stub on the thigh
const shin = bone("LeftLeg", thigh, [0, -0.42, 0]);
const foot = bone("LeftFoot", shin, [0, -0.40, 0]);
bone("LeftToe", foot, [0, -0.05, 0.08]);
const rthigh = bone("RightUpLeg", hips, [-0.09, -0.05, 0]);
const rshin = bone("RightLeg", rthigh, [0, -0.42, 0]);
bone("RightFoot", rshin, [0, -0.40, 0]);
hips.updateWorldMatrix(true, true);

const names = (arr: any[]) => arr.map((b) => b.name).join(" → ");
let fail = 0;
const check = (label: string, got: any, want: any) => {
  const ok = got === want; if (!ok) fail++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}\n        got  ${got}${ok ? "" : `\n        want ${want}`}`);
};

check("grab ELBOW → upper arm swings (twist stubs don't stop the chain)",
  names(ikChain(bones, bones.indexOf(fore))), "LeftArm → LeftForeArm");
check("grab WRIST → exact two-bone",
  names(ikChain(bones, bones.indexOf(hand))), "LeftArm → LeftForeArm → LeftHand");
check("grab ANKLE → exact two-bone (thigh twist ignored)",
  names(ikChain(bones, bones.indexOf(foot))), "LeftUpLeg → LeftLeg → LeftFoot");
check("grab KNEE → thigh only, hips pinned",
  names(ikChain(bones, bones.indexOf(shin))), "LeftUpLeg → LeftLeg");
check("grab FINGERTIP → finger only, hand pinned",
  names(ikChain(bones, bones.indexOf(fingers[0].c))), "LeftIndex1 → LeftIndex2 → LeftIndex3");
check("grab TAIL TIP → whole tail curls",
  names(ikChain(bones, bones.indexOf(tail[6]))), "Tail0 → Tail1 → Tail2 → Tail3 → Tail4 → Tail5 → Tail6");
check("grab HEAD → neck only, chest pinned",
  names(ikChain(bones, bones.indexOf(head))), "Neck → Head");
check("Shift + grab ELBOW → clavicle joins in",
  names(ikChain(bones, bones.indexOf(fore), true)), "LeftShoulder → LeftArm → LeftForeArm");
check("grab a TWIST HELPER → still solves against its parent",
  ikChain(bones, bones.indexOf(bones.find((b) => b.name === "LeftForeArmTwist1")!)).length >= 2, true);
check("grab ARMATURE ROOT → nothing to solve", ikChain(bones, bones.indexOf(hips)).length < 2, true);

// No chain may pass THROUGH a real fork (a bone with 2+ substantial subtrees).
const sig = (b: Bone) => b.children.filter((c: any) => bones.includes(c) &&
  (c.children as any[]).filter((g) => bones.includes(g)).length > 0).length;
for (let i = 0; i < bones.length; i++) {
  const chain = ikChain(bones, i);
  for (let k = 1; k < chain.length - 1; k++)
    if (sig(chain[k] as Bone) > 1) { console.log(`FAIL  chain crosses fork ${chain[k].name}`); fail++; }
}
// ---------------------------------------------------------------------------
// A Rigify DEF chain, built from the measured rest pose of a real export
// (MWSC.glb): each limb bone is split into two colinear halves of equal length,
// meeting at a 0.0° seam. Bending at a seam folds the bicep in half — the bug
// this section exists to prevent.
// ---------------------------------------------------------------------------
console.log("\n— Rigify B-bone rig —");
const rig: Bone[] = [];
const rbone = (name: string, parent: Bone | null, o: [number, number, number]) => {
  const b = new Bone(); b.name = name; b.position.set(...o); parent?.add(b); rig.push(b); return b;
};
// Angles below reproduce the measured rig: shoulder 19.6°, elbow 8.3°, wrist 6.2°,
// hip 95.2°, knee 12.9°, ankle 31.1°; every seam exactly 0°.
const deg = (d: number) => (d * Math.PI) / 180;
const along = (len: number, a: number): [number, number, number] => [len * Math.cos(a), len * Math.sin(a), 0];
const rChest = rbone("ORG-spine.003", null, [0, 1.4, 0]);
rbone("ORG-neck", rChest, [0, 0.1, 0]);          // makes the chest a real fork
const org = rbone("ORG-shoulder.L", rChest, [0.05, 0.05, 0]);
let a = deg(0);
const rUarm = rbone("DEF-upper_arm.L", org, along(0.1586, a));           // shoulder joint
a += deg(19.6);
const rUarm1 = rbone("DEF-upper_arm.L.001", rUarm, along(0.1354, a));     // SEAM (mid-bicep)
const rFore = rbone("DEF-forearm.L", rUarm1, along(0.1354, a));           // elbow: same dir = 0° seam above
a += deg(8.3);
const rFore1 = rbone("DEF-forearm.L.001", rFore, along(0.1588, a));       // SEAM (mid-forearm)
const rhand = rbone("DEF-hand.L", rFore1, along(0.1588, a));             // wrist
a += deg(6.2);
rbone("DEF-f_index.01.L", rhand, along(0.03, a));
rbone("DEF-f_index.02.L", rig[rig.length - 1], along(0.025, a));
rbone("DEF-thumb.L", rhand, along(0.028, a + deg(30)));
rbone("DEF-thumb.L.001", rig[rig.length - 1], along(0.022, a + deg(30)));
let la = deg(-95.2);
const rThigh = rbone("DEF-thigh.L", rChest, [0.09, -0.35, 0]);
const rThigh1 = rbone("DEF-thigh.L.001", rThigh, along(0.1999, la));      // SEAM
const rShin = rbone("DEF-shin.L", rThigh1, along(0.1999, la));           // knee
la += deg(12.9);
const rShin1 = rbone("DEF-shin.L.001", rShin, along(0.1912, la));        // SEAM
const rFoot = rbone("DEF-foot.L", rShin1, along(0.1912, la));            // ankle
la += deg(31.1);
rbone("DEF-toe.L", rFoot, along(0.08, la));
rChest.updateWorldMatrix(true, true);

check("Rigify: grab ELBOW → upper arm swings from the SHOULDER (no fold at the seam)",
  names(ikChain(rig, rig.indexOf(rFore))), "DEF-upper_arm.L → DEF-forearm.L");
check("Rigify: grab WRIST → true two-bone (shoulder + elbow), seams skipped",
  names(ikChain(rig, rig.indexOf(rhand))), "DEF-upper_arm.L → DEF-forearm.L → DEF-hand.L");
check("Rigify: grab ANKLE → true two-bone (hip + knee)",
  names(ikChain(rig, rig.indexOf(rFoot))), "DEF-thigh.L → DEF-shin.L → DEF-foot.L");
check("Rigify: grab KNEE → thigh swings from the hip",
  names(ikChain(rig, rig.indexOf(rShin))), "DEF-thigh.L → DEF-shin.L");
check("Rigify: grab MID-BICEP (a seam bone) → still swings from the shoulder",
  names(ikChain(rig, rig.indexOf(rUarm1))), "DEF-upper_arm.L → DEF-upper_arm.L.001");
// The seams in this rig are exactly the ".001" halves of the four limb bones.
const SEAMS = new Set(["DEF-upper_arm.L.001", "DEF-forearm.L.001", "DEF-thigh.L.001", "DEF-shin.L.001"]);
const seamRotations = rig.flatMap((_, i) => {
  const c = ikChain(rig, i);
  return c.slice(0, -1).filter((b) => SEAMS.has(b.name)).map((b) => `${rig[i].name}: ${names(c)}`);
});
check(`Rigify: no chain ever rotates a seam bone${seamRotations.length ? `\n        ${seamRotations.join("\n        ")}` : ""}`,
  seamRotations.length, 0);

// A straight-armed Mixamo-style T-pose must NOT have its elbow read as a seam:
// dead colinear, but the segments differ in length and the names don't continue.
const tpose: Bone[] = [];
const tb = (name: string, parent: Bone | null, o: [number, number, number]) => {
  const b = new Bone(); b.name = name; b.position.set(...o); parent?.add(b); tpose.push(b); return b;
};
const tChest = tb("Spine2", null, [0, 1.4, 0]);
tb("Neck", tChest, [0, 0.12, 0]);
const tArm = tb("LeftArm", tChest, [0.18, 0.1, 0]);
const tFore = tb("LeftForeArm", tArm, [0.28, 0, 0]);   // perfectly straight elbow
const tHand = tb("LeftHand", tFore, [0.26, 0, 0]);     // and straight wrist
const tf1 = tb("LeftIndex1", tHand, [0.08, 0, 0]);
tb("LeftIndex2", tf1, [0.04, 0, 0]);
tChest.updateWorldMatrix(true, true);
check("T-pose: a dead-straight elbow is still an elbow (not a seam)",
  names(ikChain(tpose, tpose.indexOf(tHand))), "LeftArm → LeftForeArm → LeftHand");

console.log(fail === 0 ? "\nAll chain cases pass." : `\n${fail} FAILURE(S)`);
process.exit(fail ? 1 : 0);
