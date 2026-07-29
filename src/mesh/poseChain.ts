import { Object3D, Vector3 } from "three";

/**
 * Works out what a drag on a skeleton joint should move.
 *
 * Grabbing a joint drags it to the cursor and everything hanging off it — the
 * rest of the arm, the hand, the fingers — rides along rigidly, because they are
 * its children. The only real question is how far UP the edit is felt, and this
 * module answers it from the skeleton's own shape so there is no "reach" dial to
 * set before you drag.
 *
 * The chain stops:
 *  - at a BRANCH POINT — a bone with more than one bone child (chest, hips, the
 *    palm above the fingers). That bone anchors the limb, so bending an elbow
 *    can never twist the spine.
 *  - after `LIMB_ROTATE` bones, keeping a limb drag to the classic two-bone solve.
 *  - except for a long single-child run (tail, tentacle, rope), which curls along
 *    its whole length.
 *
 * A short proximal stub — a clavicle, roughly half the length of an upper arm —
 * is dropped, because rotating it slides the whole limb sideways and reads as the
 * torso being yanked. `extend` reaches one bone further up for when that's wanted.
 */

export const LIMB_ROTATE = 2;   // bones a limb drag may rotate (the shoulder/elbow pair)
export const CHAIN_CAP = 8;     // hard ceiling for long chains (tails, tentacles)
export const STUB_RATIO = 0.65; // a proximal bone this much shorter than the limb is a clavicle
export const STUB_DEPTH = 1;    // bones a child needs below it before it counts as a real fork
export const SEAM_ANGLE = (1.5 * Math.PI) / 180; // rest bend below this = not an articulation
export const SEAM_LEN_TOL = 0.03;                // and matching segment lengths = a split bone

const _p = new Vector3(), _q = new Vector3(), _r = new Vector3();
const _in = new Vector3(), _out = new Vector3();

/** `DEF-upper_arm.L` → `DEF-upper_arm.L.001`, or `tail.001` → `tail.002`: the
 *  numeric-suffix naming a rig uses when it subdivides one bone. */
function nameContinues(parent: string, child: string): boolean {
  if (child.startsWith(`${parent}.`)) return /^\d+$/.test(child.slice(parent.length + 1));
  const p = /^(.*)\.(\d+)$/.exec(parent), c = /^(.*)\.(\d+)$/.exec(child);
  return !!(p && c && p[1] === c[1] && Number(c[2]) === Number(p[2]) + 1);
}

/**
 * Is this bone's origin a SEAM rather than a joint?
 *
 * Rigs subdivide limb bones to smooth the skin — Blender's B-bone segments split
 * an upper arm into `DEF-upper_arm.L` + `DEF-upper_arm.L.001`, meeting at the
 * middle of the bicep. That meeting point is not an articulation, and bending
 * there folds the arm in half: the exact "goofiness" this was reported for.
 *
 * A seam is dead straight in the rest pose (measured at exactly 0.0° on a real
 * Rigify export, where the shallowest true joint was 6.2°), and is corroborated
 * by the two halves being the same length or by the rig's own numeric naming.
 * Requiring corroboration keeps a genuinely straight rest pose — a Mixamo T-pose
 * arm, where the elbow can measure ~0° — from being mistaken for a subdivision.
 */
export function isSeam(b: Object3D, set: Set<Object3D>): boolean {
  const parent = b.parent;
  if (!parent || !set.has(parent)) return false;
  const child = soleSignificantChild(b, set); // a fork or a dead end is never a seam
  if (!child) return false;
  _in.copy(b.getWorldPosition(_p)).sub(parent.getWorldPosition(_q));
  _out.copy(child.getWorldPosition(_r)).sub(_p);
  const lenIn = _in.length(), lenOut = _out.length();
  if (lenIn < 1e-6 || lenOut < 1e-6) return false;
  if (_in.divideScalar(lenIn).angleTo(_out.divideScalar(lenOut)) > SEAM_ANGLE) return false;
  const sameLength = Math.abs(lenIn - lenOut) <= SEAM_LEN_TOL * Math.max(lenIn, lenOut);
  return sameLength || nameContinues(parent.name, b.name);
}

const _a = new Vector3(), _b = new Vector3();

/** Depth of the bone subtree below `b`, counted in bones and stopping at `max`
 *  (we only ever need to know "deeper than a stub"). */
function subtreeDepth(b: Object3D, set: Set<Object3D>, max: number): number {
  let best = 0;
  for (const c of b.children) {
    if (!set.has(c)) continue;
    best = Math.max(best, 1 + subtreeDepth(c, set, max - 1));
    if (best >= max) break;
  }
  return best;
}

/**
 * How many of `b`'s children lead somewhere — children that are bones AND carry
 * a subtree of their own.
 *
 * Dense rigs (a real one we test against has 1082 bones) hang twist, roll, cloth
 * and corrective STUBS off ordinary limb bones. Counting those would make an
 * upper arm look like a branch point — the chain would stop dead at every joint
 * and a drag would rotate a single bone. A stub is a LEAF: it ends there. A real
 * fork continues (chest → neck → head, chest → arm; hand → each finger), so only
 * children that carry a bone of their own mark one.
 */
export function boneChildCount(b: Object3D, set: Set<Object3D>): number {
  let n = 0;
  for (const c of b.children) {
    if (!set.has(c)) continue;
    if (subtreeDepth(c, set, STUB_DEPTH) >= STUB_DEPTH) n++;
  }
  return n;
}

/** The one child that continues the bone's line, or null at a leaf or a fork.
 *  Leaf stubs (twist / cloth helpers) don't count, so they neither end a segment
 *  nor hide the real continuation. */
function soleSignificantChild(b: Object3D, set: Set<Object3D>): Object3D | null {
  let found: Object3D | null = null;
  for (const c of b.children) {
    if (!set.has(c) || subtreeDepth(c, set, STUB_DEPTH) < STUB_DEPTH) continue;
    if (found) return null; // fork
    found = c;
  }
  return found;
}

/** Bones that ride along rigidly when `tip` is dragged: its whole subtree. */
export function boneSubtree(tip: Object3D, set: Set<Object3D>, out = new Set<Object3D>()): Set<Object3D> {
  for (const c of tip.children) {
    if (!set.has(c)) continue;
    out.add(c);
    boneSubtree(c, set, out);
  }
  return out;
}

/** Follow the single-child path down from an articulation, through any seams, to
 *  the next real joint — i.e. the far end of one anatomical bone. */
function segmentEnd(from: Object3D, set: Set<Object3D>): Object3D {
  let cur = from;
  for (let guard = 0; guard <= CHAIN_CAP; guard++) {
    const next = soleSignificantChild(cur, set);
    if (!next) break;             // leaf or fork ends the segment
    cur = next;
    if (!isSeam(cur, set)) break; // a real joint: this is the far end
  }
  return cur;
}

/**
 * The bones a drag on `bones[effIdx]` should solve, in root→tip order. The last
 * entry is the grabbed joint (the effector); everything before it rotates. Fewer
 * than 2 entries means there's nothing useful to solve — the caller should fall
 * back to rotating the joint in place.
 */
export function ikChain(bones: Object3D[], effIdx: number, extend = false): Object3D[] {
  const set = new Set(bones);
  const tip = bones[effIdx];
  if (!tip) return [];

  // Ancestors, nearest first, up to and including the first branch point.
  const up: Object3D[] = [];
  let cur: Object3D | null = tip;
  while (up.length < CHAIN_CAP && cur?.parent && set.has(cur.parent)) {
    cur = cur.parent;
    up.push(cur);
    if (boneChildCount(cur, set) > 1) break;
  }
  // The branch point itself is the anchor — drop it, unless it's all we have (in
  // which case rotating it beats the drag doing nothing at all).
  if (up.length > 1 && boneChildCount(up[up.length - 1], set) > 1) up.pop();
  if (up.length === 0) return [];

  // A limb bends at articulations only. Seams — the midpoints of bones the rig
  // split in half — are traversed but never rotated, so an elbow drag swings the
  // whole upper arm from the shoulder instead of folding the bicep in two.
  const seam = up.map((b) => isSeam(b, set));
  const articulations = up.filter((_, i) => !seam[i]);

  // Telling a subdivided limb from a straight tail: subdividing a bone alternates
  // seam, joint, seam, joint, so two seams NEVER meet. A run of consecutive seams
  // is a uniform chain — a tail, a rope, a straight cloth strip — and is measured
  // in bones. Anything else is measured in real joints, so a five-bone arm with
  // two joints doesn't get mistaken for a tail and curled up.
  const uniform = seam.some((s, i) => s && seam[i - 1]);
  const links = uniform ? up.length : articulations.length;
  if (links > LIMB_ROTATE + 2 && up.every((b) => boneChildCount(b, set) <= 1)) {
    return [...up].reverse().concat(tip); // curl the whole run, seams and all
  }

  const bendable = !uniform && articulations.length ? articulations : up;
  let rotate = bendable.slice(0, LIMB_ROTATE + (extend ? 1 : 0));

  // World length of the anatomical segment below `rotate[k]`: from that joint to
  // the next real joint, running THROUGH any seams (and past the grabbed bone if
  // the grab landed mid-segment, so half a bicep isn't mistaken for a whole one).
  const segLen = (k: number): number => {
    const from = rotate[k];
    const below = k === 0 ? segmentEnd(from, set) : rotate[k - 1];
    return from.getWorldPosition(_a).distanceTo(below.getWorldPosition(_b));
  };
  if (!extend && rotate.length > 1) {
    const last = rotate.length - 1;
    let rest = 0;
    for (let k = 0; k < last; k++) rest += segLen(k);
    if (segLen(last) < (STUB_RATIO * rest) / last) rotate = rotate.slice(0, last);
  }
  return [...rotate].reverse().concat(tip);
}
