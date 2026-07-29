import { Object3D, Vector3, Quaternion } from "three";

/**
 * Inverse kinematics for posing skeletons. Operates directly on three.js bone
 * `Object3D`s (the same `SkinnedMesh` skeleton bones the pose overlay draws), so
 * a solved pose flows straight into the preview mixer and the pose bake.
 *
 * Two solvers:
 *  - `solveTwoBoneIK` — exact solve for a shoulder–elbow–wrist (or hip–knee–ankle)
 *    triple, with a pole vector steering the elbow/knee. This is the one the
 *    "grab a hand and the limb follows" drag uses. It reconstructs the solved
 *    joint positions from the triangle, then rotates the bones onto them — no
 *    iteration, no accumulated-angle sign traps.
 *  - `solveCCD` — cyclic-coordinate-descent for arbitrary-length chains (spines,
 *    tails, ropes) where no closed form exists.
 *
 * Everything reads world space and writes back local rotations, so parents and
 * children stay consistent without rebinding the skeleton.
 */

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));

// Module-scratch — these run at pointer-move rate, so we reuse rather than churn.
const _a = new Vector3(), _b = new Vector3(), _c = new Vector3(), _t = new Vector3();
const _n = new Vector3(), _perp = new Vector3(), _poleDir = new Vector3();
const _bGoal = new Vector3(), _cGoal = new Vector3(), _u = new Vector3(), _v = new Vector3();
const _ac = new Vector3(), _at = new Vector3(), _axis = new Vector3();
const _n1 = new Vector3(), _n2 = new Vector3();
const _q = new Quaternion(), _qParent = new Quaternion(), _qParentInv = new Quaternion();
const _dqLocal = new Quaternion(), _dq = new Quaternion(), _qInv = new Quaternion();
const _localAxis = new Vector3(), _qWorld = new Quaternion();

/** Pre-multiply a bone's WORLD orientation by `q` (i.e. `worldNew = q · worldOld`)
 *  while writing back a valid local rotation. Because the bone's child offsets
 *  rotate with it, this is how we swing a limb segment onto a goal direction. */
function premultiplyWorld(bone: Object3D, q: Quaternion): void {
  if (bone.parent) bone.parent.getWorldQuaternion(_qParent);
  else _qParent.identity();
  _qParentInv.copy(_qParent).invert();
  // Same rotation expressed in the parent's frame: qParent⁻¹ · q · qParent.
  _dqLocal.copy(_qParentInv).multiply(q).multiply(_qParent);
  bone.quaternion.premultiply(_dqLocal).normalize();
}

/** Angle between two (non-normalized) vectors, robust at the extremes. */
function angleBetween(u: Vector3, v: Vector3): number {
  _n1.copy(u).normalize();
  _n2.copy(v).normalize();
  return Math.acos(clamp(_n1.dot(_n2), -1, 1));
}

/** Rotate `bone` about a WORLD-space `worldAxis` by `angle`, given the bone's
 *  current world quaternion `worldQ`; the axis is mapped into the bone's local
 *  frame and post-multiplied so the subtree follows. Used by the CCD sweep. */
function rotateBoneWorldAxis(bone: Object3D, worldAxis: Vector3, angle: number, worldQ: Quaternion): void {
  if (Math.abs(angle) < 1e-6 || worldAxis.lengthSq() < 1e-12) return;
  _qInv.copy(worldQ).invert();
  _localAxis.copy(worldAxis).applyQuaternion(_qInv).normalize();
  _dq.setFromAxisAngle(_localAxis, angle);
  bone.quaternion.multiply(_dq).normalize();
}

export interface TwoBoneOptions {
  /** World-space hint for which way the joint bends (where the elbow/knee should
   *  point). Omit to keep the limb's current bend plane. */
  pole?: Vector3;
}

/**
 * Analytic two-bone IK. Rotates `root` (shoulder/hip) and `mid` (elbow/knee) so
 * `tip` (wrist/ankle) reaches `target` as closely as its reach allows; the bend
 * direction comes from `options.pole`, or is preserved from the current pose.
 * `root → mid → tip` must be a direct parent chain. Mutates the three bones.
 */
export function solveTwoBoneIK(
  root: Object3D,
  mid: Object3D,
  tip: Object3D,
  target: Vector3,
  options: TwoBoneOptions = {},
): void {
  tip.updateWorldMatrix(true, false); // one refresh up the chain
  root.getWorldPosition(_a);
  mid.getWorldPosition(_b);
  tip.getWorldPosition(_c);
  _t.copy(target);

  const EPS = 1e-4;
  const lab = _b.distanceTo(_a); // upper segment (shoulder→elbow)
  const lcb = _c.distanceTo(_b); // lower segment (elbow→wrist)
  if (lab < EPS || lcb < EPS) return;

  // Distance from root to the solved tip, clamped into the reachable annulus so
  // the triangle is always valid (folded flat ≤ d ≤ fully extended).
  _n.copy(_t).sub(_a);
  if (_n.lengthSq() < EPS * EPS) _n.copy(_c).sub(_a); // target on the shoulder → aim along the limb
  const d = clamp(_n.length(), Math.abs(lab - lcb) + EPS, lab + lcb - EPS);
  _n.normalize(); // unit shoulder→target

  // Bend plane: the elbow is lifted off the shoulder→target line toward the pole
  // (or toward its current position when no pole is given).
  if (options.pole) _poleDir.copy(options.pole).sub(_a);
  else _poleDir.copy(_b).sub(_a);
  _perp.copy(_poleDir).addScaledVector(_n, -_poleDir.dot(_n)); // component ⟂ to _n
  if (_perp.lengthSq() < 1e-10) {
    // Pole colinear with the aim — pick any perpendicular so the limb still bends.
    _perp.set(1, 0, 0);
    if (Math.abs(_n.x) > 0.9) _perp.set(0, 1, 0);
    _perp.addScaledVector(_n, -_perp.dot(_n));
  }
  _perp.normalize();

  // Triangle: x is the elbow's projection onto the aim line, h its height above it.
  const x = (lab * lab - lcb * lcb + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(0, lab * lab - x * x));
  _bGoal.copy(_a).addScaledVector(_n, x).addScaledVector(_perp, h); // solved elbow
  _cGoal.copy(_a).addScaledVector(_n, d);                            // solved tip (clamped target)

  // Swing the upper bone so the elbow lands on its goal.
  _u.copy(_b).sub(_a).normalize();
  _v.copy(_bGoal).sub(_a).normalize();
  _q.setFromUnitVectors(_u, _v);
  premultiplyWorld(root, _q);

  // Refresh, then swing the lower bone so the tip lands on its goal.
  root.updateWorldMatrix(false, true);
  mid.getWorldPosition(_b); // now ≈ _bGoal
  tip.getWorldPosition(_c);
  _u.copy(_c).sub(_b).normalize();
  _v.copy(_cGoal).sub(_b).normalize();
  _q.setFromUnitVectors(_u, _v);
  premultiplyWorld(mid, _q);
}

/** A reasonable default pole (world space) for a two-bone chain: extend past the
 *  current elbow, keeping the joint pointing the way it already does. */
export function autoPole(root: Object3D, mid: Object3D, tip: Object3D, out: Vector3): Vector3 {
  root.getWorldPosition(_a);
  mid.getWorldPosition(_b);
  tip.getWorldPosition(_c);
  _ac.copy(_c).sub(_a);
  const len = _ac.lengthSq();
  _at.copy(_b).sub(_a);
  const s = len > 1e-10 ? _at.dot(_ac) / len : 0;
  _n1.copy(_a).addScaledVector(_ac, s); // closest point on shoulder→wrist line to the elbow
  out.copy(_b).sub(_n1);                 // bend direction, off that line
  if (out.lengthSq() < 1e-10) out.set(0, 0, 1);
  return out.normalize().multiplyScalar(_ac.length() + 1e-3).add(_b);
}

export interface CCDOptions {
  iterations?: number; // sweeps of the chain (default 10)
  tolerance?: number;  // stop once the tip is within this of the target (world units)
  damping?: number;    // 0..1 fraction of each correction to apply (default 1 = full)
}

/**
 * Cyclic-coordinate-descent IK for a chain of any length. `chain[0]` is the base
 * and the last entry is the effector that should reach `target`. Each sweep runs
 * effector→base, rotating every joint to aim the effector at the goal; converges
 * in a handful of sweeps for short chains. Good for spines, tails and ropes.
 * Mutates every joint's local rotation except the effector's.
 */
export function solveCCD(chain: Object3D[], target: Vector3, options: CCDOptions = {}): void {
  const n = chain.length;
  if (n < 2) return;
  const iterations = options.iterations ?? 10;
  const tol = options.tolerance ?? 1e-3;
  const damping = clamp(options.damping ?? 1, 0, 1);
  const effector = chain[n - 1];
  const base = chain[0];

  for (let iter = 0; iter < iterations; iter++) {
    effector.updateWorldMatrix(true, false);
    effector.getWorldPosition(_c);
    if (_c.distanceToSquared(target) <= tol * tol) break;

    for (let j = n - 2; j >= 0; j--) {
      const joint = chain[j];
      joint.getWorldPosition(_a);
      effector.getWorldPosition(_c);
      _ac.copy(_c).sub(_a);      // joint → effector
      _at.copy(target).sub(_a);  // joint → target
      if (_ac.lengthSq() < 1e-12 || _at.lengthSq() < 1e-12) continue;

      _axis.copy(_ac).cross(_at);
      if (_axis.lengthSq() < 1e-12) continue; // already colinear
      const angle = angleBetween(_ac, _at) * damping;

      joint.getWorldQuaternion(_qWorld);
      rotateBoneWorldAxis(joint, _axis.normalize(), angle, _qWorld);
      base.updateWorldMatrix(false, true); // moved effector for the next (lower) joint
    }
  }
}
