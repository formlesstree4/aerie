import { Object3D, Vector3, Quaternion } from "three";

/**
 * Spring-bone dynamics for secondary motion — tails, hair, cloth, antennae. This
 * is FORWARD dynamics (not IK): each spring bone's tip is a point mass pulled
 * toward its rest position by a spring, carrying inertia and sagging under
 * gravity, then the bone is rotated to aim its child at that simulated tip. When
 * the model (or a parent bone) moves, the tip lags and swings, then settles.
 *
 * The rest orientation is captured once (when the bone becomes a spring), so the
 * motion is driven by the parent/instance moving in the world — the primary
 * source of secondary motion. Stateful per bone; step once per frame.
 */

export interface SpringParams {
  stiffness: number; // 0..1 — pull back toward rest each step
  damping: number;   // 0..1 — velocity bled off each step
  gravity: number;   // world units — downward sag
}

export const DEFAULT_SPRING: SpringParams = { stiffness: 0.2, damping: 0.2, gravity: 0.5 };

export interface SpringBoneState {
  bone: Object3D;
  restQuat: Quaternion; // local rotation the bone returns to (captured at designation)
  tipLocal: Vector3;    // unit direction to the child, in bone-local space
  boneLen: number;      // distance to the child (constant)
  tip: Vector3;         // simulated child world position
  prevTip: Vector3;     // previous, for Verlet velocity
  inited: boolean;
  depth: number;        // ancestor count — step shallower bones first
}

const _head = new Vector3(), _rd = new Vector3(), _restTip = new Vector3();
const _vel = new Vector3(), _newTip = new Vector3(), _aim = new Vector3(), _tmp = new Vector3();
const _pq = new Quaternion(), _rw = new Quaternion(), _delta = new Quaternion(), _inv = new Quaternion(), _nw = new Quaternion();

/** Designate `bone` a spring, aiming at a child whose local position is `childLocal`. */
export function makeSpringBone(bone: Object3D, childLocal: Vector3): SpringBoneState {
  let depth = 0;
  for (let p = bone.parent; p; p = p.parent) depth++;
  return {
    bone,
    restQuat: bone.quaternion.clone(),
    tipLocal: childLocal.clone().normalize(),
    boneLen: childLocal.length() || 1,
    tip: new Vector3(),
    prevTip: new Vector3(),
    inited: false,
    depth,
  };
}

/** Advance one spring bone by `dt` seconds, writing the resulting local rotation.
 *  Reads the parent's current world transform, so parents must be stepped first. */
export function stepSpringBone(s: SpringBoneState, p: SpringParams, dt: number): void {
  const b = s.bone;
  b.updateWorldMatrix(true, false); // ancestors + self current
  b.getWorldPosition(_head);
  if (b.parent) b.parent.getWorldQuaternion(_pq); else _pq.identity();

  // Rest tip: where the child sits with the bone at its rest rotation.
  _rw.copy(_pq).multiply(s.restQuat);
  _rd.copy(s.tipLocal).applyQuaternion(_rw).normalize();
  _restTip.copy(_head).addScaledVector(_rd, s.boneLen);

  if (!s.inited) { s.tip.copy(_restTip); s.prevTip.copy(_restTip); s.inited = true; }

  const d = Math.min(dt, 1 / 30); // clamp so a long frame can't explode the sim
  // Verlet-ish integrate: damped inertia + spring pull to rest + gravity.
  _vel.copy(s.tip).sub(s.prevTip).multiplyScalar(1 - p.damping);
  _newTip.copy(s.tip).add(_vel);
  _newTip.addScaledVector(_tmp.copy(_restTip).sub(s.tip), p.stiffness);
  _newTip.y -= p.gravity * d;

  // Keep the tip on the sphere of radius boneLen around the (rigid) head.
  _aim.copy(_newTip).sub(_head);
  const len = _aim.length() || 1;
  _aim.multiplyScalar(1 / len);
  _newTip.copy(_head).addScaledVector(_aim, s.boneLen);

  s.prevTip.copy(s.tip);
  s.tip.copy(_newTip);

  // Rotate the bone so its child points along the simulated aim instead of rest.
  _delta.setFromUnitVectors(_rd, _aim);
  _nw.copy(_delta).multiply(_rw);            // new world orientation
  _inv.copy(_pq).invert();
  b.quaternion.copy(_inv).multiply(_nw).normalize(); // back to local
  b.updateWorldMatrix(false, true);          // push to children for the next step
}

/** Reset a spring to its rest pose (e.g. when disabling it). */
export function resetSpringBone(s: SpringBoneState): void {
  s.bone.quaternion.copy(s.restQuat);
  s.inited = false;
}
