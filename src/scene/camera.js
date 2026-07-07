import { Vector3, MathUtils } from "three";
/**
 * Orbit camera. Holds a target, spherical angles, and distance, and derives the
 * basis vectors the ray tracer needs. Three.js math here is the seed of the
 * shared scene-graph foundation that slice 2 (mesh import) will build on.
 */
export class OrbitCamera {
    target = new Vector3(20, 4, 10);
    distance = 70;
    yaw = -0.6; // radians, around +Y
    pitch = 0.28; // radians, above horizon
    fovY = MathUtils.degToRad(50);
    position = new Vector3();
    forward = new Vector3();
    right = new Vector3();
    up = new Vector3();
    static WORLD_UP = new Vector3(0, 1, 0);
    /** Recompute position + basis from the orbit parameters. */
    update() {
        this.pitch = MathUtils.clamp(this.pitch, -1.4, 1.4);
        this.distance = MathUtils.clamp(this.distance, 8, 400);
        const cp = Math.cos(this.pitch);
        const offset = new Vector3(cp * Math.sin(this.yaw), Math.sin(this.pitch), cp * Math.cos(this.yaw)).multiplyScalar(this.distance);
        this.position.copy(this.target).add(offset);
        this.forward.copy(this.target).sub(this.position).normalize();
        this.right.copy(this.forward).cross(OrbitCamera.WORLD_UP).normalize();
        this.up.copy(this.right).cross(this.forward).normalize();
    }
    get fovScale() {
        return Math.tan(this.fovY * 0.5);
    }
    orbit(dx, dy) {
        this.yaw -= dx * 0.005;
        this.pitch += dy * 0.005;
    }
    /** Rotate the view in place (FPS look): keep the camera position fixed and
     *  swing the target around it. */
    lookAngles(dYaw, dPitch) {
        this.update();
        const pos = OrbitCamera.scratch2.copy(this.position);
        this.yaw += dYaw;
        this.pitch = MathUtils.clamp(this.pitch + dPitch, -1.4, 1.4);
        const cp = Math.cos(this.pitch);
        const off = OrbitCamera.scratch
            .set(cp * Math.sin(this.yaw), Math.sin(this.pitch), cp * Math.cos(this.yaw))
            .multiplyScalar(this.distance);
        this.target.copy(pos).sub(off); // position stays put; only the view rotates
    }
    look(dx, dy) {
        this.lookAngles(-dx * 0.005, dy * 0.005);
    }
    /** Fly the camera: shift the orbit target along the (horizontal) view basis. */
    moveRelative(forwardAmt, rightAmt, upAmt) {
        const f = OrbitCamera.scratch.copy(this.forward);
        f.y = 0;
        f.normalize();
        this.target.addScaledVector(f, forwardAmt).addScaledVector(this.right, rightAmt);
        this.target.y += upAmt;
    }
    static scratch = new Vector3();
    static scratch2 = new Vector3();
    dolly(amount) {
        this.distance *= Math.exp(amount * 0.0015);
    }
}
