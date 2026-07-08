import { WebGLRenderer, Scene as ThreeScene, PerspectiveCamera, DirectionalLight, HemisphereLight, Mesh, PlaneGeometry, MeshStandardMaterial, Color, Fog, AnimationMixer, Clock, MathUtils, DoubleSide, Float32BufferAttribute, } from "three";
import { LightType } from "./scene/scene";
import { previewFromBlas } from "./scene/serialize";
import { worldTerrainHeight, worldTerrainColor } from "./gen/terrainField";
import { primitiveLocalGeometry } from "./mesh/tessellate";
/**
 * Real-time rasterized preview (Bryce's fast "nanopreview"). Renders the
 * imported model — textured and animated live — over a ground plane, with the
 * sun synced from the app scene. The ray-traced view is the slow final render.
 */
export class Preview {
    canvas;
    app;
    renderer;
    scene = new ThreeScene();
    camera = new PerspectiveCamera(50, 1, 0.1, 4000);
    sun = new DirectionalLight(0xfff3e0, 3);
    instances = new Map();
    posing = new Set(); // instances whose animation is paused for manual posing
    clock = new Clock();
    // Procedural world surfaces mirrored from the ray tracer so placement in the
    // preview matches the final render. Built lazily / on worldVersion change.
    ground; // flat fallback when terrain is disabled
    terrainMesh = null;
    waterMesh = null;
    terrainSig = ""; // shape params + patch center the heights were last built for
    terrainColorSig = ""; // palette/threshold params the vertex colors were last built for
    // SDF primitives rasterized as tessellated meshes, keyed by primitive id.
    prims = new Map();
    static TERRAIN_SIZE = 3000; // world units the patch covers
    static TERRAIN_SEG = 200; // grid resolution (segments per side)
    static TERRAIN_SNAP = 250; // patch re-centers in steps this large
    constructor(canvas, app) {
        this.canvas = canvas;
        this.app = app;
        this.renderer = new WebGLRenderer({ canvas, antialias: true });
        this.scene.background = new Color(0xc7d6eb);
        this.scene.fog = new Fog(0xc7d6eb, 120, 900);
        this.ground = new Mesh(new PlaneGeometry(4000, 4000), new MeshStandardMaterial({ color: 0x5b6b45, roughness: 1 }));
        this.ground.rotation.x = -Math.PI / 2;
        this.scene.add(this.ground);
        this.scene.add(new HemisphereLight(0xbcd0ff, 0x52502f, 0.6));
        this.sun.position.set(50, 60, 30);
        this.scene.add(this.sun, this.sun.target);
    }
    addInstance(id, group, animations, blas) {
        const mixer = animations.length ? new AnimationMixer(group) : null;
        mixer?.clipAction(animations[0]).play();
        this.scene.add(group);
        // Remember the geometry this group was built from, so syncInstances can
        // detect when an undo/redo (or edit/bake) swapped the BLAS and rebuild.
        const inst = this.app.instances.find((i) => i.id === id);
        const resolvedBlas = blas !== undefined ? blas : inst ? this.app.blases[inst.blasIndex] ?? null : null;
        this.instances.set(id, { group, inner: group.children[0], mixer, blas: resolvedBlas });
    }
    /** Skeleton bones of an instance (for posing), or null if it isn't rigged. */
    getBones(id) {
        const e = this.instances.get(id);
        if (!e)
            return null;
        let bones = null;
        e.inner.traverse((o) => {
            const sk = o;
            if (sk.isSkinnedMesh && sk.skeleton && !bones)
                bones = sk.skeleton.bones;
        });
        return bones;
    }
    /** The instance's transformed scene-graph root (for world-geometry extraction). */
    instanceRoot(id) {
        return this.instances.get(id)?.group ?? null;
    }
    /** Pause/resume an instance's animation so manual bone edits aren't overwritten. */
    setPosing(id, posing) {
        if (posing)
            this.posing.add(id);
        else
            this.posing.delete(id);
    }
    isPosing(id) {
        return this.posing.has(id);
    }
    /** Put an instance's geometry in LOCAL space (instance transform removed) and
     *  return its root, so the current animation pose can be re-baked. */
    prepareBake(id) {
        const e = this.instances.get(id);
        if (!e)
            return null;
        e.group.position.set(0, 0, 0);
        e.group.quaternion.identity();
        e.group.scale.setScalar(1);
        e.group.updateMatrixWorld(true);
        return e.inner;
    }
    removeInstance(id) {
        const e = this.instances.get(id);
        if (e) {
            this.scene.remove(e.group);
            this.instances.delete(id);
        }
    }
    /** Drop every instance (e.g. before loading a saved scene). */
    clearInstances() {
        for (const id of [...this.instances.keys()])
            this.removeInstance(id);
    }
    /** Sync preview group transforms to the app instances; drop dead ones and
     *  rebuild any whose geometry changed (undo/redo, vertex edit, bake). */
    syncInstances(app) {
        const live = new Set(app.instances.map((i) => i.id));
        for (const id of [...this.instances.keys()])
            if (!live.has(id))
                this.removeInstance(id);
        for (const inst of app.instances) {
            let e = this.instances.get(inst.id);
            const currentBlas = app.blases[inst.blasIndex] ?? null;
            // Missing group (e.g. undo of a delete after its group was reaped) or a
            // swapped BLAS (regenerate / vertex edit / bake undo) → rebuild from the
            // current geometry. Unchanged geometry keeps its reference, so rigged
            // imports are left untouched here.
            if (!e || e.blas !== currentBlas) {
                if (e)
                    this.removeInstance(inst.id);
                this.addInstance(inst.id, previewFromBlas(app, inst.blasIndex), [], currentBlas);
                e = this.instances.get(inst.id);
            }
            e.group.position.copy(inst.position);
            e.group.rotation.copy(inst.rotation);
            e.group.scale.setScalar(inst.scale);
        }
    }
    setActive(active) {
        this.canvas.style.display = active ? "block" : "none";
        if (active)
            this.clock.getDelta(); // drop accumulated idle time
    }
    resize(w, h) {
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.setSize(w, h, false);
        this.camera.aspect = w / Math.max(1, h);
        this.camera.updateProjectionMatrix();
    }
    /** Mirror the ray tracer's procedural terrain + water so placement matches the
     *  final render. The terrain patch follows the camera target (snapped to a
     *  coarse grid) and is only re-displaced when its shape params or center
     *  change; water is a flat plane that just re-centers under the camera. */
    syncWorld(cam) {
        const w = this.app.world;
        if (w.terrainEnabled) {
            this.ground.visible = false;
            if (!this.terrainMesh) {
                const S = Preview.TERRAIN_SIZE, N = Preview.TERRAIN_SEG;
                const geo = new PlaneGeometry(S, S, N, N);
                geo.setAttribute("color", new Float32BufferAttribute(new Float32Array(geo.attributes.position.count * 3), 3));
                this.terrainMesh = new Mesh(geo, new MeshStandardMaterial({ roughness: 1, side: DoubleSide, vertexColors: true }));
                this.terrainMesh.rotation.x = -Math.PI / 2;
                this.scene.add(this.terrainMesh);
                this.terrainSig = "";
                this.terrainColorSig = "";
            }
            // Center the patch on the camera target, snapped so it only shifts (and
            // re-displaces) in discrete steps rather than every frame.
            const snap = Preview.TERRAIN_SNAP;
            const cx = Math.round(cam.target.x / snap) * snap;
            const cz = Math.round(cam.target.z / snap) * snap;
            this.terrainMesh.position.set(cx, 0, cz);
            const geo = this.terrainMesh.geometry;
            const pos = geo.attributes.position;
            // Heights + normals: rebuilt only when the shape or patch center changes
            // (sky/exposure/etc. also bump worldVersion but must not force a rebuild).
            const shapeSig = [
                cx, cz,
                w.terrainAmp, w.terrainFreq, w.terrainRidge, w.terrainOffset,
                w.terrainOctaves, w.terrainWarp, w.terrainWarpFreq, w.terrainSeed,
                w.terrainBasis, w.terrainFractal, w.terrainBasis2, w.terrainFractal2,
                w.terrainFreq2, w.terrainOctaves2, w.terrainWeight2,
                w.terrainTerraceSteps, w.terrainTerraceSharp,
            ].join(",");
            let reshaped = false;
            if (shapeSig !== this.terrainSig) {
                for (let i = 0; i < pos.count; i++) {
                    // Mesh rotated -90° about X: local (x, y) → world (x, -y); local Z → world Y.
                    // Add the patch center to sample the field at true world coordinates.
                    pos.setZ(i, worldTerrainHeight(cx + pos.getX(i), cz - pos.getY(i), w));
                }
                pos.needsUpdate = true;
                geo.computeVertexNormals();
                this.terrainSig = shapeSig;
                reshaped = true;
            }
            // Vertex colors: re-run when the shape changed (normals moved) or when a
            // palette/threshold param changed. Cheaper than a re-displace on its own.
            const colorSig = [
                w.terrainAmp, w.terrainOffset, w.terrainSnowLine, w.terrainSlopeRock,
                ...w.terrainLow, ...w.terrainRock, ...w.terrainHigh,
            ].join(",");
            if (reshaped || colorSig !== this.terrainColorSig) {
                const nor = geo.attributes.normal;
                const col = geo.getAttribute("color");
                const rgb = [0, 0, 0];
                for (let i = 0; i < pos.count; i++) {
                    // World-up normal component is the mesh's local +Z (rotation maps local Z → world Y).
                    worldTerrainColor(cx + pos.getX(i), pos.getZ(i), cz - pos.getY(i), nor.getZ(i), w, rgb);
                    col.setXYZ(i, rgb[0], rgb[1], rgb[2]);
                }
                col.needsUpdate = true;
                this.terrainColorSig = colorSig;
            }
            this.terrainMesh.visible = true;
        }
        else {
            this.ground.visible = true;
            if (this.terrainMesh)
                this.terrainMesh.visible = false;
        }
        if (w.waterEnabled) {
            if (!this.waterMesh) {
                this.waterMesh = new Mesh(new PlaneGeometry(8000, 8000), new MeshStandardMaterial({
                    transparent: true, opacity: 0.6, roughness: 0.15, metalness: 0.2, side: DoubleSide,
                }));
                this.waterMesh.rotation.x = -Math.PI / 2; // WATER_Y = 0
                this.scene.add(this.waterMesh);
            }
            this.waterMesh.position.set(cam.target.x, 0, cam.target.z); // keep it under the view
            this.waterMesh.material.color.setRGB(w.waterColor[0], w.waterColor[1], w.waterColor[2]);
            this.waterMesh.visible = true;
        }
        else if (this.waterMesh) {
            this.waterMesh.visible = false;
        }
    }
    /** Rasterize the scene's SDF primitives as tessellated meshes. Geometry is
     *  rebuilt only when a primitive's shape (type/size) changes; transform, color
     *  and solid/carve state refresh cheaply. Diffs against the live prim set so
     *  added/removed primitives appear/disappear on their own. */
    syncPrims() {
        const live = new Set(this.app.prims.map((p) => p.id));
        for (const [id, e] of [...this.prims]) {
            if (!live.has(id)) {
                this.scene.remove(e.mesh);
                e.mesh.geometry.dispose();
                this.prims.delete(id);
            }
        }
        for (const prim of this.app.prims) {
            const sig = [prim.type, prim.a, prim.b, prim.c].join(",");
            let e = this.prims.get(prim.id);
            if (!e || e.sig !== sig) {
                const geom = primitiveLocalGeometry(prim);
                if (e) {
                    e.mesh.geometry.dispose();
                    e.mesh.geometry = geom;
                    e.sig = sig;
                }
                else {
                    const mesh = new Mesh(geom, new MeshStandardMaterial());
                    this.scene.add(mesh);
                    e = { mesh, sig };
                    this.prims.set(prim.id, e);
                }
            }
            e.mesh.position.copy(prim.position);
            e.mesh.rotation.copy(prim.rotation);
            const mat = e.mesh.material;
            mat.color.setRGB(prim.color[0], prim.color[1], prim.color[2]);
            mat.metalness = prim.reflectivity;
            mat.roughness = Math.max(0.04, 1 - prim.reflectivity);
            // Carve (subtractive) prims aren't solid in the final render — show them as
            // translucent ghosts so the boolean volume is visible but distinct.
            mat.transparent = prim.subtractive;
            mat.opacity = prim.subtractive ? 0.3 : 1;
        }
    }
    render(cam) {
        // Terrain/water depend on the camera (patch follows the target), so refresh
        // every frame; the heavy re-displacement is guarded inside syncWorld.
        this.syncWorld(cam);
        this.syncPrims();
        const dt = this.clock.getDelta();
        for (const [id, e] of this.instances) {
            if (!this.posing.has(id))
                e.mixer?.update(dt);
        }
        this.camera.position.copy(cam.position);
        this.camera.up.copy(cam.up);
        this.camera.lookAt(cam.target);
        this.camera.fov = MathUtils.radToDeg(cam.fovY);
        this.camera.updateProjectionMatrix();
        // Sync the sun from the first directional light (illumination = -travel).
        const dir = this.app.lights.find((l) => l.type === LightType.Directional);
        if (dir) {
            this.sun.position.copy(dir.direction).multiplyScalar(100);
            this.sun.color.setRGB(dir.color[0], dir.color[1], dir.color[2]);
            this.sun.intensity = dir.intensity;
        }
        else {
            this.sun.intensity = 0; // no sun in the scene → genuinely dark
        }
        this.renderer.render(this.scene, this.camera);
    }
}
