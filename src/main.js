import { Renderer } from "./webgpu/renderer";
import { OrbitCamera } from "./scene/camera";
import { defaultScene, Primitive, MeshInstance, Light, LightType, PrimType, PrimPattern, PRIM_LABELS } from "./scene/scene";
import { buildUI } from "./ui";
import { Preview } from "./preview";
import { importModel, bakePose, worldMesh } from "./mesh/modelImport";
import "./mesh/loaders"; // registers glTF / FBX / STL / PLY loaders
import { convertPrimitive, primitiveGeometry } from "./mesh/tessellate";
import { landscapeModel, defaultLandform, LANDSCAPE_HALF } from "./mesh/landscape";
import { evaluateHeight } from "./gen/evaluate";
import { csgBuildModel, trisToGeometry } from "./mesh/boolean";
import { serializeScene, deserializeScene, previewFromBlas, b64ToU8 } from "./scene/serialize";
import { buildEditMesh, rebuildBlas } from "./mesh/editmesh";
import { History } from "./scene/history";
import { evalCutscene, cutsceneDuration, keyTime } from "./scene/cutscene";
import { Muxer, ArrayBufferTarget } from "webm-muxer";
import { Vector3, Euler, Matrix3, Matrix4, Quaternion } from "three";
const canvas = document.getElementById("view");
const previewCanvas = document.getElementById("preview");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const hud = document.getElementById("hud");
const errBox = document.getElementById("err");
function fail(msg) {
    errBox.style.display = "grid";
    errBox.textContent = msg;
    throw new Error(msg);
}
async function main() {
    if (!navigator.gpu) {
        fail("WebGPU is not available in this browser.\n\n" +
            "Use a recent Chrome/Edge (or Firefox Nightly with WebGPU enabled), " +
            "and make sure hardware acceleration is on.");
    }
    const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
    });
    if (!adapter)
        fail("No suitable GPU adapter found.");
    // The compute tracer binds 10 storage buffers (8 scene pools + TLAS nodes +
    // TLAS order); the spec only guarantees 8, so raise the limit to what the
    // adapter actually supports.
    const NEEDED_STORAGE_BUFFERS = 10;
    const maxStorage = adapter.limits.maxStorageBuffersPerShaderStage;
    if (maxStorage < NEEDED_STORAGE_BUFFERS) {
        fail(`This GPU allows only ${maxStorage} storage buffers per shader stage; ` +
            `the renderer needs ${NEEDED_STORAGE_BUFFERS}.`);
    }
    const device = await adapter.requestDevice({
        requiredLimits: { maxStorageBuffersPerShaderStage: NEEDED_STORAGE_BUFFERS },
    });
    device.lost.then((info) => fail(`GPU device lost: ${info.message}`));
    const context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });
    const renderer = new Renderer(device, context, format);
    const cam = new OrbitCamera();
    const scene = defaultScene();
    const preview = new Preview(previewCanvas, scene);
    // scene.version at the last save / load / new — used to detect unsaved changes.
    let savedVersion = scene.version;
    let mode = "render";
    const applyMode = () => preview.setActive(mode === "preview");
    let showCarverGhosts = false; // draw negative (carve) prims as wireframe cages
    let focusPick = false; // armed by the Render lab: next viewport click sets DoF focus
    // Active viewport tool: "move" = click-select + drag-move (default); "select"
    // = left-drag box-selects objects (or vertices, in edit mode). Toggled by the
    // 1/2 hotkeys and the Tools menu. Alt+left-drag always orbits.
    let tool = "move";
    // Gizmo transform mode (translate/rotate/scale), toggled by W/E/R + Tools menu.
    // Declared here (before buildUI) because the menu bar reads it during setup.
    let gizmoMode = "translate";
    // Grid snapping for gizmo transforms (translate → grid, rotate → 15°, scale → 0.25).
    let snapEnabled = false;
    let snapGrid = 1;
    const camBookmarks = [null, null, null, null];
    // Undo/redo history (assigned once the UI exists, since it owns selection).
    let history;
    // Vertex-editing state (declared early: read by UI hooks during setup).
    let editInstance = null;
    let editMesh = null;
    // Skeleton-posing state.
    let poseInst = null;
    let poseBones = [];
    let selectedBone = -1;
    let poseRoot = -1; // topmost bone (whole-armature translation)
    const poseOrig = []; // original bone euler angles (for reset)
    const poseRootOrig = new Vector3(); // original root position (for reset)
    // Scratch quaternions for drag-to-rotate.
    const qParent = new Quaternion();
    const qWorld = new Quaternion();
    const qDeltaA = new Quaternion();
    const qDeltaB = new Quaternion();
    function togglePose(inst) {
        if (poseInst === inst) {
            preview.setPosing(inst.id, false);
            poseInst = null;
            poseBones = [];
            selectedBone = -1;
            poseRoot = -1;
            return;
        }
        const bones = preview.getBones(inst.id);
        if (!bones || bones.length === 0)
            return; // not rigged
        poseInst = inst;
        poseBones = bones;
        selectedBone = 0;
        poseOrig.length = 0;
        for (const b of bones)
            poseOrig.push([b.rotation.x, b.rotation.y, b.rotation.z]);
        // Root = first bone whose parent isn't itself a bone.
        const boneSet = new Set(bones);
        poseRoot = bones.findIndex((b) => !b.parent || !boneSet.has(b.parent));
        if (poseRoot >= 0)
            poseRootOrig.copy(bones[poseRoot].position);
        preview.setPosing(inst.id, true);
        mode = "preview"; // skinned deformation is only live in the raster preview
        applyMode();
    }
    let importing = false;
    async function doImport(file) {
        if (importing)
            return;
        importing = true;
        hud.textContent = `Importing ${file.name}…`;
        try {
            const res = await importModel(file);
            const inst = scene.addModel(res.model, cam.target.clone());
            // Keep the original bytes so save/load can re-import (rig + textured preview).
            scene.blasFile[inst.blasIndex] = { name: file.name, bytes: new Uint8Array(await file.arrayBuffer()) };
            preview.addInstance(inst.id, res.group, res.animations);
            ui.select(inst);
            mode = "preview";
            applyMode();
            history.commit();
        }
        catch (e) {
            hud.textContent = `Import failed: ${e instanceof Error ? e.message : e}`;
        }
        finally {
            importing = false;
        }
    }
    function bakeSelected() {
        const sel = ui.getSelected();
        if (!(sel instanceof MeshInstance))
            return;
        const inner = preview.prepareBake(sel.id);
        if (!inner)
            return;
        scene.replaceBlas(sel.blasIndex, bakePose(inner, scene.blasMatBase[sel.blasIndex]));
        preview.syncInstances(scene); // restore the preview group's transform
        if (mode === "preview") {
            mode = "render"; // show the baked result
            applyMode();
        }
        history.commit();
    }
    async function fileToLayer(file, size) {
        const bmp = await createImageBitmap(file);
        const cnv = document.createElement("canvas");
        cnv.width = cnv.height = size;
        const ctx2d = cnv.getContext("2d");
        ctx2d.drawImage(bmp, 0, 0, size, size);
        return new Uint8Array(ctx2d.getImageData(0, 0, size, size).data);
    }
    function onPrimImage(prim) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.addEventListener("change", async () => {
            const f = input.files?.[0];
            if (!f)
                return;
            try {
                prim.imageLayer = scene.addPrimImage(await fileToLayer(f, scene.primImageSize));
                prim.pattern = PrimPattern.Image;
                scene.touch();
                history.commit();
                ui.refresh();
            }
            catch (e) {
                hud.textContent = `Image load failed: ${e instanceof Error ? e.message : e}`;
            }
        });
        input.click();
    }
    // Convert an SDF primitive into an editable mesh and drop straight into the
    // point editor (the SDF prim is replaced by the new instance).
    function editPrim(prim) {
        const { model, group } = convertPrimitive(prim);
        const inst = scene.addModel(model, prim.position.clone());
        inst.rotation.copy(prim.rotation);
        preview.addInstance(inst.id, group, []);
        scene.remove(prim);
        scene.touchInstances();
        history.commit(); // converting a prim → mesh is one undo step
        ui.select(inst);
        toggleEdit(inst); // enter vertex editing (forces ray-traced view)
        ui.refresh();
    }
    // Upload a base-color texture for one material of an imported model. The
    // model's UVs already exist, so the image maps correctly (fixes white imports
    // whose textures were external/missing — e.g. many FBX exports).
    function onMeshTexture(blasIndex, mi) {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.addEventListener("change", async () => {
            const f = input.files?.[0];
            if (!f)
                return;
            try {
                const layer = await fileToLayer(f, scene.meshTexSize);
                scene.setMaterial(blasIndex, mi, { texLayer: scene.addMeshTexture(layer) });
                history.commit();
                ui.refresh();
            }
            catch (e) {
                hud.textContent = `Texture load failed: ${e instanceof Error ? e.message : e}`;
            }
        });
        input.click();
    }
    // Drop triangles whose (local) material id is hidden, from a 32-float tri array.
    function dropHiddenTris(tris, hidden) {
        const STRIDE = 32;
        const tcount = tris.length / STRIDE;
        const kept = [];
        for (let t = 0; t < tcount; t++)
            if (!hidden.has(tris[t * STRIDE + 30]))
                kept.push(t);
        const out = new Float32Array(kept.length * STRIDE);
        kept.forEach((t, i) => out.set(tris.subarray(t * STRIDE, (t + 1) * STRIDE), i * STRIDE));
        return out;
    }
    // Bake a boolean group (any mix of primitives + models) into one multi-material
    // CSG mesh (textures preserved), then remove the inputs.
    function bakeBoolean(groupId) {
        if (groupId <= 0)
            return;
        const inputs = [];
        const consumed = [];
        for (const p of scene.prims) {
            if (p.group !== groupId)
                continue;
            consumed.push(p);
            const desc = {
                name: p.name,
                color: [...p.color],
                metalness: p.reflectivity,
                roughness: Math.max(0.04, 1 - p.reflectivity),
                tex: null,
                normal: null,
            };
            inputs.push({ geometry: primitiveGeometry(p), descs: [desc], subtractive: p.subtractive });
        }
        preview.syncInstances(scene);
        for (const inst of scene.instances) {
            if (inst.group !== groupId)
                continue;
            const root = preview.instanceRoot(inst.id);
            if (!root)
                continue;
            consumed.push(inst);
            const { tris, matCount } = worldMesh(root);
            const descs = [];
            const hidden = new Set();
            for (let k = 0; k < matCount; k++) {
                descs.push(scene.materialDescriptor(inst.blasIndex, k));
                if (scene.getMaterial(inst.blasIndex, k).hidden)
                    hidden.add(k);
            }
            // Drop hidden-material triangles (e.g. toon-outline shells) before CSG.
            const geomTris = hidden.size ? dropHiddenTris(tris, hidden) : tris;
            inputs.push({ geometry: trisToGeometry(geomTris), descs, subtractive: inst.subtractive });
        }
        if (!inputs.some((i) => !i.subtractive)) {
            hud.textContent = `Boolean group ${groupId}: needs at least one Solid object.`;
            return;
        }
        let built;
        try {
            built = csgBuildModel(inputs, `Boolean ${groupId}`);
        }
        catch (e) {
            hud.textContent = `Boolean failed: ${e instanceof Error ? e.message : e}`;
            return;
        }
        if (!built)
            return;
        const newInst = scene.addModel(built.model, built.center);
        preview.addInstance(newInst.id, built.group, []);
        for (const it of consumed) {
            if (it instanceof MeshInstance)
                preview.removeInstance(it.id);
            scene.remove(it);
        }
        scene.touchInstances();
        ui.select(newInst);
        mode = "render";
        applyMode();
        history.commit();
        ui.refresh();
    }
    // Generate a procedural landscape as a placeable model. The landform spec
    // (recipe + seed) rides along on the instance so it can be re-authored.
    function addLandscape(type) {
        const spec = defaultLandform(type);
        const { model, group } = landscapeModel(spec.recipe, spec.seed);
        const spawn = cam.target.clone();
        spawn.y = 0;
        const inst = scene.addModel(model, spawn);
        inst.landform = spec;
        preview.addInstance(inst.id, group, []);
        ui.select(inst);
        history.commit();
        ui.refresh();
    }
    // Scatter many instances of one baked shape across a landform's surface,
    // filtered by altitude band + max slope, with random scale/spin. All clones
    // share a single BLAS (GPU-instancing style), so hundreds stay cheap.
    function scatterEcosystem(landform, o) {
        const lf = landform.landform;
        if (!lf)
            return;
        const { recipe, seed } = lf;
        const amp = recipe.amplitude;
        const H = LANDSCAPE_HALF;
        const model = landform.modelMatrix(new Matrix4());
        const nrot = new Quaternion().setFromEuler(landform.rotation); // uniform scale → rotates normals
        const e = 0.01; // normalized finite-difference step
        const up = new Vector3(0, 1, 0);
        const hAt = (u, v) => evaluateHeight(recipe, u, v, seed) * amp;
        const placements = [];
        const maxTries = o.count * 30;
        for (let tries = 0; tries < maxTries && placements.length < o.count; tries++) {
            const u = Math.random() * 2 - 1;
            const v = Math.random() * 2 - 1;
            const t = evaluateHeight(recipe, u, v, seed); // 0..1 normalized height
            if (t < o.altMin || t > o.altMax)
                continue;
            // Surface normal from local finite differences; uniform scale preserves it.
            const nLocal = new Vector3(hAt(u - e, v) - hAt(u + e, v), 2 * (e * H), hAt(u, v - e) - hAt(u, v + e)).normalize();
            if (1 - nLocal.y > o.maxSlope)
                continue;
            const pos = new Vector3(u * H, t * amp, v * H).applyMatrix4(model);
            const spin = Math.random() * Math.PI * 2;
            const rot = new Euler();
            if (o.align) {
                const nWorld = nLocal.clone().applyQuaternion(nrot).normalize();
                const q = new Quaternion().setFromUnitVectors(up, nWorld);
                q.multiply(new Quaternion().setFromAxisAngle(up, spin));
                rot.setFromQuaternion(q);
            }
            else {
                rot.y = spin;
            }
            const scale = o.scaleMin + Math.random() * Math.max(0, o.scaleMax - o.scaleMin);
            placements.push({ pos, rot, scale });
        }
        if (placements.length === 0) {
            hud.textContent = "Scatter: no spots match those slope/altitude limits.";
            return;
        }
        // Bake the chosen shape into one shared BLAS (first instance), then clone.
        const template = new Primitive(o.shape);
        template.color = [o.color[0], o.color[1], o.color[2]];
        const { model: shapeModel } = convertPrimitive(template);
        const label = PRIM_LABELS[o.shape];
        const first = scene.addModel(shapeModel, placements[0].pos.clone());
        first.rotation.copy(placements[0].rot);
        first.scale = placements[0].scale;
        first.name = `${label} scatter 1`;
        const made = [first];
        for (let i = 1; i < placements.length; i++) {
            const inst = scene.addDuplicate(first);
            inst.position.copy(placements[i].pos);
            inst.rotation.copy(placements[i].rot);
            inst.scale = placements[i].scale;
            inst.name = `${label} scatter ${i + 1}`;
            made.push(inst);
        }
        scene.touchInstances();
        preview.syncInstances(scene);
        history.commit();
        ui.setSelection(made);
        ui.refresh();
        hud.textContent = `Scattered ${made.length} ${label}(s) on ${landform.name}.`;
    }
    // Re-bake a placed landform in place from its (edited) recipe + seed.
    function regenerateLandform(inst) {
        if (!inst.landform)
            return;
        const { model, group } = landscapeModel(inst.landform.recipe, inst.landform.seed);
        // Rebase the freshly-baked tris' material ids into this instance's pool slot.
        const matBase = scene.blasMatBase[inst.blasIndex] ?? 0;
        const tris = model.blas.tris;
        for (let i = 0; i < tris.length; i += 32)
            tris[i + 30] += matBase;
        scene.replaceBlas(inst.blasIndex, model.blas);
        preview.removeInstance(inst.id);
        preview.addInstance(inst.id, group, []);
        preview.syncInstances(scene);
        history.commit();
        ui.refresh();
    }
    // Re-roll a landform's seed, then re-bake.
    function rerollLandform(inst) {
        if (!inst.landform)
            return;
        inst.landform.seed = Math.floor(Math.random() * 1000);
        regenerateLandform(inst);
    }
    // Spawn a glowing planet: a point light with a visible emissive body.
    function addPlanet() {
        const spawn = cam.target.clone();
        spawn.y += 45;
        const l = scene.addLight(LightType.Point, spawn);
        l.bodyRadius = 10;
        l.color = [1.0, 0.84, 0.62];
        l.intensity = 2200;
        l.range = 260;
        l.name = `Planet ${l.id}`;
        scene.touch();
        ui.select(l);
        history.commit();
        ui.refresh();
    }
    // ---- new / save / load scene ----
    // A small three-way modal ("Save" / "Don't save" / "Cancel"). Resolves to the
    // chosen action; backdrop click or Esc counts as Cancel.
    function askSaveChanges() {
        return new Promise((resolve) => {
            const backdrop = document.createElement("div");
            backdrop.className = "modal-backdrop";
            const box = document.createElement("div");
            box.className = "modal";
            const title = document.createElement("div");
            title.className = "modal-title";
            title.textContent = "Unsaved changes";
            const msg = document.createElement("div");
            msg.className = "modal-msg";
            msg.textContent = "Save the current scene before starting a new one?";
            const btns = document.createElement("div");
            btns.className = "modal-btns";
            const close = (val) => {
                document.removeEventListener("keydown", onKey, true);
                backdrop.remove();
                resolve(val);
            };
            const onKey = (e) => {
                if (e.key === "Escape") {
                    e.stopPropagation();
                    close("cancel");
                }
            };
            const mk = (label, val, primary = false) => {
                const b = document.createElement("button");
                b.className = "btn" + (primary ? " primary" : "");
                b.textContent = label;
                b.addEventListener("click", () => close(val));
                return b;
            };
            btns.append(mk("Save", "save", true), mk("Don't save", "discard"), mk("Cancel", "cancel"));
            box.append(title, msg, btns);
            backdrop.append(box);
            backdrop.addEventListener("click", (e) => { if (e.target === backdrop)
                close("cancel"); });
            document.addEventListener("keydown", onKey, true);
            document.body.append(backdrop);
        });
    }
    // Reset the live scene back to the vanilla default, dropping models, edit/pose
    // modes, and undo history.
    function resetScene() {
        scene.restoreState(defaultScene().captureState());
        editInstance = null;
        editMesh = null;
        poseInst = null;
        poseBones = [];
        selectedBone = -1;
        selectedVerts.clear();
        preview.clearInstances();
        preview.syncInstances(scene);
        cutsceneKeys = [];
        cutsceneMode = false;
        cutsceneSel = -1;
        cutsceneTime = 0;
        cutscenePlaying = false;
        cutsceneHome = null;
        mode = "render";
        applyMode();
        savedVersion = scene.version;
        history.reset();
        ui.select(scene.prims[0] ?? scene.lights[0] ?? null);
        ui.refresh();
        hud.textContent = "New scene";
    }
    // New Scene: offer to save first if there are unsaved changes, then reset.
    async function newScene() {
        if (scene.version !== savedVersion) {
            const choice = await askSaveChanges();
            if (choice === "cancel")
                return;
            if (choice === "save")
                saveScene();
        }
        resetScene();
    }
    function saveScene() {
        // Serialize the canonical (home) object/DoF state, not a mid-cutscene pose.
        if (cutsceneMode) {
            restoreHome();
            cutsceneTime = 0;
            cutscenePlaying = false;
            ui.refresh();
        }
        // Object ids regenerate on load, so persist cutscene object refs as positional
        // slots over [prims…, instances…] (which save/load preserves in order).
        const slotOf = new Map();
        let slot = 0;
        for (const p of scene.prims)
            slotOf.set(p.id, slot++);
        for (const m of scene.instances)
            slotOf.set(m.id, slot++);
        const cutsceneOut = cutsceneKeys.map((k) => ({
            ...k,
            objects: k.objects
                ?.map((o) => ({ ...o, id: slotOf.get(o.id) ?? -1 }))
                .filter((o) => o.id >= 0),
        }));
        const json = serializeScene(scene, {
            fileFor: (bi) => scene.blasFile[bi],
            poseFor: (id) => {
                const bones = preview.getBones(id);
                if (!bones)
                    return null;
                return {
                    bones: bones.map((b) => ({
                        p: [b.position.x, b.position.y, b.position.z],
                        q: [b.quaternion.x, b.quaternion.y, b.quaternion.z, b.quaternion.w],
                    })),
                    paused: preview.isPosing(id),
                };
            },
            cutscene: cutsceneOut,
        });
        const blob = new Blob([json], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `aerie-${Date.now()}.aerie`;
        a.click();
        URL.revokeObjectURL(url);
        savedVersion = scene.version; // scene now matches what's on disk
    }
    async function loadScene(file) {
        try {
            const data = deserializeScene(scene, await file.text());
            // Exit any per-object modes that reference now-gone objects.
            editInstance = null;
            editMesh = null;
            poseInst = null;
            poseBones = [];
            selectedBone = -1;
            preview.clearInstances();
            const savedInstances = data.mesh?.instances ?? [];
            for (let i = 0; i < scene.instances.length; i++) {
                const inst = scene.instances[i];
                const sd = savedInstances[i];
                let rebuilt = false;
                if (sd?.file) {
                    try {
                        const bytes = b64ToU8(sd.file.bytes);
                        const res = await importModel(new File([bytes], sd.file.name));
                        preview.addInstance(inst.id, res.group, res.animations);
                        scene.blasFile[inst.blasIndex] = { name: sd.file.name, bytes };
                        if (sd.pose) {
                            const bones = preview.getBones(inst.id);
                            if (bones) {
                                for (let k = 0; k < bones.length && k < sd.pose.bones.length; k++) {
                                    const bp = sd.pose.bones[k];
                                    bones[k].position.set(bp.p[0], bp.p[1], bp.p[2]);
                                    bones[k].quaternion.set(bp.q[0], bp.q[1], bp.q[2], bp.q[3]);
                                }
                                if (sd.pose.paused)
                                    preview.setPosing(inst.id, true);
                            }
                        }
                        rebuilt = true;
                    }
                    catch {
                        /* re-import failed → fall back to a baked preview below */
                    }
                }
                if (!rebuilt)
                    preview.addInstance(inst.id, previewFromBlas(scene, inst.blasIndex), []);
            }
            // Map serialized object slots back to the freshly-created objects' ids.
            const slotToId = [];
            for (const p of scene.prims)
                slotToId.push(p.id);
            for (const m of scene.instances)
                slotToId.push(m.id);
            cutsceneKeys = (Array.isArray(data.cutscene) ? data.cutscene : []).map((k) => ({
                ...k,
                objects: k.objects
                    ?.map((o) => ({ ...o, id: slotToId[o.id] ?? -1 }))
                    .filter((o) => o.id >= 0),
            }));
            cutsceneSel = cutsceneKeys.length ? 0 : -1;
            cutsceneTime = 0;
            cutscenePlaying = false;
            cutsceneMode = false;
            cutsceneHome = null;
            if (cutsceneKeys.length)
                cutsceneKeys[0].duration = 0;
            ui.select(scene.prims[0] ?? scene.instances[0] ?? scene.lights[0] ?? null);
            history.commit();
            savedVersion = scene.version; // freshly loaded scene matches its file
            ui.refresh();
            hud.textContent = `Loaded ${file.name}`;
        }
        catch (e) {
            hud.textContent = `Load failed: ${e instanceof Error ? e.message : e}`;
        }
    }
    let renderingImage = false;
    async function onRenderImage() {
        if (renderingImage)
            return;
        renderingImage = true;
        const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
        const W = 1920;
        const H = Math.round(W / aspect);
        const prevHud = hud.textContent;
        hud.textContent = `Rendering ${W}×${H} PNG…`;
        try {
            cam.update();
            const blob = await renderer.renderToImage(cam, W, H, 240);
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `aerie-${Date.now()}.png`;
            a.click();
            URL.revokeObjectURL(url);
            hud.textContent = prevHud;
        }
        catch (e) {
            hud.textContent = `Render failed: ${e instanceof Error ? e.message : e}`;
        }
        finally {
            renderingImage = false;
        }
    }
    // Resolve the scene object under a screen point (lights tested first, then a
    // GPU pick for prims/meshes). Used by drag-to-apply from the material dock.
    async function pickObjectAt(clientX, clientY) {
        const light = pickLight(clientX, clientY);
        if (light)
            return light;
        cam.update();
        const dpr = renderer.width / Math.max(1, canvas.clientWidth);
        const r = await renderer.pick(cam, clientX * dpr, clientY * dpr);
        if (r.kind === 4 && r.index >= 0 && r.index < scene.prims.length)
            return scene.prims[r.index];
        if (r.kind === 5 && r.index >= 0 && r.index < scene.instances.length)
            return scene.instances[r.index];
        return null;
    }
    // While true, the live rAF loop idles so an off-screen export (turntable /
    // cutscene) owns the GPU and uniforms.
    let offlineRendering = false;
    // Encode an off-screen render to WebM with WebCodecs. Each frame is stamped
    // with an explicit presentation timestamp (i / fps), so the clip's duration is
    // exactly frames/fps regardless of how long each frame takes to ray-trace —
    // unlike MediaRecorder, which captures in wall-clock real time.
    async function recordWebM(label, name, W, H, frames, fps, renderFrame) {
        if (offlineRendering)
            return;
        if (typeof VideoEncoder === "undefined") {
            hud.textContent = `${label}: video export needs WebCodecs (use a recent Chrome/Edge).`;
            return;
        }
        const candidates = [
            { mux: "V_VP9", enc: "vp09.00.41.08" },
            { mux: "V_VP9", enc: "vp09.00.31.08" },
            { mux: "V_VP8", enc: "vp8" },
        ];
        let codec = null;
        for (const c of candidates) {
            const ok = await VideoEncoder.isConfigSupported({ codec: c.enc, width: W, height: H, bitrate: 12_000_000, framerate: fps });
            if (ok.supported) {
                codec = c;
                break;
            }
        }
        if (!codec) {
            hud.textContent = `${label}: no supported video codec at ${W}×${H}.`;
            return;
        }
        const muxer = new Muxer({
            target: new ArrayBufferTarget(),
            video: { codec: codec.mux, width: W, height: H, frameRate: fps },
        });
        const encoder = new VideoEncoder({
            output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
            error: (e) => { hud.textContent = `${label} encode error: ${e.message}`; },
        });
        encoder.configure({ codec: codec.enc, width: W, height: H, bitrate: 12_000_000, framerate: fps });
        const cnv = document.createElement("canvas");
        cnv.width = W;
        cnv.height = H;
        const ctx = cnv.getContext("2d");
        const usPerFrame = 1_000_000 / fps;
        offlineRendering = true;
        try {
            for (let i = 0; i < frames; i++) {
                const px = await renderFrame(i);
                ctx.putImageData(new ImageData(px, W, H), 0, 0);
                const vf = new VideoFrame(cnv, { timestamp: Math.round(i * usPerFrame), duration: Math.round(usPerFrame) });
                encoder.encode(vf, { keyFrame: i % fps === 0 }); // a keyframe each second for seeking
                vf.close();
                hud.textContent = `${label} ${i + 1}/${frames} · ${W}×${H}`;
                if (encoder.encodeQueueSize > 6)
                    await new Promise((r) => setTimeout(r, 0)); // light backpressure
            }
            await encoder.flush();
            muxer.finalize();
            const blob = new Blob([muxer.target.buffer], { type: "video/webm" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = name;
            a.click();
            URL.revokeObjectURL(url);
            hud.textContent = `${label} saved · ${frames} frames @ ${fps}fps (${(frames / fps).toFixed(1)}s)`;
        }
        catch (e) {
            hud.textContent = `${label} failed: ${e instanceof Error ? e.message : e}`;
        }
        finally {
            try {
                encoder.close();
            }
            catch { /* already closed */ }
            renderer.resetAccumulation();
            offlineRendering = false;
        }
    }
    // 360° turntable around the camera target → WebM.
    async function renderTurntable(o) {
        if (offlineRendering)
            return;
        if (mode !== "render") {
            mode = "render";
            applyMode();
            ui.refresh();
        }
        const W = Math.max(16, Math.round(o.width));
        const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
        const H = Math.max(16, Math.round(W / aspect));
        const frames = Math.max(1, Math.round(o.seconds * o.fps));
        const startYaw = cam.yaw;
        await recordWebM("Turntable", `aerie-turntable-${Date.now()}.webm`, W, H, frames, o.fps, (i) => {
            cam.yaw = startYaw + (i / frames) * Math.PI * 2;
            cam.update();
            return renderer.renderToPixels(cam, W, H, o.samples);
        });
        cam.yaw = startYaw;
        cam.update();
    }
    // ---- cutscene (camera + DoF keyframe timeline) ----
    let cutsceneKeys = [];
    let cutsceneMode = false; // dock visible / editing
    let cutsceneSel = -1; // selected keyframe index
    let cutsceneTime = 0; // playhead seconds
    let cutscenePlaying = false;
    let lastCutsceneSync = 0; // throttle UI refresh during playback
    const normalizeCutscene = () => { if (cutsceneKeys.length)
        cutsceneKeys[0].duration = 0; };
    let cutsceneHome = null;
    const csQuat = new Quaternion();
    // Snapshot every animatable object's transform (prims: pos+rot; meshes: +scale).
    function captureObjects() {
        const out = [];
        for (const p of scene.prims) {
            csQuat.setFromEuler(p.rotation);
            out.push({ id: p.id, pos: [p.position.x, p.position.y, p.position.z], quat: [csQuat.x, csQuat.y, csQuat.z, csQuat.w], scale: 1 });
        }
        for (const m of scene.instances) {
            csQuat.setFromEuler(m.rotation);
            out.push({ id: m.id, pos: [m.position.x, m.position.y, m.position.z], quat: [csQuat.x, csQuat.y, csQuat.z, csQuat.w], scale: m.scale });
        }
        return out;
    }
    function captureKey() {
        return {
            target: [cam.target.x, cam.target.y, cam.target.z],
            distance: cam.distance, yaw: cam.yaw, pitch: cam.pitch,
            aperture: scene.world.aperture, focusDistance: scene.world.focusDistance,
            duration: cutsceneKeys.length === 0 ? 0 : 2,
            ease: "smooth",
            objects: captureObjects(),
        };
    }
    const objectsById = () => {
        const m = new Map();
        for (const p of scene.prims)
            m.set(p.id, p);
        for (const inst of scene.instances)
            m.set(inst.id, inst);
        return m;
    };
    function captureHome() {
        const xforms = [];
        for (const p of scene.prims)
            xforms.push({ id: p.id, pos: [p.position.x, p.position.y, p.position.z], rot: [p.rotation.x, p.rotation.y, p.rotation.z], scale: 1 });
        for (const m of scene.instances)
            xforms.push({ id: m.id, pos: [m.position.x, m.position.y, m.position.z], rot: [m.rotation.x, m.rotation.y, m.rotation.z], scale: m.scale });
        cutsceneHome = { xforms, aperture: scene.world.aperture, focusDistance: scene.world.focusDistance };
    }
    // Put objects + DoF back to their canonical (pre-cutscene) state.
    function restoreHome() {
        if (!cutsceneHome)
            return;
        const byId = objectsById();
        for (const h of cutsceneHome.xforms) {
            const obj = byId.get(h.id);
            if (!obj)
                continue;
            obj.position.set(h.pos[0], h.pos[1], h.pos[2]);
            obj.rotation.set(h.rot[0], h.rot[1], h.rot[2]);
            if (obj instanceof MeshInstance)
                obj.scale = h.scale;
        }
        scene.world.aperture = cutsceneHome.aperture;
        scene.world.focusDistance = cutsceneHome.focusDistance;
        scene.touchInstances();
        scene.touchWorld();
    }
    // Pose the camera + DoF + tracked objects at timeline time `t`.
    function applyCutsceneAt(t) {
        const s = evalCutscene(cutsceneKeys, t);
        if (!s)
            return;
        cam.target.set(s.target[0], s.target[1], s.target[2]);
        cam.distance = s.distance;
        cam.yaw = s.yaw;
        cam.pitch = s.pitch;
        cam.update();
        scene.world.aperture = s.aperture;
        scene.world.focusDistance = s.focusDistance;
        if (s.objects && s.objects.length) {
            const byId = objectsById();
            for (const o of s.objects) {
                const obj = byId.get(o.id);
                if (!obj)
                    continue;
                obj.position.set(o.pos[0], o.pos[1], o.pos[2]);
                csQuat.set(o.quat[0], o.quat[1], o.quat[2], o.quat[3]);
                obj.rotation.setFromQuaternion(csQuat);
                if (obj instanceof MeshInstance)
                    obj.scale = o.scale;
            }
            scene.touchInstances();
        }
        scene.touchWorld();
    }
    async function renderCutscene(o) {
        if (offlineRendering)
            return;
        if (cutsceneKeys.length < 2) {
            hud.textContent = "Cutscene: add at least two keyframes first.";
            return;
        }
        if (mode !== "render") {
            mode = "render";
            applyMode();
            ui.refresh();
        }
        const total = cutsceneDuration(cutsceneKeys);
        const W = Math.max(16, Math.round(o.width));
        const aspect = canvas.clientWidth / Math.max(1, canvas.clientHeight);
        const H = Math.max(16, Math.round(W / aspect));
        const frames = Math.max(2, Math.round(total * o.fps));
        await recordWebM("Cutscene", `aerie-cutscene-${Date.now()}.webm`, W, H, frames, o.fps, (i) => {
            applyCutsceneAt((total * i) / (frames - 1));
            // The live loop is paused during export, so push the per-frame state into
            // the renderer ourselves: prims (scene), mesh instance matrices, and DoF.
            renderer.uploadScene(scene);
            renderer.uploadInstances(scene);
            renderer.uploadWorld(scene);
            return renderer.renderToPixels(cam, W, H, o.samples);
        });
        applyCutsceneAt(cutsceneTime); // return camera + DoF + objects to the playhead
    }
    const cutscene = {
        active: () => cutsceneMode,
        toggle: () => {
            cutsceneMode = !cutsceneMode;
            if (cutsceneMode) {
                captureHome(); // remember canonical object/DoF state to restore on exit
            }
            else {
                cutscenePlaying = false;
                restoreHome(); // non-destructive: scene returns to its canonical state
                cutsceneHome = null;
                cutsceneTime = 0;
            }
            ui.refresh();
        },
        count: () => cutsceneKeys.length,
        selected: () => cutsceneSel,
        time: () => cutsceneTime,
        duration: () => cutsceneDuration(cutsceneKeys),
        playing: () => cutscenePlaying,
        keyInfo: (i) => ({ duration: cutsceneKeys[i].duration, ease: cutsceneKeys[i].ease, time: keyTime(cutsceneKeys, i) }),
        add: () => { cutsceneKeys.push(captureKey()); normalizeCutscene(); cutsceneSel = cutsceneKeys.length - 1; cutsceneTime = keyTime(cutsceneKeys, cutsceneSel); markDirty(); ui.refresh(); },
        remove: (i) => { cutsceneKeys.splice(i, 1); normalizeCutscene(); cutsceneSel = Math.min(cutsceneSel, cutsceneKeys.length - 1); markDirty(); ui.refresh(); },
        recapture: (i) => { const k = captureKey(); k.duration = cutsceneKeys[i].duration; k.ease = cutsceneKeys[i].ease; cutsceneKeys[i] = k; normalizeCutscene(); markDirty(); ui.refresh(); },
        select: (i) => { cutsceneSel = i; cutsceneTime = keyTime(cutsceneKeys, i); applyCutsceneAt(cutsceneTime); ui.refresh(); },
        setDuration: (i, s) => { if (i > 0) {
            cutsceneKeys[i].duration = Math.max(0, s);
            markDirty();
        } },
        setEase: (i, e) => { cutsceneKeys[i].ease = e; markDirty(); ui.refresh(); },
        scrub: (t) => { cutscenePlaying = false; cutsceneTime = t; applyCutsceneAt(t); },
        playPause: () => {
            if (cutsceneKeys.length < 2)
                return;
            if (!cutscenePlaying && cutsceneTime >= cutsceneDuration(cutsceneKeys))
                cutsceneTime = 0;
            cutscenePlaying = !cutscenePlaying;
            ui.refresh();
        },
        render: (o) => renderCutscene(o),
    };
    // Cutscene edits mark the scene dirty (it serializes into .aerie). Bump only
    // when currently clean — avoids re-uploading the scene on every slider tick.
    function markDirty() { if (scene.version === savedVersion)
        scene.touch(); }
    const ui = buildUI(scene, () => cam.target.clone(), {
        onImport: doImport,
        onToggleMode: () => {
            mode = mode === "render" ? "preview" : "render";
            applyMode();
        },
        isPreview: () => mode === "preview",
        onBakePose: bakeSelected,
        onPrimImage,
        onRenderImage,
        onToggleEdit: toggleEdit,
        onEditPrim: editPrim,
        onMeshTexture,
        onBakeBoolean: bakeBoolean,
        onAddLandscape: addLandscape,
        onScatterEcosystem: scatterEcosystem,
        onRegenerateLandform: regenerateLandform,
        onRerollLandform: rerollLandform,
        onAddPlanet: addPlanet,
        onNewScene: newScene,
        onSaveScene: saveScene,
        onLoadScene: loadScene,
        isEditing: () => editInstance !== null,
        isGhostCarvers: () => showCarverGhosts,
        onToggleGhostCarvers: () => {
            showCarverGhosts = !showCarverGhosts;
        },
        isFocusPick: () => focusPick,
        onToggleFocusPick: () => {
            focusPick = !focusPick;
            hud.textContent = focusPick ? "Click a point in the view to focus on…" : "";
        },
        onPickAt: pickObjectAt,
        onStatus: (msg) => { hud.textContent = msg; },
        onRenderTurntable: renderTurntable,
        cutscene,
        boneNames: (inst) => preview.getBones(inst.id)?.map((b, i) => b.name || `bone ${i}`) ?? [],
        isPosing: () => poseInst !== null,
        onTogglePose: togglePose,
        poseSelectedBone: () => selectedBone,
        onSelectBone: (i) => { selectedBone = i; },
        boneRotation: (i) => {
            const b = poseBones[i];
            return b ? [b.rotation.x, b.rotation.y, b.rotation.z] : [0, 0, 0];
        },
        setBoneRotation: (i, axis, v) => {
            const b = poseBones[i];
            if (!b)
                return;
            if (axis === 0)
                b.rotation.x = v;
            else if (axis === 1)
                b.rotation.y = v;
            else
                b.rotation.z = v;
        },
        resetBone: (i) => {
            const b = poseBones[i];
            const o = poseOrig[i];
            if (b && o)
                b.rotation.set(o[0], o[1], o[2]);
        },
        poseRootIndex: () => poseRoot,
        rootPosition: () => {
            const b = poseBones[poseRoot];
            return b ? [b.position.x, b.position.y, b.position.z] : [0, 0, 0];
        },
        setRootPosition: (axis, v) => {
            const b = poseBones[poseRoot];
            if (!b)
                return;
            if (axis === 0)
                b.position.x = v;
            else if (axis === 1)
                b.position.y = v;
            else
                b.position.z = v;
        },
        resetRoot: () => {
            const b = poseBones[poseRoot];
            if (b)
                b.position.copy(poseRootOrig);
        },
        onDuplicate: duplicateSelection,
        onAlign: alignSelection,
        onDistribute: distributeSelection,
        onScatter: scatterSelection,
        onMirror: mirrorSelection,
        getBookmarks: () => camBookmarks.map((b) => b !== null),
        onSaveBookmark: saveBookmark,
        onRecallBookmark: recallBookmark,
        getSnap: () => ({ enabled: snapEnabled, grid: snapGrid }),
        onToggleSnap: () => { snapEnabled = !snapEnabled; },
        onSetGrid: (v) => { snapGrid = Math.max(0.05, v); },
        getTool: () => tool,
        onSetTool: (t) => { tool = t; ui.refresh(); },
        getGizmoMode: () => gizmoMode,
        onSetGizmoMode: (m) => { gizmoMode = m; ui.refresh(); },
        onCommit: () => history.commit(),
        onEdit: () => history.commitSoon(),
    });
    history = new History(scene, () => ui.getSelection(), (items) => ui.setSelection(items), () => { ui.refresh(); renderer.resetAccumulation(); });
    applyMode();
    // Drag-and-drop import anywhere on the window.
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => {
        e.preventDefault();
        const f = e.dataTransfer?.files?.[0];
        if (!f)
            return;
        if (/\.(aerie|csbryce|json)$/i.test(f.name))
            loadScene(f);
        else
            doImport(f);
    });
    let uploadedVersion = -1;
    let uploadedMeshStruct = -1;
    let uploadedInst = -1;
    let uploadedMeshMat = -1;
    let uploadedWorld = -1;
    let uploadedPrimTex = -1;
    // ---- sizing (cap DPR so the path tracer stays interactive) ----
    const MAX_SAMPLES = 512;
    function resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
        const w = Math.floor(canvas.clientWidth * dpr);
        const h = Math.floor(canvas.clientHeight * dpr);
        preview.resize(canvas.clientWidth, canvas.clientHeight);
        overlay.width = canvas.clientWidth;
        overlay.height = canvas.clientHeight;
        if (w === renderer.width && h === renderer.height)
            return;
        canvas.width = w;
        canvas.height = h;
        renderer.resize(w, h);
    }
    new ResizeObserver(resize).observe(canvas);
    resize();
    let drag = "none";
    let lastX = 0;
    let lastY = 0;
    let dragTarget = null;
    // Box-select rectangle (CSS px, same space as projectToScreen).
    let marqStartX = 0, marqStartY = 0, marqCurX = 0, marqCurY = 0;
    const xformSnap = new Map();
    const objDragHit0 = new Vector3();
    const groupDelta = new Vector3();
    const gizmoPivot = new Vector3();
    let activeGizmoMode = "translate";
    let gizmoDownX = 0, gizmoDownY = 0;
    const rotQ = new Quaternion();
    const tmpQ = new Quaternion();
    const tmpPivot = new Vector3();
    let movedDuringDrag = false; // an actual transform happened → worth an undo step
    let draggingDirLight = false; // re-aiming a directional light's handle
    let dragVertical = false;
    const planePoint = new Vector3();
    const planeNormal = new Vector3();
    const scratch = new Vector3();
    // Axis-constrained gizmo drag.
    const WORLD_AXES = [new Vector3(1, 0, 0), new Vector3(0, 1, 0), new Vector3(0, 0, 1)];
    const dragAxis = new Vector3();
    const axisStartPos = new Vector3();
    let axisStartS = 0;
    let activeAxisIdx = -1;
    function screenRay(cx, cy) {
        const w = canvas.clientWidth;
        const h = canvas.clientHeight;
        const ndcx = (cx / w) * 2 - 1;
        const ndcy = 1 - (cy / h) * 2;
        const rd = new Vector3()
            .copy(cam.forward)
            .addScaledVector(cam.right, ndcx * (w / h) * cam.fovScale)
            .addScaledVector(cam.up, ndcy * cam.fovScale)
            .normalize();
        return { ro: cam.position, rd };
    }
    function rayPlane(ro, rd, p0, n) {
        const denom = rd.dot(n);
        if (Math.abs(denom) < 1e-6)
            return null;
        const t = scratch.copy(p0).sub(ro).dot(n) / denom;
        return t < 0 ? null : new Vector3().copy(ro).addScaledVector(rd, t);
    }
    // ---- group transform (single object or multi-selection) ----
    function snapOf(it) {
        return {
            pos: it.position.clone(),
            rot: it instanceof Light ? null : it.rotation.clone(),
            dir: it instanceof Light ? it.direction.clone() : null,
            scale: it instanceof MeshInstance ? it.scale : 1,
            a: it instanceof Primitive ? it.a : 0,
            b: it instanceof Primitive ? it.b : 0,
            c: it instanceof Primitive ? it.c : 0,
        };
    }
    /** Snapshot the transform of every item to move with `target` (the group if it
     *  belongs to a multi-selection, otherwise just itself). */
    function beginGroupSnapshot(target) {
        xformSnap.clear();
        const sel = ui.getSelection();
        const group = sel.includes(target) && sel.length > 1 ? sel : [target];
        for (const it of group)
            xformSnap.set(it, snapOf(it));
    }
    /** Snapshot the whole current selection (for pivot rotate/scale). */
    function beginSelectionSnapshot() {
        xformSnap.clear();
        for (const it of ui.getSelection())
            xformSnap.set(it, snapOf(it));
    }
    function touchAfterXform() {
        let inst = false, other = false;
        for (const it of xformSnap.keys()) {
            if (it instanceof MeshInstance)
                inst = true;
            else
                other = true;
        }
        if (inst)
            scene.touchInstances();
        if (other)
            scene.touch();
        movedDuringDrag = true;
    }
    function applyTranslate() {
        if (snapEnabled) { // quantize the movement to grid increments (keeps formation)
            groupDelta.x = Math.round(groupDelta.x / snapGrid) * snapGrid;
            groupDelta.y = Math.round(groupDelta.y / snapGrid) * snapGrid;
            groupDelta.z = Math.round(groupDelta.z / snapGrid) * snapGrid;
        }
        for (const [it, s] of xformSnap)
            it.position.copy(s.pos).add(groupDelta);
        touchAfterXform();
    }
    function applyRotate(angle) {
        if (snapEnabled) {
            const step = Math.PI / 12;
            angle = Math.round(angle / step) * step;
        } // 15°
        rotQ.setFromAxisAngle(WORLD_AXES[activeAxisIdx], angle);
        for (const [it, s] of xformSnap) {
            it.position.copy(s.pos).sub(gizmoPivot).applyQuaternion(rotQ).add(gizmoPivot);
            if (s.rot && !(it instanceof Light)) {
                tmpQ.setFromEuler(s.rot).premultiply(rotQ);
                it.rotation.setFromQuaternion(tmpQ);
            }
            if (s.dir && it instanceof Light)
                it.direction.copy(s.dir).applyQuaternion(rotQ);
        }
        touchAfterXform();
    }
    function applyScale(factor) {
        if (snapEnabled)
            factor = Math.round(factor / 0.25) * 0.25; // 0.25 steps
        const f = Math.max(0.02, factor);
        for (const [it, s] of xformSnap) {
            it.position.copy(s.pos).sub(gizmoPivot).multiplyScalar(f).add(gizmoPivot);
            if (it instanceof MeshInstance)
                it.scale = Math.max(0.02, s.scale * f);
            else if (it instanceof Primitive) {
                it.a = s.a * f;
                it.b = s.b * f;
                it.c = s.c * f;
            }
        }
        touchAfterXform();
    }
    // The gizmo shows for anything with a meaningful position; a lone directional
    // light keeps its dedicated handle re-aim (drag the sun marker) instead.
    function gizmoActive() {
        return ui.getSelection().some((it) => it instanceof Primitive || it instanceof MeshInstance || (it instanceof Light && it.type === LightType.Point));
    }
    // Pivot of the current selection (average of gizmo centres) and a handle radius.
    function selectionPivot(out) {
        const sel = ui.getSelection();
        out.set(0, 0, 0);
        if (!sel.length)
            return out;
        for (const it of sel)
            out.add(gizmoFrame(it).center);
        return out.multiplyScalar(1 / sel.length);
    }
    function selectionRadius(pivot) {
        let r = 6;
        for (const it of ui.getSelection()) {
            const f = gizmoFrame(it);
            r = Math.max(r, f.center.distanceTo(pivot) + f.radius * 0.6);
        }
        return r;
    }
    // ---- box (marquee) selection ----
    const inMarquee = (p, x0, y0, x1, y1) => {
        const s = projectToScreen(p);
        return !!s && s.x >= x0 && s.x <= x1 && s.y >= y0 && s.y <= y1;
    };
    function resolveMarquee(e) {
        const x0 = Math.min(marqStartX, marqCurX), x1 = Math.max(marqStartX, marqCurX);
        const y0 = Math.min(marqStartY, marqCurY), y1 = Math.max(marqStartY, marqCurY);
        const isClick = x1 - x0 < 4 && y1 - y0 < 4;
        const additive = e.shiftKey || e.ctrlKey;
        if (editInstance && editMesh) {
            if (isClick)
                clickSelectVertex(e, additive);
            else
                boxSelectVerts(x0, y0, x1, y1, additive);
        }
        else {
            if (isClick)
                clickSelectObject(e, additive);
            else
                boxSelectObjects(x0, y0, x1, y1, additive);
        }
    }
    function boxSelectObjects(x0, y0, x1, y1, additive) {
        const hits = [];
        for (const p of scene.prims)
            if (inMarquee(p.position, x0, y0, x1, y1))
                hits.push(p);
        for (const m of scene.instances)
            if (inMarquee(m.position, x0, y0, x1, y1))
                hits.push(m);
        for (const l of scene.lights)
            if (inMarquee(lightGizmoPos(l, tmpV), x0, y0, x1, y1))
                hits.push(l);
        if (additive) {
            const cur = ui.getSelection();
            for (const h of hits)
                if (!cur.includes(h))
                    cur.push(h);
            ui.setSelection(cur);
        }
        else {
            ui.setSelection(hits);
        }
    }
    function applyClickSelection(target, additive) {
        if (!target) {
            if (!additive)
                ui.select(null);
            return;
        }
        if (additive) {
            const cur = ui.getSelection();
            const i = cur.indexOf(target);
            if (i >= 0)
                cur.splice(i, 1);
            else
                cur.push(target);
            ui.setSelection(cur);
        }
        else {
            ui.select(target);
        }
    }
    function clickSelectObject(e, additive) {
        const light = pickLight(e.clientX, e.clientY);
        if (light) {
            applyClickSelection(light, additive);
            return;
        }
        const dpr = renderer.width / Math.max(1, canvas.clientWidth);
        renderer.pick(cam, e.clientX * dpr, e.clientY * dpr).then((r) => {
            let target = null;
            if (r.kind === 4 && r.index >= 0 && r.index < scene.prims.length)
                target = scene.prims[r.index];
            else if (r.kind === 5 && r.index >= 0 && r.index < scene.instances.length)
                target = scene.instances[r.index];
            applyClickSelection(target, additive);
        });
    }
    function boxSelectVerts(x0, y0, x1, y1, additive) {
        updateEditMatrices();
        if (!additive)
            selectedVerts.clear();
        const n = editMesh.verts.length / 3;
        for (let i = 0; i < n; i++) {
            if (inMarquee(vertWorld(i, vTmp), x0, y0, x1, y1))
                selectedVerts.add(i);
        }
    }
    function clickSelectVertex(e, additive) {
        updateEditMatrices();
        const vIdx = pickVertex(e.clientX, e.clientY);
        if (vIdx < 0) {
            if (!additive)
                selectedVerts.clear();
            return;
        }
        if (additive) {
            if (selectedVerts.has(vIdx))
                selectedVerts.delete(vIdx);
            else
                selectedVerts.add(vIdx);
        }
        else {
            selectedVerts.clear();
            selectedVerts.add(vIdx);
        }
    }
    for (const c of [canvas, previewCanvas]) {
        c.addEventListener("contextmenu", (e) => e.preventDefault()); // allow right-drag
        c.addEventListener("pointerdown", (e) => {
            cam.update();
            lastX = e.clientX;
            lastY = e.clientY;
            c.setPointerCapture(e.pointerId);
            if (e.button === 1) {
                drag = "look";
                e.preventDefault();
                return;
            } // middle
            if (e.button === 2) {
                drag = "pan";
                return;
            } // right
            if (e.button === 0 && e.altKey) {
                drag = "orbit";
                return;
            } // Alt+left always orbits
            // Click-to-focus (armed from the Render lab): the next left click sets the
            // depth-of-field focal distance to whatever surface is under the cursor.
            if (focusPick && e.button === 0) {
                drag = "none";
                const dpr = renderer.width / Math.max(1, canvas.clientWidth);
                renderer.pick(cam, e.clientX * dpr, e.clientY * dpr).then((r) => {
                    if (r.kind !== 0 && Number.isFinite(r.dist) && r.dist > 0) {
                        scene.world.focusDistance = r.dist;
                        scene.touchWorld();
                        hud.textContent = `Focus set: ${r.dist.toFixed(1)} units`;
                    }
                    else {
                        hud.textContent = "Focus: nothing under the cursor";
                    }
                    focusPick = false;
                    ui.refresh();
                });
                return;
            }
            if (editInstance) {
                editPointerDown(e);
                return;
            } // vertex editing
            if (poseInst) {
                const bi = pickBone(e.clientX, e.clientY);
                if (bi >= 0) {
                    selectedBone = bi;
                    ui.refresh();
                    drag = "bone";
                    return;
                } // click+drag rotates
                drag = "orbit"; // empty space → orbit the camera
                return;
            }
            // Grab a gizmo handle on the current selection? The gizmo sits at the
            // selection pivot and obeys the active mode (translate/rotate/scale).
            if (gizmoActive()) {
                const pivot = selectionPivot(tmpPivot);
                const radius = Math.max(8, selectionRadius(pivot));
                const idx = pickAxisAt(pivot, radius, e.clientX, e.clientY);
                if (idx >= 0) {
                    activeAxisIdx = idx;
                    activeGizmoMode = gizmoMode;
                    dragAxis.copy(WORLD_AXES[idx]);
                    gizmoPivot.copy(pivot);
                    gizmoDownX = e.clientX;
                    gizmoDownY = e.clientY;
                    beginSelectionSnapshot();
                    if (gizmoMode === "translate") {
                        axisStartPos.copy(pivot);
                        const ray = screenRay(e.clientX, e.clientY);
                        axisStartS = closestAxisParam(ray.ro, ray.rd, axisStartPos, dragAxis) ?? 0;
                    }
                    drag = "axis";
                    return;
                }
            }
            // Box-Select tool: left-drag rubber-bands a selection (resolved on
            // pointerup; a click with no drag falls back to a single pick).
            if (tool === "select") {
                marqStartX = marqCurX = e.clientX;
                marqStartY = marqCurY = e.clientY;
                drag = "marquee";
                return;
            }
            // --- Move tool: click-select + drag-move ---
            // Lights aren't in the GPU pick buffer — test their viewport handles first.
            const light = pickLight(e.clientX, e.clientY);
            if (light) {
                ui.select(light);
                startLightDrag(light, e);
                return;
            }
            drag = "pending";
            const dpr = renderer.width / Math.max(1, canvas.clientWidth);
            renderer.pick(cam, e.clientX * dpr, e.clientY * dpr).then((r) => {
                if (drag !== "pending")
                    return;
                let target = null;
                if (r.kind === 4 && r.index >= 0 && r.index < scene.prims.length) {
                    target = scene.prims[r.index];
                }
                else if (r.kind === 5 && r.index >= 0 && r.index < scene.instances.length) {
                    target = scene.instances[r.index];
                }
                if (target) {
                    // Dragging an object that's already part of a multi-selection moves the
                    // whole group; otherwise it becomes the sole selection.
                    const selList = ui.getSelection();
                    if (!(selList.includes(target) && selList.length > 1))
                        ui.select(target);
                    dragTarget = target;
                    dragVertical = e.shiftKey;
                    planePoint.copy(target.position);
                    if (dragVertical)
                        planeNormal.copy(cam.forward).setY(0).normalize();
                    else
                        planeNormal.set(0, 1, 0);
                    const ray = screenRay(lastX, lastY);
                    const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
                    if (hit)
                        objDragHit0.copy(hit);
                    beginGroupSnapshot(target);
                    drag = "object";
                }
                else {
                    ui.select(null); // clicked the world → deselect
                    drag = "orbit";
                }
            });
        });
        c.addEventListener("pointerup", (e) => {
            if (drag === "object" || drag === "axis" || drag === "bone")
                ui.refresh(); // sync inspector fields
            if ((drag === "object" || drag === "axis") && movedDuringDrag)
                history.commit();
            if (drag === "vert" && vertDragMode !== "none")
                commitEdit(); // rebuild BLAS
            if (drag === "marquee")
                resolveMarquee(e);
            drag = "none";
            dragTarget = null;
            draggingDirLight = false;
            activeAxisIdx = -1;
            xformSnap.clear();
            movedDuringDrag = false;
            vertDragMode = "none";
        });
        c.addEventListener("pointermove", (e) => {
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            if (drag === "bone" && poseInst && selectedBone >= 0) {
                // Arcball: rotate the selected bone around the camera's screen axes.
                const b = poseBones[selectedBone];
                const k = 0.012; // radians per pixel
                if (b.parent)
                    b.parent.getWorldQuaternion(qParent);
                else
                    qParent.identity();
                qWorld.copy(qParent).multiply(b.quaternion); // current world orientation
                qDeltaA.setFromAxisAngle(cam.up, dx * k);
                qDeltaB.setFromAxisAngle(cam.right, dy * k);
                qDeltaA.multiply(qDeltaB);
                qWorld.premultiply(qDeltaA);
                b.quaternion.copy(qParent.invert()).multiply(qWorld).normalize(); // back to local
                lastX = e.clientX;
                lastY = e.clientY;
            }
            else if (drag === "orbit") {
                cam.orbit(dx, dy);
                lastX = e.clientX;
                lastY = e.clientY;
                renderer.resetAccumulation();
            }
            else if (drag === "look") {
                cam.look(dx, dy);
                lastX = e.clientX;
                lastY = e.clientY;
                renderer.resetAccumulation();
            }
            else if (drag === "pan") {
                const k = cam.distance * 0.0015;
                cam.moveRelative(-dy * k, dx * k, 0); // truck across the horizontal plane
                lastX = e.clientX;
                lastY = e.clientY;
                renderer.resetAccumulation();
            }
            else if (drag === "marquee") {
                marqCurX = e.clientX;
                marqCurY = e.clientY;
            }
            else if (drag === "axis") {
                if (activeGizmoMode === "rotate") {
                    applyRotate((e.clientX - gizmoDownX) * 0.01); // dx → radians about the axis
                }
                else if (activeGizmoMode === "scale") {
                    applyScale(Math.exp(-(e.clientY - gizmoDownY) * 0.01)); // drag up = larger
                }
                else {
                    const ray = screenRay(e.clientX, e.clientY);
                    const s = closestAxisParam(ray.ro, ray.rd, axisStartPos, dragAxis);
                    if (s !== null) {
                        groupDelta.copy(dragAxis).multiplyScalar(s - axisStartS);
                        applyTranslate();
                    }
                }
            }
            else if (drag === "vert") {
                const ray = screenRay(e.clientX, e.clientY);
                if (vertDragMode === "axis") {
                    const s = closestAxisParam(ray.ro, ray.rd, axisStartPos, dragAxis);
                    if (s !== null)
                        applyWorldDelta(vTmp.copy(dragAxis).multiplyScalar(s - axisStartS));
                }
                else if (vertDragMode === "plane") {
                    const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
                    if (hit)
                        applyWorldDelta(hit.sub(editPlaneStart));
                }
            }
            else if (draggingDirLight && dragTarget instanceof Light) {
                const ray = screenRay(e.clientX, e.clientY);
                const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
                if (hit) {
                    const d = hit.sub(cam.target);
                    d.y = Math.max(d.y, 2); // keep the sun above the horizon
                    dragTarget.direction.copy(d).normalize();
                    scene.touch();
                }
            }
            else if (drag === "object" && dragTarget) {
                const ray = screenRay(e.clientX, e.clientY);
                const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
                if (hit) {
                    groupDelta.copy(hit).sub(objDragHit0);
                    if (dragVertical) {
                        groupDelta.x = 0;
                        groupDelta.z = 0;
                    } // Shift = vertical only
                    else
                        groupDelta.y = 0;
                    applyTranslate();
                }
            }
        });
        c.addEventListener("wheel", (e) => {
            e.preventDefault();
            cam.dolly(e.deltaY);
            renderer.resetAccumulation();
        }, { passive: false });
    }
    // ---- selection ergonomics (Delete / Esc / Ctrl+A / F) ----
    function deleteSelection() {
        const sel = ui.getSelection();
        if (!sel.length)
            return;
        for (const it of sel)
            scene.remove(it);
        const fb = scene.prims[0] ?? scene.instances[0] ?? scene.lights[0] ?? null;
        ui.setSelection(fb ? [fb] : []); // refreshes the UI
        history.commit();
    }
    function duplicateSelection() {
        if (editInstance || poseInst)
            return;
        const sel = ui.getSelection();
        if (!sel.length)
            return;
        const copies = [];
        for (const it of sel) {
            const dup = scene.addDuplicate(it);
            dup.position.x += 4;
            dup.position.z += 4; // offset so the copy is visible
            if (dup instanceof MeshInstance)
                preview.addInstance(dup.id, previewFromBlas(scene, dup.blasIndex), []);
            copies.push(dup);
        }
        scene.touchInstances();
        ui.setSelection(copies);
        history.commit();
        ui.refresh();
    }
    // Align selected objects' centres along one axis (0=X,1=Y,2=Z).
    function alignSelection(axis, mode) {
        const sel = ui.getSelection();
        if (sel.length < 2)
            return;
        const vals = sel.map((it) => it.position.getComponent(axis));
        const target = mode === "min" ? Math.min(...vals)
            : mode === "max" ? Math.max(...vals)
                : vals.reduce((a, b) => a + b, 0) / vals.length;
        for (const it of sel)
            it.position.setComponent(axis, target);
        afterArrange(sel);
    }
    // Space selected objects evenly between the two extremes along one axis.
    function distributeSelection(axis) {
        const sel = ui.getSelection();
        if (sel.length < 3)
            return;
        const sorted = [...sel].sort((a, b) => a.position.getComponent(axis) - b.position.getComponent(axis));
        const lo = sorted[0].position.getComponent(axis);
        const hi = sorted[sorted.length - 1].position.getComponent(axis);
        const step = (hi - lo) / (sorted.length - 1);
        sorted.forEach((it, i) => it.position.setComponent(axis, lo + i * step));
        afterArrange(sel);
    }
    // Randomly jitter selected objects across the ground plane (uniform disk),
    // with a little random spin so non-spherical shapes look naturally scattered.
    function scatterSelection(radius) {
        const sel = ui.getSelection();
        if (!sel.length)
            return;
        for (const it of sel) {
            const ang = Math.random() * Math.PI * 2;
            const r = Math.sqrt(Math.random()) * radius; // sqrt → even areal spread
            it.position.x += Math.cos(ang) * r;
            it.position.z += Math.sin(ang) * r;
            if (!(it instanceof Light))
                it.rotation.y += (Math.random() - 0.5) * Math.PI;
        }
        afterArrange(sel);
    }
    // Mirror the selection across the plane (perpendicular to `axis`) through the
    // selection pivot: reflect positions, conjugate rotations (S·R·S keeps a proper
    // rotation), and flip a directional light's aim. Mesh handedness is unchanged
    // (no negative scale), so asymmetric meshes are repositioned/re-oriented only.
    function mirrorSelection(axis) {
        const sel = ui.getSelection();
        if (!sel.length)
            return;
        const pivot = selectionPivot(new Vector3());
        const s = new Matrix4().makeScale(axis === 0 ? -1 : 1, axis === 1 ? -1 : 1, axis === 2 ? -1 : 1);
        const m = new Matrix4();
        for (const it of sel) {
            it.position.setComponent(axis, 2 * pivot.getComponent(axis) - it.position.getComponent(axis));
            if (it instanceof Light) {
                if (it.type === LightType.Directional)
                    it.direction.setComponent(axis, -it.direction.getComponent(axis));
            }
            else {
                m.makeRotationFromEuler(it.rotation).premultiply(s).multiply(s);
                it.rotation.setFromRotationMatrix(m);
            }
        }
        afterArrange(sel);
    }
    function afterArrange(sel) {
        if (sel.some((it) => it instanceof MeshInstance))
            scene.touchInstances();
        else
            scene.touch();
        history.commit();
        ui.refresh();
        renderer.resetAccumulation();
    }
    function saveBookmark(i) {
        camBookmarks[i] = { target: [cam.target.x, cam.target.y, cam.target.z], yaw: cam.yaw, pitch: cam.pitch, distance: cam.distance };
        ui.refresh();
    }
    function recallBookmark(i) {
        const b = camBookmarks[i];
        if (!b)
            return;
        cam.target.set(b.target[0], b.target[1], b.target[2]);
        cam.yaw = b.yaw;
        cam.pitch = b.pitch;
        cam.distance = b.distance;
        cam.update();
        renderer.resetAccumulation();
    }
    function selectAll() {
        if (editInstance && editMesh) {
            const n = editMesh.verts.length / 3;
            for (let i = 0; i < n; i++)
                selectedVerts.add(i); // edit mode: all vertices
            return;
        }
        ui.setSelection(scene.selectables);
    }
    function escAction() {
        if (editInstance) {
            toggleEdit(editInstance);
            ui.refresh();
            return;
        } // exit vertex edit
        if (poseInst) {
            togglePose(poseInst);
            ui.refresh();
            return;
        } // exit pose
        ui.select(null); // deselect and drop back to the Move tool
        tool = "move";
        ui.refresh();
    }
    function frameSelection() {
        const sel = ui.getSelection();
        if (!sel.length)
            return;
        const frames = sel.map((it) => gizmoFrame(it));
        const c = new Vector3();
        for (const f of frames)
            c.add(f.center);
        c.multiplyScalar(1 / frames.length);
        let r = 6;
        for (const f of frames)
            r = Math.max(r, f.center.distanceTo(c) + f.radius);
        cam.target.copy(c);
        cam.distance = Math.min(400, Math.max(12, r * 2.4));
        renderer.resetAccumulation();
    }
    // ---- keyboard: arrows fly, Ctrl/Shift + arrows look (rotate view in place) ----
    const ARROWS = new Set(["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"]);
    const pressed = new Set();
    let rotateMod = false;
    const typingInField = () => {
        const a = document.activeElement;
        return a instanceof HTMLInputElement || a instanceof HTMLSelectElement;
    };
    window.addEventListener("keydown", (e) => {
        if (typingInField())
            return;
        rotateMod = e.ctrlKey || e.shiftKey;
        if ((e.ctrlKey || e.metaKey) && (e.key === "z" || e.key === "Z")) {
            e.preventDefault();
            if (editInstance || poseInst)
                return; // don't rewrite the scene mid-edit
            if (e.shiftKey)
                history.redo();
            else
                history.undo();
            return;
        }
        if ((e.ctrlKey || e.metaKey) && (e.key === "y" || e.key === "Y")) {
            e.preventDefault();
            if (editInstance || poseInst)
                return;
            history.redo();
            return;
        }
        if (e.key === "1" && tool !== "move") {
            tool = "move";
            ui.refresh();
            return;
        }
        if (e.key === "2" && tool !== "select") {
            tool = "select";
            ui.refresh();
            return;
        }
        if ((e.ctrlKey || e.metaKey) && (e.key === "a" || e.key === "A")) {
            e.preventDefault();
            selectAll();
            return;
        }
        if ((e.ctrlKey || e.metaKey) && (e.key === "d" || e.key === "D")) {
            e.preventDefault();
            duplicateSelection();
            return;
        }
        if (e.key === "Delete" || e.key === "Backspace") {
            if (editInstance || poseInst)
                return; // leave vertex/bone handling alone
            e.preventDefault();
            deleteSelection();
            return;
        }
        if (e.key === "Escape") {
            escAction();
            return;
        }
        if ((e.key === "f" || e.key === "F") && !e.ctrlKey && !e.metaKey) {
            frameSelection();
            return;
        }
        if (!e.ctrlKey && !e.metaKey) {
            if (e.key === "w" || e.key === "W") {
                gizmoMode = "translate";
                ui.refresh();
                return;
            }
            if (e.key === "e" || e.key === "E") {
                gizmoMode = "rotate";
                ui.refresh();
                return;
            }
            if (e.key === "r" || e.key === "R") {
                gizmoMode = "scale";
                ui.refresh();
                return;
            }
        }
        if (ARROWS.has(e.key)) {
            e.preventDefault();
            pressed.add(e.key);
        }
    });
    window.addEventListener("keyup", (e) => {
        rotateMod = e.ctrlKey || e.shiftKey;
        pressed.delete(e.key);
    });
    window.addEventListener("blur", () => pressed.clear());
    function applyCameraKeys(dt) {
        if (pressed.size === 0)
            return false;
        const move = 36 * dt; // world units / sec
        const rot = 1.6 * dt; // radians / sec
        if (rotateMod) {
            // Look around in place — camera position stays fixed, only the view turns.
            if (pressed.has("ArrowLeft"))
                cam.lookAngles(rot, 0);
            if (pressed.has("ArrowRight"))
                cam.lookAngles(-rot, 0);
            if (pressed.has("ArrowUp"))
                cam.lookAngles(0, -rot);
            if (pressed.has("ArrowDown"))
                cam.lookAngles(0, rot);
        }
        else {
            if (pressed.has("ArrowUp"))
                cam.moveRelative(move, 0, 0);
            if (pressed.has("ArrowDown"))
                cam.moveRelative(-move, 0, 0);
            if (pressed.has("ArrowLeft"))
                cam.moveRelative(0, -move, 0);
            if (pressed.has("ArrowRight"))
                cam.moveRelative(0, move, 0);
        }
        return true;
    }
    // ---- selection gizmo overlay (2D, drawn over both views) ----
    const tmpV = new Vector3();
    const axPt = new Vector3();
    const lightTmp = new Vector3();
    const AXIS_COLORS = ["#ff5566", "#55dd66", "#5599ff"];
    const AXIS_LABELS = ["X", "Y", "Z"];
    function projectToScreen(p) {
        const v = tmpV.copy(p).sub(cam.position);
        const z = v.dot(cam.forward);
        if (z <= 0.05)
            return null; // behind camera
        const aspect = overlay.width / Math.max(1, overlay.height);
        const ndcx = v.dot(cam.right) / (z * aspect * cam.fovScale);
        const ndcy = v.dot(cam.up) / (z * cam.fovScale);
        return {
            x: (ndcx * 0.5 + 0.5) * overlay.width,
            y: (1 - (ndcy * 0.5 + 0.5)) * overlay.height,
        };
    }
    // Directional lights have no position; show their handle this far from the
    // current view target, along the light direction.
    const DIR_HANDLE_DIST = 45;
    function lightGizmoPos(l, out = new Vector3()) {
        if (l.type === LightType.Directional) {
            return out.copy(cam.target).addScaledVector(l.direction, DIR_HANDLE_DIST);
        }
        return out.copy(l.position);
    }
    function gizmoFrame(sel) {
        if (sel instanceof Light)
            return { center: lightGizmoPos(sel), radius: 7 };
        const radius = sel instanceof Primitive ? Math.max(sel.a, sel.b, sel.c) + 1.5 : 12 * sel.scale;
        return { center: new Vector3().copy(sel.position), radius };
    }
    /** Nearest light whose viewport handle is under the cursor, or null. */
    function pickLight(cx, cy) {
        let best = null;
        let bestD = 14; // px hit radius
        for (const l of scene.lights) {
            const s = projectToScreen(lightGizmoPos(l, tmpV));
            if (!s)
                continue;
            const d = Math.hypot(cx - s.x, cy - s.y);
            if (d < bestD) {
                bestD = d;
                best = l;
            }
        }
        return best;
    }
    /** Begin dragging a light: point lights move by position (Shift = up/down),
     *  directional lights re-aim by their viewport handle. */
    function startLightDrag(l, e) {
        dragTarget = l;
        if (l.type === LightType.Directional) {
            draggingDirLight = true;
            planePoint.copy(lightGizmoPos(l));
            planeNormal.copy(cam.forward).multiplyScalar(-1).normalize(); // screen-facing plane
            drag = "object";
            return;
        }
        draggingDirLight = false;
        dragVertical = e.shiftKey;
        planePoint.copy(l.position);
        if (dragVertical)
            planeNormal.copy(cam.forward).setY(0).normalize();
        else
            planeNormal.set(0, 1, 0);
        const ray = screenRay(e.clientX, e.clientY);
        const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
        if (hit)
            objDragHit0.copy(hit);
        beginGroupSnapshot(l);
        drag = "object";
    }
    // Parameter along an axis line (base + s·axis) closest to the camera ray.
    function closestAxisParam(ro, rd, base, axis) {
        const w0 = tmpV.copy(ro).sub(base);
        const b = rd.dot(axis); // rd and axis are unit length
        const denom = 1 - b * b;
        if (Math.abs(denom) < 1e-5)
            return null; // ray ∥ axis
        return (axis.dot(w0) - b * rd.dot(w0)) / denom;
    }
    function pointSegDist(px, py, ax, ay, bx, by) {
        const dx = bx - ax, dy = by - ay;
        const len2 = dx * dx + dy * dy;
        const t = len2 > 0 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
        return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    }
    /** Which axis handle (0/1/2) of a gizmo at `center` is under the cursor, or -1. */
    function pickAxisAt(center, radius, cx, cy) {
        const c = projectToScreen(center);
        if (!c)
            return -1;
        let best = -1;
        let bestDist = 11; // px hit threshold
        for (let i = 0; i < 3; i++) {
            const tip = projectToScreen(axPt.copy(center).addScaledVector(WORLD_AXES[i], radius));
            if (!tip)
                continue;
            const d = pointSegDist(cx, cy, c.x, c.y, tip.x, tip.y);
            if (d < bestDist) {
                bestDist = d;
                best = i;
            }
        }
        return best;
    }
    // Draw the X/Y/Z axis handles of a gizmo centered at a world point.
    function drawAxes(center, radius) {
        const c = projectToScreen(center);
        if (!c)
            return;
        octx.font = "bold 12px ui-monospace, monospace";
        for (let i = 0; i < 3; i++) {
            const e = projectToScreen(axPt.copy(center).addScaledVector(WORLD_AXES[i], radius));
            if (!e)
                continue;
            const active = i === activeAxisIdx;
            octx.strokeStyle = AXIS_COLORS[i];
            octx.lineWidth = active ? 4 : 2.5;
            octx.beginPath();
            octx.moveTo(c.x, c.y);
            octx.lineTo(e.x, e.y);
            octx.stroke();
            octx.fillStyle = AXIS_COLORS[i];
            octx.beginPath();
            octx.arc(e.x, e.y, active ? 5.5 : 4, 0, Math.PI * 2);
            octx.fill();
            octx.fillText(AXIS_LABELS[i], e.x + 5, e.y + 4);
        }
        octx.fillStyle = "rgba(255,255,255,0.9)";
        octx.beginPath();
        octx.arc(c.x, c.y, 3, 0, Math.PI * 2);
        octx.fill();
    }
    // Draw clickable handles for every light; the selected one gets extra chrome.
    function drawLightMarkers() {
        const sel = ui.getSelected();
        const selSet = ui.getSelection();
        for (const l of scene.lights) {
            const s = projectToScreen(lightGizmoPos(l, lightTmp));
            if (!s)
                continue;
            const on = selSet.includes(l);
            octx.save();
            if (l.type === LightType.Directional) {
                if (on) {
                    const tc = projectToScreen(cam.target);
                    if (tc) {
                        octx.strokeStyle = "rgba(255,225,150,0.5)";
                        octx.lineWidth = 1.5;
                        octx.setLineDash([5, 4]);
                        octx.beginPath();
                        octx.moveTo(tc.x, tc.y);
                        octx.lineTo(s.x, s.y);
                        octx.stroke();
                        octx.setLineDash([]);
                    }
                }
                const r = on ? 6 : 5;
                octx.strokeStyle = octx.fillStyle = on ? "#ffe680" : "rgba(255,225,150,0.85)";
                octx.lineWidth = on ? 2 : 1.5;
                octx.beginPath();
                octx.arc(s.x, s.y, r, 0, Math.PI * 2);
                octx.fill();
                for (let i = 0; i < 8; i++) {
                    const a = (i / 8) * Math.PI * 2;
                    octx.beginPath();
                    octx.moveTo(s.x + Math.cos(a) * (r + 2), s.y + Math.sin(a) * (r + 2));
                    octx.lineTo(s.x + Math.cos(a) * (r + 6), s.y + Math.sin(a) * (r + 6));
                    octx.stroke();
                }
            }
            else {
                const r = on ? 6 : 5;
                octx.strokeStyle = octx.fillStyle = on ? "#bfe0ff" : "rgba(150,200,255,0.85)";
                octx.lineWidth = on ? 2 : 1.5;
                octx.beginPath();
                octx.arc(s.x, s.y, r, 0, Math.PI * 2);
                octx.stroke();
                octx.beginPath();
                octx.arc(s.x, s.y, r - 2.5, 0, Math.PI * 2);
                octx.fill();
            }
            octx.restore();
        }
        if (sel instanceof Light && sel.type === LightType.Point) {
            const { center, radius } = gizmoFrame(sel);
            drawAxes(center, radius);
        }
    }
    // Oriented half-extents of a primitive's bounding cage (matches the SDF dims).
    const ghostQuat = new Quaternion();
    const ghostCorner = new Vector3();
    function primHalfExtents(p, out) {
        switch (p.type) {
            case PrimType.Box:
            case PrimType.RoundedBox: return out.set(p.a, p.b, p.c);
            case PrimType.Torus: return out.set(p.a + p.b, p.b, p.a + p.b);
            case PrimType.Cylinder:
            case PrimType.Cone:
            case PrimType.Pyramid: return out.set(p.a, p.b, p.a);
            case PrimType.Capsule: return out.set(p.a, p.b + p.a, p.a);
            default: return out.set(p.a, p.a, p.a); // sphere, octahedron
        }
    }
    // Wireframe cages over every "carve" primitive, so invisible cutters are visible.
    function drawCarverGhosts() {
        if (!showCarverGhosts)
            return;
        octx.save();
        octx.setLineDash([4, 3]);
        octx.lineWidth = 1.25;
        for (const p of scene.prims) {
            if (!p.subtractive)
                continue;
            ghostQuat.setFromEuler(p.rotation);
            const h = primHalfExtents(p, scratch);
            const pts = [];
            for (let i = 0; i < 8; i++) {
                ghostCorner.set((i & 1 ? h.x : -h.x), (i & 2 ? h.y : -h.y), (i & 4 ? h.z : -h.z)).applyQuaternion(ghostQuat).add(p.position);
                pts.push(projectToScreen(ghostCorner));
            }
            octx.strokeStyle = ui.getSelection().includes(p) ? "#ff9a5a" : "rgba(255,140,90,0.7)";
            for (let i = 0; i < 8; i++) {
                for (const bit of [1, 2, 4]) {
                    if (i & bit)
                        continue;
                    const a = pts[i], b = pts[i | bit];
                    if (!a || !b)
                        continue;
                    octx.beginPath();
                    octx.moveTo(a.x, a.y);
                    octx.lineTo(b.x, b.y);
                    octx.stroke();
                }
            }
        }
        octx.restore();
    }
    // Skeleton overlay: bone lines + clickable joints.
    const boneWorld = new Vector3();
    function pickBone(cx, cy) {
        let best = -1;
        let bestD = 13;
        for (let i = 0; i < poseBones.length; i++) {
            const s = projectToScreen(poseBones[i].getWorldPosition(boneWorld));
            if (!s)
                continue;
            const d = Math.hypot(cx - s.x, cy - s.y);
            if (d < bestD) {
                bestD = d;
                best = i;
            }
        }
        return best;
    }
    function drawPose() {
        if (!poseInst || poseBones.length === 0)
            return;
        const boneSet = new Set(poseBones);
        octx.strokeStyle = "rgba(120,220,160,0.55)";
        octx.lineWidth = 1.5;
        for (const b of poseBones) {
            if (!b.parent || !boneSet.has(b.parent))
                continue;
            const a = projectToScreen(b.getWorldPosition(boneWorld));
            const c = projectToScreen(b.parent.getWorldPosition(boneWorld));
            if (!a || !c)
                continue;
            octx.beginPath();
            octx.moveTo(a.x, a.y);
            octx.lineTo(c.x, c.y);
            octx.stroke();
        }
        for (let i = 0; i < poseBones.length; i++) {
            const s = projectToScreen(poseBones[i].getWorldPosition(boneWorld));
            if (!s)
                continue;
            const on = i === selectedBone;
            octx.fillStyle = on ? "#ffe680" : "rgba(150,235,180,0.95)";
            octx.beginPath();
            octx.arc(s.x, s.y, on ? 5.5 : 3.5, 0, Math.PI * 2);
            octx.fill();
        }
        octx.fillStyle = "rgba(255,255,255,0.85)";
        octx.font = "12px ui-monospace, monospace";
        octx.fillText("POSE MODE · click+drag a joint to rotate it", 12, overlay.height - 16);
    }
    function drawGizmo() {
        octx.clearRect(0, 0, overlay.width, overlay.height);
        if (editInstance && editMesh) {
            drawEditOverlay();
            drawMarquee();
            return;
        }
        if (poseInst) {
            drawPose();
            return;
        }
        drawCarverGhosts();
        drawLightMarkers();
        drawSelectionBoxes(); // dashed outline on every selected primitive/instance
        // One manipulator at the selection pivot, in the active transform mode.
        if (gizmoActive()) {
            const pivot = selectionPivot(tmpPivot);
            const radius = Math.max(8, selectionRadius(pivot));
            const c = projectToScreen(pivot);
            if (c) {
                if (gizmoMode === "rotate") {
                    const edge = projectToScreen(tmpV.copy(pivot).addScaledVector(cam.right, radius));
                    const rpx = edge ? Math.max(12, Math.hypot(edge.x - c.x, edge.y - c.y)) : 24;
                    octx.save();
                    octx.strokeStyle = "rgba(255,255,255,0.5)";
                    octx.lineWidth = 1.25;
                    octx.beginPath();
                    octx.arc(c.x, c.y, rpx, 0, Math.PI * 2);
                    octx.stroke();
                    octx.restore();
                }
                drawAxes(pivot, radius);
                octx.save();
                octx.fillStyle = "rgba(255,255,255,0.85)";
                octx.font = "11px ui-monospace, monospace";
                octx.fillText(gizmoMode, c.x + 9, c.y - 9);
                octx.restore();
            }
        }
        drawMarquee();
    }
    // Dashed outline around every selected primitive/instance (group context).
    function drawSelectionBoxes() {
        const sel = ui.getSelection();
        octx.save();
        octx.strokeStyle = "rgba(255,255,255,0.8)";
        octx.lineWidth = 1.25;
        octx.setLineDash([5, 4]);
        for (const it of sel) {
            if (!(it instanceof Primitive || it instanceof MeshInstance))
                continue;
            const { center, radius } = gizmoFrame(it);
            const c = projectToScreen(center);
            if (!c)
                continue;
            const edge = projectToScreen(tmpV.copy(center).addScaledVector(cam.right, radius));
            const rpx = edge ? Math.max(8, Math.hypot(edge.x - c.x, edge.y - c.y)) : 12;
            octx.strokeRect(c.x - rpx, c.y - rpx, rpx * 2, rpx * 2);
        }
        octx.setLineDash([]);
        octx.restore();
    }
    // The in-progress box-select rectangle.
    function drawMarquee() {
        if (drag !== "marquee")
            return;
        const x = Math.min(marqStartX, marqCurX), y = Math.min(marqStartY, marqCurY);
        const w = Math.abs(marqCurX - marqStartX), h = Math.abs(marqCurY - marqStartY);
        octx.save();
        octx.fillStyle = "rgba(120,200,255,0.12)";
        octx.strokeStyle = "rgba(120,200,255,0.9)";
        octx.lineWidth = 1;
        octx.fillRect(x, y, w, h);
        octx.strokeRect(x, y, w, h);
        octx.restore();
    }
    // ---- mesh vertex editing ----
    const selectedVerts = new Set();
    const editModel = new Matrix4();
    const editInvRot = new Matrix3();
    const editTmpMat = new Matrix4();
    const editPlaneStart = new Vector3();
    const vertDragStart = new Map();
    let vertDragMode = "none";
    const VERT_GIZMO_RADIUS = 6;
    const vTmp = new Vector3();
    const vTmp2 = new Vector3();
    function updateEditMatrices() {
        if (!editInstance)
            return;
        editInstance.modelMatrix(editModel);
        editInvRot.setFromMatrix4(editTmpMat.copy(editModel).invert());
    }
    function vertWorld(i, out) {
        const v = editMesh.verts;
        return out.set(v[3 * i], v[3 * i + 1], v[3 * i + 2]).applyMatrix4(editModel);
    }
    function vertCentroidWorld(out) {
        out.set(0, 0, 0);
        const v = editMesh.verts;
        for (const i of selectedVerts)
            out.set(out.x + v[3 * i], out.y + v[3 * i + 1], out.z + v[3 * i + 2]);
        if (selectedVerts.size > 0)
            out.multiplyScalar(1 / selectedVerts.size);
        return out.applyMatrix4(editModel);
    }
    function pickVertex(cx, cy) {
        const n = editMesh.verts.length / 3;
        let best = -1;
        let bestD = 10;
        for (let i = 0; i < n; i++) {
            const s = projectToScreen(vertWorld(i, vTmp));
            if (!s)
                continue;
            const d = Math.hypot(cx - s.x, cy - s.y);
            if (d < bestD) {
                bestD = d;
                best = i;
            }
        }
        return best;
    }
    function snapshotVerts() {
        vertDragStart.clear();
        const v = editMesh.verts;
        for (const i of selectedVerts)
            vertDragStart.set(i, [v[3 * i], v[3 * i + 1], v[3 * i + 2]]);
    }
    function applyWorldDelta(worldDelta) {
        const local = vTmp2.copy(worldDelta).applyMatrix3(editInvRot);
        const v = editMesh.verts;
        for (const [i, s] of vertDragStart) {
            v[3 * i] = s[0] + local.x;
            v[3 * i + 1] = s[1] + local.y;
            v[3 * i + 2] = s[2] + local.z;
        }
    }
    function commitEdit() {
        if (editInstance && editMesh) {
            scene.replaceBlas(editInstance.blasIndex, rebuildBlas(editMesh));
            history.commit(); // a vertex edit is one undo step (undo after exiting edit mode)
        }
    }
    function toggleEdit(inst) {
        if (editInstance === inst) {
            editInstance = null;
            editMesh = null;
            selectedVerts.clear();
            return;
        }
        editInstance = inst;
        editMesh = buildEditMesh(scene.blases[inst.blasIndex]);
        selectedVerts.clear();
        mode = "render";
        applyMode(); // edits are only visible in the ray-traced view
    }
    // Left-button pointerdown while editing: pick/move vertices.
    function editPointerDown(e) {
        updateEditMatrices();
        if (selectedVerts.size > 0) {
            const idx = pickAxisAt(vertCentroidWorld(vTmp), VERT_GIZMO_RADIUS, e.clientX, e.clientY);
            if (idx >= 0) {
                activeAxisIdx = idx;
                dragAxis.copy(WORLD_AXES[idx]);
                axisStartPos.copy(vertCentroidWorld(vTmp));
                const ray = screenRay(e.clientX, e.clientY);
                axisStartS = closestAxisParam(ray.ro, ray.rd, axisStartPos, dragAxis) ?? 0;
                snapshotVerts();
                vertDragMode = "axis";
                drag = "vert";
                return;
            }
        }
        // Box-Select tool: left-drag rubber-bands vertices (single-picks on a click).
        if (tool === "select") {
            marqStartX = marqCurX = e.clientX;
            marqStartY = marqCurY = e.clientY;
            drag = "marquee";
            return;
        }
        const vIdx = pickVertex(e.clientX, e.clientY);
        if (vIdx < 0) {
            selectedVerts.clear();
            drag = "orbit";
            return;
        }
        if (e.shiftKey) {
            if (selectedVerts.has(vIdx))
                selectedVerts.delete(vIdx);
            else
                selectedVerts.add(vIdx);
            drag = "none";
            return;
        }
        if (!selectedVerts.has(vIdx)) {
            selectedVerts.clear();
            selectedVerts.add(vIdx);
        }
        planeNormal.set(0, 1, 0);
        planePoint.copy(vertCentroidWorld(vTmp));
        const ray = screenRay(e.clientX, e.clientY);
        const hit = rayPlane(ray.ro, ray.rd, planePoint, planeNormal);
        if (hit)
            editPlaneStart.copy(hit);
        snapshotVerts();
        vertDragMode = "plane";
        drag = "vert";
    }
    function drawEditOverlay() {
        const em = editMesh;
        updateEditMatrices();
        const n = em.verts.length / 3;
        const step = Math.max(1, Math.ceil(n / 4000)); // decimate display on big meshes
        octx.fillStyle = "rgba(120,200,255,0.5)";
        for (let i = 0; i < n; i += step) {
            if (selectedVerts.has(i))
                continue;
            const s = projectToScreen(vertWorld(i, vTmp));
            if (s)
                octx.fillRect(s.x - 1.5, s.y - 1.5, 3, 3);
        }
        octx.fillStyle = "#ffcc33";
        for (const i of selectedVerts) {
            const s = projectToScreen(vertWorld(i, vTmp));
            if (s)
                octx.fillRect(s.x - 2.5, s.y - 2.5, 5, 5);
        }
        if (selectedVerts.size > 0)
            drawAxes(vertCentroidWorld(vTmp), VERT_GIZMO_RADIUS);
        octx.fillStyle = "rgba(255,255,255,0.85)";
        octx.font = "12px ui-monospace, monospace";
        octx.fillText(`EDIT MODE · ${selectedVerts.size} selected / ${n} verts · ` +
            (tool === "select" ? "drag to box-select (1=Move)" : "click+drag points, Shift=multi (2=Box-Select)"), 12, overlay.height - 16);
    }
    // ---- loop ----
    let fps = 0;
    let lastTime = performance.now();
    function frame() {
        const now = performance.now();
        const dt = Math.min(0.05, (now - lastTime) / 1000);
        // An off-screen export (turntable / cutscene) drives the renderer at its own
        // size; keep the rAF loop alive but idle so it doesn't fight the GPU/uniforms.
        if (offlineRendering) {
            lastTime = now;
            requestAnimationFrame(frame);
            return;
        }
        // Cutscene playback: advance the clock and pose the camera + DoF from the path.
        if (cutscenePlaying) {
            cutsceneTime += dt;
            const total = cutsceneDuration(cutsceneKeys);
            applyCutsceneAt(cutsceneTime);
            if (cutsceneTime >= total) {
                cutsceneTime = total;
                cutscenePlaying = false;
                ui.refresh(); // playback ended → restore Play button state
            }
            else if (now - lastCutsceneSync > 110) {
                lastCutsceneSync = now;
                ui.refreshCutscene(); // move the scrubber/time without rebuilding everything
            }
        }
        resize();
        if (scene.version !== uploadedVersion) {
            renderer.uploadScene(scene);
            uploadedVersion = scene.version;
        }
        if (scene.meshStructVersion !== uploadedMeshStruct) {
            renderer.uploadMeshPools(scene); // also re-uploads instances + materials
            uploadedMeshStruct = scene.meshStructVersion;
            uploadedInst = scene.instanceVersion;
            uploadedMeshMat = scene.meshMatVersion;
        }
        else if (scene.instanceVersion !== uploadedInst) {
            renderer.uploadInstances(scene);
            uploadedInst = scene.instanceVersion;
        }
        if (scene.meshMatVersion !== uploadedMeshMat) {
            renderer.uploadMeshMaterials(scene);
            uploadedMeshMat = scene.meshMatVersion;
        }
        if (scene.worldVersion !== uploadedWorld) {
            renderer.uploadWorld(scene);
            uploadedWorld = scene.worldVersion;
        }
        if (scene.primTexVersion !== uploadedPrimTex) {
            renderer.uploadPrimTextures(scene);
            uploadedPrimTex = scene.primTexVersion;
        }
        if (applyCameraKeys(dt))
            renderer.resetAccumulation();
        cam.update();
        if (importing) {
            requestAnimationFrame(frame);
            return;
        }
        if (mode === "preview") {
            preview.syncInstances(scene);
            preview.render(cam);
            drawGizmo();
            hud.textContent =
                `Aerie · Preview (raster)\n` +
                    `${tool === "select" ? "Box-Select" : "Move/Camera"} tool` +
                    `${scene.instances.length ? ` · ${scene.instances.length} model(s)` : ""}`;
            lastTime = now;
            requestAnimationFrame(frame);
            return;
        }
        if (renderer.frame < MAX_SAMPLES)
            renderer.renderSample(cam);
        drawGizmo();
        fps = fps * 0.9 + (1000 / Math.max(1, now - lastTime)) * 0.1;
        lastTime = now;
        const done = renderer.frame >= MAX_SAMPLES;
        hud.textContent =
            `Aerie · Render (ray-traced)\n` +
                `${renderer.width}×${renderer.height} · ${Math.min(renderer.frame, MAX_SAMPLES)}/${MAX_SAMPLES}` +
                `${done ? " ✓" : ""} · ${fps.toFixed(0)} fps` +
                `${scene.instances.length ? ` · ${scene.instances.length} model(s)` : ""}\n` +
                `${tool === "select" ? "Box-Select" : "Move/Camera"} · gizmo ${gizmoMode}` +
                `${snapEnabled ? ` · snap ${snapGrid}` : ""}`;
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}
main().catch((e) => fail(String(e?.stack ?? e)));
