import { Primitive, MeshInstance, PrimType, LightType, PrimPattern, CloudType, PRIM_LABELS, PATTERN_LABELS, CLOUD_LABELS, MAX_CLOUD_LAYERS, MAX_BOOL_GROUPS, } from "./scene/scene";
import { acceptAttribute } from "./mesh/modelImport";
import { LANDSCAPE_LABELS, LANDSCAPE_TYPES } from "./mesh/landscape";
/**
 * Minimal scene editor: import models, switch preview/render, add
 * primitives/lights, and edit the selected item with live sliders. Every
 * mutation calls scene.touch(); the render loop watches scene.version.
 */
export function buildUI(scene, getSpawn, hooks) {
    const menubar = document.getElementById("menubar");
    const panel = document.getElementById("panel"); // inspector (right)
    const scenePanel = document.getElementById("scenePanel"); // object list (left-top)
    const labsPanel = document.getElementById("labsPanel"); // procedural labs (left-bottom)
    const libraryDock = document.getElementById("libraryDock"); // material library (bottom)
    const cutsceneDock = document.getElementById("cutsceneDock"); // cutscene timeline (bottom)
    // Selection set; the primary (inspector target) is the last entry.
    let selection = scene.prims[0] ? [scene.prims[0]] : [];
    let scatterAmount = 12; // world-units radius for the Scatter action
    // Ecosystem-scatter settings (one-shot action params; not part of the scene).
    const eco = {
        shape: PrimType.Cone, count: 80, altMin: 0, altMax: 0.85, maxSlope: 0.55,
        scaleMin: 0.4, scaleMax: 1.2, align: true, color: [0.22, 0.42, 0.18],
    };
    const ECO_SHAPES = [
        PrimType.Cone, PrimType.Sphere, PrimType.Box, PrimType.Cylinder,
        PrimType.Capsule, PrimType.Octahedron, PrimType.Pyramid,
    ].map((t) => [t, PRIM_LABELS[t]]);
    // Turntable-render settings (one-shot export params).
    const turntable = { seconds: 8, fps: 30, samples: 96, width: 1280 };
    let addCloudType = CloudType.Cumulus; // pending "add cloud layer" choice
    const primary = () => selection[selection.length - 1] ?? null;
    let activeLab = null;
    let openMenu = null; // which top menu-bar dropdown is open
    let tab = "transform"; // active inspector tab (shared across object types)
    let matIndex = 0; // which model material the Material tab edits
    // Collapsible Sky-lab groups; open by name.
    const openGroups = new Set(["Presets", "Arrange", "Depth of field", "Lighting"]);
    // Keyboard reference — lives behind the menu-bar "?" instead of the HUD.
    const SHORTCUTS = [
        "1 / 2        Move / Select tool",
        "W / E / R    Gizmo move / rotate / scale",
        "Alt+L-drag   Orbit camera",
        "M-drag       Look      R-drag    Pan",
        "Wheel        Zoom      Arrows    Fly",
        "Ctrl+Z / Y   Undo / redo",
        "Ctrl+D       Duplicate     Del   Delete",
        "Ctrl+A       Select all    F     Frame",
        "Esc          Deselect",
    ].join("\n");
    // --- element helpers ---
    const el = (tag, cls, text) => {
        const e = document.createElement(tag);
        if (cls)
            e.className = cls;
        if (text)
            e.textContent = text;
        return e;
    };
    const button = (label, onClick) => {
        const b = el("button", "btn", label);
        b.addEventListener("click", onClick);
        return b;
    };
    // Hidden file pickers, created once and triggered from the Scene menu.
    const sceneFileInput = el("input");
    sceneFileInput.type = "file";
    sceneFileInput.accept = ".aerie,.csbryce,.json";
    sceneFileInput.style.display = "none";
    sceneFileInput.addEventListener("change", () => {
        const f = sceneFileInput.files?.[0];
        if (f)
            hooks.onLoadScene(f);
    });
    const modelFileInput = el("input");
    modelFileInput.type = "file";
    modelFileInput.accept = acceptAttribute();
    modelFileInput.style.display = "none";
    modelFileInput.addEventListener("change", () => {
        const f = modelFileInput.files?.[0];
        if (f)
            hooks.onImport(f);
    });
    const galleryFileInput = el("input");
    galleryFileInput.type = "file";
    galleryFileInput.accept = ".aeriescene,.json";
    galleryFileInput.style.display = "none";
    galleryFileInput.addEventListener("change", () => {
        const f = galleryFileInput.files?.[0];
        if (f)
            hooks.onImportToGallery(f);
        galleryFileInput.value = ""; // allow re-importing the same file
    });
    document.body.append(sceneFileInput, modelFileInput, galleryFileInput);
    function slider(label, value, min, max, step, onInput) {
        const row = el("div", "row");
        const lab = el("label", "lab", label);
        const range = el("input");
        range.type = "range";
        range.min = String(min);
        range.max = String(max);
        range.step = String(step);
        range.value = String(value);
        const num = el("input", "val");
        num.type = "number";
        num.step = String(step);
        num.value = String(+value.toFixed(4));
        // Range and number box stay in sync; the number box accepts free-hand values
        // (even outside the slider's range — the slider just pegs at its ends).
        range.addEventListener("input", () => {
            const v = parseFloat(range.value);
            num.value = String(+v.toFixed(4));
            onInput(v);
            scene.touch();
            hooks.onEdit();
        });
        num.addEventListener("input", () => {
            const v = parseFloat(num.value);
            if (Number.isNaN(v))
                return;
            range.value = String(v);
            onInput(v);
            scene.touch();
            hooks.onEdit();
        });
        row.append(lab, range, num);
        return row;
    }
    function toggle(label, value, onChange) {
        const row = el("div", "row");
        const lab = el("label", "lab", label);
        const input = el("input");
        input.type = "checkbox";
        input.checked = value;
        input.addEventListener("change", () => {
            onChange(input.checked);
            scene.touchWorld();
            hooks.onEdit();
        });
        row.append(lab, input);
        return row;
    }
    function worldSlider(label, value, min, max, step, set) {
        return slider(label, value, min, max, step, (v) => {
            set(v);
            scene.touchWorld();
        });
    }
    function worldDropdown(label, value, options, set) {
        const row = el("div", "row");
        const lab = el("label", "lab", label);
        const sel = document.createElement("select");
        for (const [val, text] of options) {
            const o = document.createElement("option");
            o.value = String(val);
            o.textContent = text;
            if (val === value)
                o.selected = true;
            sel.append(o);
        }
        sel.addEventListener("change", () => {
            set(parseInt(sel.value, 10));
            scene.touchWorld();
            hooks.onEdit();
        });
        row.append(lab, sel);
        return row;
    }
    function dropdown(label, value, options, onChange) {
        const row = el("div", "row");
        const lab = el("label", "lab", label);
        const sel = document.createElement("select");
        for (const [val, text] of options) {
            const o = document.createElement("option");
            o.value = String(val);
            o.textContent = text;
            if (val === value)
                o.selected = true;
            sel.append(o);
        }
        sel.addEventListener("change", () => {
            onChange(parseInt(sel.value, 10));
            scene.touch();
            hooks.onEdit();
        });
        row.append(lab, sel);
        return row;
    }
    // Plain controls that only call back (no scene.touch / no undo step) — for
    // editing buffered data such as a landform recipe before an explicit re-bake.
    function plainSlider(label, value, min, max, step, set) {
        const row = el("div", "row");
        const range = el("input");
        range.type = "range";
        range.min = String(min);
        range.max = String(max);
        range.step = String(step);
        range.value = String(value);
        const num = el("input", "val");
        num.type = "number";
        num.step = String(step);
        num.value = String(+value.toFixed(4));
        range.addEventListener("input", () => { const v = parseFloat(range.value); num.value = String(+v.toFixed(4)); set(v); });
        num.addEventListener("input", () => { const v = parseFloat(num.value); if (Number.isNaN(v))
            return; range.value = String(v); set(v); });
        row.append(el("label", "lab", label), range, num);
        return row;
    }
    function plainDropdown(label, value, options, set) {
        const row = el("div", "row");
        const sel = document.createElement("select");
        for (const [val, text] of options) {
            const o = document.createElement("option");
            o.value = String(val);
            o.textContent = text;
            if (val === value)
                o.selected = true;
            sel.append(o);
        }
        sel.addEventListener("change", () => set(parseInt(sel.value, 10)));
        row.append(el("label", "lab", label), sel);
        return row;
    }
    function colorPicker(value, onInput) {
        const row = el("div", "row");
        const lab = el("label", "lab", "color");
        const input = el("input");
        input.type = "color";
        const toHex = (c) => Math.round(Math.min(1, Math.max(0, c)) * 255)
            .toString(16)
            .padStart(2, "0");
        input.value = `#${toHex(value[0])}${toHex(value[1])}${toHex(value[2])}`;
        input.addEventListener("input", () => {
            const h = input.value;
            value[0] = parseInt(h.slice(1, 3), 16) / 255;
            value[1] = parseInt(h.slice(3, 5), 16) / 255;
            value[2] = parseInt(h.slice(5, 7), 16) / 255;
            onInput();
            scene.touch();
            hooks.onEdit();
        });
        row.append(lab, input);
        return row;
    }
    /** Editable name for the selected item; live-updates the scene list on blur. */
    function nameField(item) {
        const input = el("input", "name-input");
        input.type = "text";
        input.value = item.name;
        input.spellcheck = false;
        input.addEventListener("input", () => {
            item.name = input.value;
        });
        input.addEventListener("change", () => { hooks.onEdit(); refresh(); });
        return input;
    }
    /** Tab bar for the inspector. Clamps `tab` to a valid key for this object,
     *  then renders one button per tab. */
    function tabBar(tabs) {
        if (!tabs.some(([key]) => key === tab))
            tab = tabs[0][0];
        const bar = el("div", "btns tabs");
        for (const [key, label] of tabs) {
            const b = button(label, () => {
                tab = key;
                refresh();
            });
            if (tab === key)
                b.classList.add("on");
            bar.append(b);
        }
        return bar;
    }
    /** A collapsible section; `build` only runs (and renders) when expanded. */
    function group(title, build) {
        const wrap = el("div", "group");
        const open = openGroups.has(title);
        const header = el("div", "group-h", `${open ? "▾" : "▸"} ${title}`);
        header.addEventListener("click", () => {
            if (open)
                openGroups.delete(title);
            else
                openGroups.add(title);
            refresh();
        });
        wrap.append(header);
        if (open) {
            const body = el("div", "group-body");
            build(body);
            wrap.append(body);
        }
        return wrap;
    }
    const MAT_STORE_KEY = "aerie.materialPresets";
    const loadPresets = () => {
        try {
            const s = localStorage.getItem(MAT_STORE_KEY);
            return s ? JSON.parse(s) : [];
        }
        catch {
            return [];
        }
    };
    const savePresets = () => {
        try {
            localStorage.setItem(MAT_STORE_KEY, JSON.stringify(matPresets));
        }
        catch { /* private mode */ }
    };
    let matPresets = loadPresets();
    let matClip = null; // in-session copy/paste buffer
    const grabPrim = (p) => ({
        color: [p.color[0], p.color[1], p.color[2]],
        colorB: [p.colorB[0], p.colorB[1], p.colorB[2]],
        pattern: p.pattern, patternScale: p.patternScale,
        reflectivity: p.reflectivity, bump: p.bump, bumpScale: p.bumpScale,
    });
    const putPrim = (p, m) => {
        p.color = [m.color[0], m.color[1], m.color[2]];
        p.colorB = [m.colorB[0], m.colorB[1], m.colorB[2]];
        p.pattern = m.pattern;
        p.patternScale = m.patternScale;
        p.reflectivity = m.reflectivity;
        p.bump = m.bump;
        p.bumpScale = m.bumpScale;
    };
    const grabMesh = (bi, mi) => {
        const mt = scene.getMaterial(bi, mi);
        return { color: [mt.color[0], mt.color[1], mt.color[2]], metalness: mt.metalness, roughness: mt.roughness };
    };
    const putMesh = (bi, mi, m) => scene.setMaterial(bi, mi, { color: [m.color[0], m.color[1], m.color[2]], metalness: m.metalness, roughness: m.roughness });
    /** Apply a preset to a dropped object if the kinds match. Mesh presets paint
     *  every material slot ("make this object look like that"). */
    function applyPresetTo(target, preset) {
        if (target instanceof Primitive && preset.kind === "prim" && preset.prim) {
            putPrim(target, preset.prim);
            scene.touch();
            hooks.onCommit();
            return true;
        }
        if (target instanceof MeshInstance && preset.kind === "mesh" && preset.mesh) {
            const count = Math.max(1, scene.materialCount(target.blasIndex));
            for (let i = 0; i < count; i++)
                putMesh(target.blasIndex, i, preset.mesh);
            hooks.onCommit();
            return true;
        }
        return false;
    }
    /** Copy / Paste / Save row in the inspector Material tab. The saved-preset
     *  list lives in the bottom Material Library dock, not here. */
    function materialLibrary(host, kind, capture, apply) {
        host.append(group("Material library", (g) => {
            const tools = el("div", "btns");
            tools.append(button("Copy", () => { matClip = capture(); refresh(); }));
            const paste = button("Paste", () => { if (matClip)
                apply(matClip); });
            if (!matClip || matClip.kind !== kind)
                paste.disabled = true;
            tools.append(paste);
            tools.append(button("Save…", () => {
                const name = prompt("Material preset name:")?.trim();
                if (!name)
                    return;
                matPresets.push({ ...capture(), name });
                savePresets();
                libraryOpen = true;
                refresh();
            }));
            g.append(tools);
            g.append(el("div", "hint", "Saved materials live in the Library dock (bottom). Drag one onto an object, or right-click to copy."));
        }));
    }
    // --- material library dock (bottom): preview swatches, drag-to-apply, right-click-copy ---
    let libraryOpen = localStorage.getItem("aerie.libraryOpen") !== "0";
    // Cheap value-noise fbm for pattern preview swatches (not the real shader).
    const hash2 = (x, y) => {
        const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
        return s - Math.floor(s);
    };
    const vnoise = (x, y) => {
        const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
        const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
        const a = hash2(xi, yi), b = hash2(xi + 1, yi), c = hash2(xi, yi + 1), d = hash2(xi + 1, yi + 1);
        return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
    };
    const fbm = (x, y) => {
        let s = 0, amp = 0.5, f = 1;
        for (let i = 0; i < 4; i++) {
            s += amp * vnoise(x * f, y * f);
            f *= 2;
            amp *= 0.5;
        }
        return s;
    };
    /** Draw a representative thumbnail of a material onto a small canvas. */
    function drawSwatch(cv, p) {
        const w = (cv.width = 96), h = (cv.height = 66);
        const ctx = cv.getContext("2d");
        if (p.kind === "mesh" && p.mesh) {
            const c = p.mesh.color;
            ctx.fillStyle = `rgb(${(c[0] * 255) | 0},${(c[1] * 255) | 0},${(c[2] * 255) | 0})`;
            ctx.fillRect(0, 0, w, h);
            const g = ctx.createRadialGradient(w * 0.36, h * 0.3, 1, w * 0.36, h * 0.3, w * 0.62);
            const gloss = 1 - p.mesh.roughness;
            g.addColorStop(0, `rgba(255,255,255,${0.6 * gloss})`);
            g.addColorStop(1, "rgba(255,255,255,0)");
            ctx.fillStyle = g;
            ctx.fillRect(0, 0, w, h);
            return;
        }
        const m = p.prim;
        const A = m.color, B = m.colorB;
        const img = ctx.createImageData(w, h);
        const d = img.data;
        const patT = (nx, ny) => {
            switch (m.pattern) {
                case 1: return ((Math.floor(nx * 5) + Math.floor(ny * 4)) & 1); // checker
                case 2: return Math.min(1, Math.max(0, fbm(nx * 4, ny * 4))); // noise
                case 3: {
                    const r = fbm(nx * 2, ny * 2) * 2.5 + Math.hypot(nx - 0.2, ny - 0.5) * 7;
                    return Math.abs(((r % 1) + 1) % 1 * 2 - 1);
                } // wood rings
                case 4: return 0.5 + 0.5 * Math.sin(nx * 6 + fbm(nx * 3, ny * 3) * 5); // marble
                case 5: return nx; // gradient
                default: return 0; // solid / image → color A
            }
        };
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const t = patT(x / w, y / h);
                const o = (y * w + x) * 4;
                d[o] = (A[0] + (B[0] - A[0]) * t) * 255;
                d[o + 1] = (A[1] + (B[1] - A[1]) * t) * 255;
                d[o + 2] = (A[2] + (B[2] - A[2]) * t) * 255;
                d[o + 3] = 255;
            }
        }
        ctx.putImageData(img, 0, 0);
        if (m.pattern === 6) { // image pattern: mark it (no per-preset image data)
            ctx.fillStyle = "rgba(0,0,0,0.45)";
            ctx.fillRect(0, h - 16, w, 16);
            ctx.fillStyle = "#dce9ff";
            ctx.font = "10px ui-monospace, monospace";
            ctx.fillText("IMG", 6, h - 5);
        }
    }
    /** Pointer-drag a swatch onto the viewport to apply it to the object there. */
    function startSwatchDrag(preset, e) {
        const startX = e.clientX, startY = e.clientY;
        let ghost = null;
        const move = (ev) => {
            if (!ghost) {
                if (Math.hypot(ev.clientX - startX, ev.clientY - startY) < 5)
                    return; // drag threshold
                ghost = el("div", "drag-ghost");
                const cv = document.createElement("canvas");
                drawSwatch(cv, preset);
                ghost.append(cv);
                document.body.append(ghost);
            }
            ghost.style.left = `${ev.clientX - 32}px`;
            ghost.style.top = `${ev.clientY - 22}px`;
        };
        const up = (ev) => {
            document.removeEventListener("pointermove", move);
            document.removeEventListener("pointerup", up);
            if (ghost)
                ghost.remove();
            if (!ghost)
                return; // never moved → not a drag
            const onCanvas = document.elementFromPoint(ev.clientX, ev.clientY);
            if (onCanvas instanceof HTMLCanvasElement) {
                hooks.onPickAt(ev.clientX, ev.clientY).then((target) => {
                    if (target && applyPresetTo(target, preset)) {
                        hooks.onStatus(`Applied “${preset.name}” to ${target.name}.`);
                        refresh();
                    }
                    else {
                        hooks.onStatus(target ? `“${preset.name}” doesn't fit that object type.` : "Dropped on empty space.");
                    }
                });
            }
        };
        document.addEventListener("pointermove", move);
        document.addEventListener("pointerup", up);
    }
    function buildLibraryDock() {
        libraryDock.replaceChildren();
        // The cutscene timeline takes the bottom while it's open.
        libraryDock.classList.toggle("empty", matPresets.length === 0 || hooks.cutscene.active());
        libraryDock.classList.toggle("collapsed", !libraryOpen);
        if (matPresets.length === 0)
            return;
        const header = el("div", "dock-h");
        header.append(el("div", "dock-title", "Material Library"), el("div", "dock-count", `${matPresets.length}`), el("div", "dock-arrow", libraryOpen ? "▾" : "▴"));
        header.addEventListener("click", () => {
            libraryOpen = !libraryOpen;
            localStorage.setItem("aerie.libraryOpen", libraryOpen ? "1" : "0");
            refresh();
        });
        libraryDock.append(header);
        const body = el("div", "dock-body");
        for (const preset of matPresets) {
            const sw = el("div", "swatch");
            sw.title = `${preset.name} — drag onto an object, right-click to copy`;
            const cv = document.createElement("canvas");
            drawSwatch(cv, preset);
            const name = el("div", "sw-name", preset.name);
            const kind = el("div", "sw-kind", preset.kind === "prim" ? "shape" : "model");
            const del = el("div", "sw-del", "×");
            del.title = "Delete preset";
            sw.append(cv, name, kind, del);
            sw.addEventListener("pointerdown", (e) => {
                if (e.button !== 0)
                    return; // left = drag; right handled by contextmenu
                e.preventDefault();
                startSwatchDrag(preset, e);
            });
            sw.addEventListener("contextmenu", (e) => {
                e.preventDefault();
                matClip = { ...preset };
                hooks.onStatus(`Copied “${preset.name}” — Paste in the Material tab.`);
                refresh();
            });
            del.addEventListener("pointerdown", (e) => e.stopPropagation()); // don't start a drag
            del.addEventListener("click", (e) => {
                e.stopPropagation();
                matPresets.splice(matPresets.indexOf(preset), 1);
                savePresets();
                refresh();
            });
            body.append(sw);
        }
        libraryDock.append(body);
    }
    // --- cutscene timeline dock ---
    function buildCutsceneDock() {
        const cs = hooks.cutscene;
        cutsceneDock.replaceChildren();
        cutsceneDock.classList.toggle("hidden", !cs.active());
        if (!cs.active())
            return;
        const n = cs.count();
        const total = cs.duration();
        const head = el("div", "cs-head");
        head.append(el("div", "cs-title", "Cutscene"));
        const play = button(cs.playing() ? "⏸ Pause" : "▶ Play", () => cs.playPause());
        play.style.flex = "0 0 auto";
        if (n < 2)
            play.disabled = true;
        head.append(play, el("div", "cs-time", `${cs.time().toFixed(1)} / ${total.toFixed(1)}s`), el("div", "cs-spacer"));
        const addBtn = button("＋ Add keyframe", () => cs.add());
        addBtn.style.flex = "0 0 auto";
        const renderBtn = button("Render WebM", () => cs.render({ ...turntable }));
        renderBtn.style.flex = "0 0 auto";
        if (n < 2)
            renderBtn.disabled = true;
        const exitBtn = button("✕", () => cs.toggle());
        exitBtn.style.flex = "0 0 auto";
        head.append(addBtn, renderBtn, exitBtn);
        cutsceneDock.append(head);
        if (n === 0) {
            cutsceneDock.append(el("div", "hint", "Pose the camera, DoF, and your objects, then “Add keyframe”. Each key snapshots the view + every object's transform; move things between keys to animate them. Objects return to their saved state when you exit."));
            return;
        }
        const scrub = el("input", "cs-scrub");
        scrub.type = "range";
        scrub.min = "0";
        scrub.max = String(Math.max(0.0001, total));
        scrub.step = "0.01";
        scrub.value = String(cs.time());
        scrub.addEventListener("input", () => cs.scrub(parseFloat(scrub.value)));
        cutsceneDock.append(scrub);
        const strip = el("div", "cs-keys");
        for (let i = 0; i < n; i++) {
            const info = cs.keyInfo(i);
            const chip = el("div", "cs-key" + (i === cs.selected() ? " sel" : ""));
            chip.append(el("div", "cs-k-i", `K${i + 1}`), el("div", "cs-k-t", `${info.time.toFixed(1)}s`));
            chip.addEventListener("click", () => cs.select(i));
            strip.append(chip);
        }
        cutsceneDock.append(strip);
        const sel = cs.selected();
        if (sel >= 0 && sel < n) {
            const info = cs.keyInfo(sel);
            const ed = el("div", "cs-editor");
            if (sel > 0)
                ed.append(plainSlider("duration", info.duration, 0, 20, 0.1, (v) => cs.setDuration(sel, v)));
            else
                ed.append(el("div", "hint", "First keyframe is the start (no incoming duration)."));
            const easeRow = el("div", "btns");
            ["smooth", "linear"].forEach((e) => {
                const b = button(e, () => cs.setEase(sel, e));
                if (info.ease === e)
                    b.classList.add("on");
                easeRow.append(b);
            });
            ed.append(easeRow);
            // Atmosphere: animate time of day (sun + sky), exposure and haze per key.
            const atmo = cs.keyAtmo(sel);
            ed.append(el("div", "hint", "Atmosphere — animate the light across the shot"));
            ed.append(plainSlider("time of day", atmo.timeOfDay, 0, 24, 0.1, (v) => cs.setKeyAtmo(sel, "timeOfDay", v)));
            ed.append(plainSlider("exposure", atmo.exposure, 0.2, 3, 0.05, (v) => cs.setKeyAtmo(sel, "exposure", v)));
            ed.append(plainSlider("haze", atmo.haze, 0, 0.02, 0.0005, (v) => cs.setKeyAtmo(sel, "haze", v)));
            const ops = el("div", "btns");
            ops.append(button("Jump to", () => cs.select(sel)), button("Re-capture", () => cs.recapture(sel)), button("Delete", () => cs.remove(sel)));
            ed.append(ops);
            ed.append(el("div", "hint", "Jump to: pose the scene at this keyframe. Re-capture: overwrite it with the current camera, DoF, and object transforms."));
            cutsceneDock.append(ed);
        }
    }
    // --- top menu bar ---
    /** A menu-bar dropdown. `build` only runs while the menu is open. */
    function menu(name, label, build, align = "left") {
        const wrap = el("div", "menu" + (openMenu === name ? " open" : ""));
        const btn = el("button", "menu-btn", label);
        btn.addEventListener("click", (e) => {
            e.stopPropagation();
            openMenu = openMenu === name ? null : name;
            refresh();
        });
        wrap.append(btn);
        if (openMenu === name) {
            const pop = el("div", "menu-pop" + (align === "right" ? " right" : ""));
            pop.addEventListener("click", (e) => e.stopPropagation());
            build(pop);
            wrap.append(pop);
        }
        return wrap;
    }
    /** A single dropdown row. Closes the menu after acting unless `keep` is set
     *  (toggles that should stay open for repeated tweaking). */
    function menuItem(label, onClick, opts = {}) {
        const b = el("button", "menu-item");
        b.append(el("span", "mi-label", label));
        if (opts.dot)
            b.append(el("span", "dot", "●"));
        b.addEventListener("click", (e) => {
            e.stopPropagation();
            if (!opts.keep)
                openMenu = null;
            onClick();
            refresh();
        });
        return b;
    }
    /** Boolean (CSG) role + group selectors + a bake button — shared by prims/models. */
    function booleanControls(box, obj) {
        box.append(el("div", "hint", "Boolean"));
        box.append(dropdown("role", obj.subtractive ? 1 : 0, [[0, "Solid"], [1, "Carve out"]], (v) => {
            obj.subtractive = v === 1;
            refresh();
        }));
        const groupOpts = [[0, "none"]];
        for (let i = 1; i <= MAX_BOOL_GROUPS; i++)
            groupOpts.push([i, `group ${i}`]);
        box.append(dropdown("group", obj.group, groupOpts, (v) => { obj.group = v; refresh(); }));
        if (obj.group > 0) {
            box.append(button(`Bake boolean (group ${obj.group})`, () => hooks.onBakeBoolean(obj.group)));
            box.append(el("div", "hint", "Merges group's Solids minus its Carvers into one mesh."));
        }
    }
    // --- inspector ---
    function buildInspector() {
        const box = el("div", "inspector");
        const selected = primary();
        if (!selected) {
            box.append(el("div", "hint", "Nothing selected."));
            return box;
        }
        if (selection.length > 1) {
            box.append(el("div", "hint", `${selection.length} selected · editing primary`));
            box.append(group("Arrange", (g) => {
                const dist = selection.length > 2;
                ["X", "Y", "Z"].forEach((label, a) => {
                    const row = el("div", "btns");
                    row.append(el("label", "lab", `align ${label}`));
                    row.append(button("min", () => hooks.onAlign(a, "min")));
                    row.append(button("mid", () => hooks.onAlign(a, "center")));
                    row.append(button("max", () => hooks.onAlign(a, "max")));
                    if (dist)
                        row.append(button("dist", () => hooks.onDistribute(a)));
                    g.append(row);
                });
                g.append(plainSlider("scatter amt", scatterAmount, 1, 80, 1, (v) => (scatterAmount = v)));
                const sc = el("div", "btns");
                sc.append(button("Scatter", () => hooks.onScatter(scatterAmount)));
                g.append(sc);
            }));
        }
        box.append(nameField(selected));
        if (selected instanceof Primitive) {
            const p = selected;
            box.append(tabBar([["transform", "Transform"], ["size", "Size"], ["material", "Material"]]));
            if (tab === "transform") {
                box.append(slider("pos x", p.position.x, -120, 120, 0.5, (v) => (p.position.x = v)));
                box.append(slider("pos y", p.position.y, -10, 80, 0.5, (v) => (p.position.y = v)));
                box.append(slider("pos z", p.position.z, -120, 120, 0.5, (v) => (p.position.z = v)));
                box.append(slider("rot x", p.rotation.x, -3.14, 3.14, 0.02, (v) => (p.rotation.x = v)));
                box.append(slider("rot y", p.rotation.y, -3.14, 3.14, 0.02, (v) => (p.rotation.y = v)));
                box.append(slider("rot z", p.rotation.z, -3.14, 3.14, 0.02, (v) => (p.rotation.z = v)));
                booleanControls(box, p);
                box.append(button("Edit points →", () => { tab = "edit"; hooks.onEditPrim(p); }));
                box.append(el("div", "hint", "Converts this shape to an editable mesh you can point-stretch."));
            }
            else if (tab === "size") {
                const dim = (lbl, key) => box.append(slider(lbl, p[key], 0.5, 20, 0.1, (v) => (p[key] = v)));
                switch (p.type) {
                    case PrimType.Sphere:
                        dim("radius", "a");
                        break;
                    case PrimType.Octahedron:
                        dim("size", "a");
                        break;
                    case PrimType.Box:
                        dim("half-x", "a");
                        dim("half-y", "b");
                        dim("half-z", "c");
                        break;
                    case PrimType.Torus:
                        dim("major R", "a");
                        dim("minor r", "b");
                        break;
                    case PrimType.Cylinder:
                        dim("radius", "a");
                        dim("half-height", "b");
                        break;
                    case PrimType.Cone:
                        dim("radius", "a");
                        dim("half-height", "b");
                        break;
                    case PrimType.Capsule:
                        dim("radius", "a");
                        dim("half-height", "b");
                        break;
                    case PrimType.Pyramid:
                        dim("base half", "a");
                        dim("half-height", "b");
                        break;
                    case PrimType.RoundedBox:
                        dim("half-x", "a");
                        dim("half-y", "b");
                        dim("half-z", "c");
                        box.append(slider("corner", p.cornerRadius, 0, Math.min(p.a, p.b, p.c), 0.05, (v) => (p.cornerRadius = v)));
                        break;
                }
            }
            else {
                box.append(slider("reflectivity", p.reflectivity, 0, 1, 0.01, (v) => (p.reflectivity = v)));
                box.append(dropdown("pattern", p.pattern, Object.keys(PATTERN_LABELS).map((k) => [Number(k), PATTERN_LABELS[k]]), (v) => {
                    p.pattern = v;
                    refresh();
                }));
                box.append(el("div", "hint", "color A"));
                box.append(colorPicker(p.color, () => { }));
                if (p.pattern !== PrimPattern.Solid && p.pattern !== PrimPattern.Image) {
                    box.append(el("div", "hint", "color B"));
                    box.append(colorPicker(p.colorB, () => { }));
                    box.append(slider("scale", p.patternScale, 0.2, 30, 0.1, (v) => (p.patternScale = v)));
                }
                if (p.pattern === PrimPattern.Image) {
                    box.append(slider("tex scale", p.patternScale, 0.2, 30, 0.1, (v) => (p.patternScale = v)));
                    const ib = el("div", "btns");
                    ib.append(button("Upload image", () => hooks.onPrimImage(p)));
                    box.append(ib);
                    box.append(el("div", "hint", p.imageLayer >= 0 ? `image #${p.imageLayer}` : "no image yet"));
                }
                box.append(slider("bump", p.bump, 0, 1, 0.02, (v) => (p.bump = v)));
                if (p.bump > 0) {
                    box.append(slider("bump scale", p.bumpScale, 0.5, 30, 0.1, (v) => (p.bumpScale = v)));
                }
                // Save/apply the full procedural surface; Paste/Apply hits every selected prim.
                materialLibrary(box, "prim", () => ({ name: "clipboard", kind: "prim", prim: grabPrim(p) }), (preset) => {
                    if (!preset.prim)
                        return;
                    for (const it of selection)
                        if (it instanceof Primitive)
                            putPrim(it, preset.prim);
                    scene.touch();
                    hooks.onCommit();
                    refresh();
                });
            }
        }
        else if (selected instanceof MeshInstance) {
            const m = selected;
            const bump = () => scene.touchInstances();
            // While point-editing, keep Done reachable from any tab.
            if (hooks.isEditing()) {
                box.append(button("✓ Done editing", () => {
                    hooks.onToggleEdit(m);
                    refresh();
                }));
                box.append(el("div", "hint", "Click points to select (Shift=multi); drag the axes to move."));
            }
            const mtabs = [["transform", "Transform"], ["size", "Size"], ["material", "Material"], ["pose", "Pose"], ["edit", "Edit"]];
            if (m.landform)
                mtabs.push(["terrain", "Terrain"]);
            box.append(tabBar(mtabs));
            if (tab === "transform") {
                box.append(slider("pos x", m.position.x, -120, 120, 0.5, (v) => { m.position.x = v; bump(); }));
                box.append(slider("pos y", m.position.y, -10, 80, 0.5, (v) => { m.position.y = v; bump(); }));
                box.append(slider("pos z", m.position.z, -120, 120, 0.5, (v) => { m.position.z = v; bump(); }));
                box.append(slider("rot x", m.rotation.x, -3.14, 3.14, 0.02, (v) => { m.rotation.x = v; bump(); }));
                box.append(slider("rot y", m.rotation.y, -3.14, 3.14, 0.02, (v) => { m.rotation.y = v; bump(); }));
                box.append(slider("rot z", m.rotation.z, -3.14, 3.14, 0.02, (v) => { m.rotation.z = v; bump(); }));
                booleanControls(box, m);
            }
            else if (tab === "size") {
                box.append(slider("scale", m.scale, 0.1, 6, 0.05, (v) => { m.scale = v; bump(); }));
            }
            else if (tab === "material") {
                const count = scene.materialCount(m.blasIndex);
                if (count === 0) {
                    box.append(el("div", "hint", "This model has no materials."));
                }
                else {
                    if (matIndex >= count)
                        matIndex = 0;
                    if (count > 1) {
                        box.append(dropdown("material", matIndex, Array.from({ length: count }, (_, i) => [i, scene.getMaterial(m.blasIndex, i).name]), (v) => { matIndex = v; refresh(); }));
                    }
                    const mat = scene.getMaterial(m.blasIndex, matIndex);
                    box.append(toggle("hidden", mat.hidden, (v) => {
                        scene.setMaterial(m.blasIndex, matIndex, { hidden: v });
                    }));
                    box.append(el("div", "hint", "base color"));
                    box.append(colorPicker(mat.color, () => scene.setMaterial(m.blasIndex, matIndex, { color: mat.color })));
                    box.append(slider("metalness", mat.metalness, 0, 1, 0.02, (v) => scene.setMaterial(m.blasIndex, matIndex, { metalness: v })));
                    box.append(slider("roughness", mat.roughness, 0, 1, 0.02, (v) => scene.setMaterial(m.blasIndex, matIndex, { roughness: v })));
                    box.append(el("div", "hint", mat.texLayer >= 0 ? `texture #${mat.texLayer}` : "no texture"));
                    const texBtns = el("div", "btns");
                    texBtns.append(button("Upload texture", () => hooks.onMeshTexture(m.blasIndex, matIndex)));
                    if (mat.texLayer >= 0) {
                        texBtns.append(button("Clear", () => {
                            scene.setMaterial(m.blasIndex, matIndex, { texLayer: -1 });
                            refresh();
                        }));
                    }
                    box.append(texBtns);
                    box.append(el("div", "hint", "Upload a base-color image if it imported white (UVs are kept)."));
                    if (count > 1) {
                        box.append(button("Apply metal/rough to all", () => {
                            scene.applyModelPBR(m.blasIndex, mat.metalness, mat.roughness);
                            refresh();
                        }));
                    }
                    // Library applies to the currently-selected material slot (base color + PBR).
                    materialLibrary(box, "mesh", () => ({ name: "clipboard", kind: "mesh", mesh: grabMesh(m.blasIndex, matIndex) }), (preset) => {
                        if (!preset.mesh)
                            return;
                        scene.setMaterial(m.blasIndex, matIndex, {
                            color: [preset.mesh.color[0], preset.mesh.color[1], preset.mesh.color[2]],
                            metalness: preset.mesh.metalness,
                            roughness: preset.mesh.roughness,
                        });
                        hooks.onCommit();
                        refresh();
                    });
                }
            }
            else if (tab === "pose") {
                const bones = hooks.boneNames(m);
                if (bones.length === 0) {
                    box.append(el("div", "hint", "This model has no rig (skeleton) to pose."));
                }
                else {
                    const posing = hooks.isPosing();
                    box.append(button(posing ? "✓ Done posing" : `Pose rig (${bones.length} bones)`, () => {
                        hooks.onTogglePose(m);
                        refresh();
                    }));
                    if (posing) {
                        box.append(el("div", "hint", "Click a joint and drag to rotate it; or use the sliders."));
                        // Root (whole-armature) translation.
                        if (hooks.poseRootIndex() >= 0) {
                            const rp = hooks.rootPosition();
                            box.append(el("div", "hint", "Root position"));
                            box.append(slider("root x", rp[0], -50, 50, 0.25, (v) => hooks.setRootPosition(0, v)));
                            box.append(slider("root y", rp[1], -50, 50, 0.25, (v) => hooks.setRootPosition(1, v)));
                            box.append(slider("root z", rp[2], -50, 50, 0.25, (v) => hooks.setRootPosition(2, v)));
                            box.append(button("Reset root", () => { hooks.resetRoot(); refresh(); }));
                        }
                        const sel = hooks.poseSelectedBone();
                        box.append(el("div", "hint", "Bone rotation"));
                        box.append(dropdown("bone", sel, bones.map((n, i) => [i, n]), (v) => { hooks.onSelectBone(v); refresh(); }));
                        if (sel >= 0) {
                            const r = hooks.boneRotation(sel);
                            box.append(slider("rot x", r[0], -3.14, 3.14, 0.01, (v) => hooks.setBoneRotation(sel, 0, v)));
                            box.append(slider("rot y", r[1], -3.14, 3.14, 0.01, (v) => hooks.setBoneRotation(sel, 1, v)));
                            box.append(slider("rot z", r[2], -3.14, 3.14, 0.01, (v) => hooks.setBoneRotation(sel, 2, v)));
                            box.append(button("Reset bone", () => { hooks.resetBone(sel); refresh(); }));
                        }
                        box.append(button("Bake pose → render", () => hooks.onBakePose()));
                        box.append(el("div", "hint", "Bakes the posed model into the ray-traced view."));
                    }
                }
            }
            else if (tab === "terrain" && m.landform) {
                const lf = m.landform;
                const r = lf.recipe;
                const l0 = r.layers[0];
                const BASES = ["value", "gradient", "worley"];
                const FRACTALS = ["fbm", "ridged", "billow"];
                const MASKS = ["none", "peak", "plateau"];
                box.append(el("div", "hint", `${LANDSCAPE_LABELS[lf.kind]} · seed ${lf.seed}`));
                box.append(plainSlider("amplitude", r.amplitude, 1, 60, 0.5, (v) => (r.amplitude = v)));
                box.append(plainSlider("octaves", l0.octaves, 1, 8, 1, (v) => (l0.octaves = v)));
                box.append(plainDropdown("basis", BASES.indexOf(l0.basis), [[0, "Value"], [1, "Gradient"], [2, "Worley"]], (v) => (l0.basis = BASES[v])));
                box.append(plainDropdown("fractal", FRACTALS.indexOf(l0.fractal), [[0, "fbm"], [1, "Ridged"], [2, "Billow"]], (v) => (l0.fractal = FRACTALS[v])));
                box.append(plainSlider("terrace steps", r.terraceSteps, 0, 8, 1, (v) => (r.terraceSteps = v)));
                box.append(plainSlider("terrace sharp", r.terraceSharpness, 0, 1, 0.05, (v) => (r.terraceSharpness = v)));
                box.append(plainSlider("warp", r.warpStrength, 0, 0.6, 0.02, (v) => (r.warpStrength = v)));
                box.append(plainDropdown("mask", MASKS.indexOf(r.mask), [[0, "None"], [1, "Peak"], [2, "Plateau"]], (v) => (r.mask = MASKS[v])));
                box.append(plainSlider("mask radius", r.maskRadius, 0.2, 1.2, 0.02, (v) => (r.maskRadius = v)));
                const acts = el("div", "btns");
                acts.append(button("Regenerate", () => hooks.onRegenerateLandform(m)));
                acts.append(button("Re-roll seed", () => hooks.onRerollLandform(m)));
                box.append(acts);
                box.append(el("div", "hint", "Tweak knobs, then Regenerate to re-bake the mesh."));
                box.append(group("Scatter", (g) => {
                    g.append(plainDropdown("shape", eco.shape, ECO_SHAPES.map(([t, l]) => [t, l]), (v) => { eco.shape = v; }));
                    g.append(plainSlider("count", eco.count, 1, 400, 1, (v) => (eco.count = Math.round(v))));
                    g.append(plainSlider("alt min", eco.altMin, 0, 1, 0.02, (v) => (eco.altMin = v)));
                    g.append(plainSlider("alt max", eco.altMax, 0, 1, 0.02, (v) => (eco.altMax = v)));
                    g.append(plainSlider("max slope", eco.maxSlope, 0, 1, 0.02, (v) => (eco.maxSlope = v)));
                    g.append(plainSlider("scale min", eco.scaleMin, 0.1, 4, 0.05, (v) => (eco.scaleMin = v)));
                    g.append(plainSlider("scale max", eco.scaleMax, 0.1, 4, 0.05, (v) => (eco.scaleMax = v)));
                    g.append(button(`align to surface: ${eco.align ? "on" : "off"}`, () => { eco.align = !eco.align; refresh(); }));
                    g.append(el("div", "hint", "color"));
                    g.append(colorPicker(eco.color, () => { }));
                    g.append(button(`Scatter ${eco.count} →`, () => hooks.onScatterEcosystem(m, { ...eco, color: [...eco.color] })));
                    g.append(el("div", "hint", "Places instances on this landform; all share one mesh, so it stays cheap."));
                }));
            }
            else {
                // "edit" tab — direct vertex (point-stretch) editing. Edits apply live
                // as you drag, so there is nothing to bake here.
                if (hooks.isEditing()) {
                    box.append(el("div", "hint", "Drag points in the viewport to reshape the mesh. " +
                        "Changes apply live — use “✓ Done editing” above when finished."));
                }
                else {
                    box.append(button("Edit vertices", () => {
                        hooks.onToggleEdit(m);
                        refresh();
                    }));
                    box.append(el("div", "hint", "Point-stretch this mesh. Edits show in the ray-traced view."));
                }
            }
        }
        else {
            const l = selected;
            box.append(tabBar([["transform", "Transform"], ["light", "Light"]]));
            if (tab === "transform") {
                if (l.type === LightType.Directional) {
                    box.append(slider("dir x", l.direction.x, -1, 1, 0.02, (v) => { l.direction.x = v; l.direction.normalize(); }));
                    box.append(slider("dir y", l.direction.y, 0.02, 1, 0.02, (v) => { l.direction.y = v; l.direction.normalize(); }));
                    box.append(slider("dir z", l.direction.z, -1, 1, 0.02, (v) => { l.direction.z = v; l.direction.normalize(); }));
                    box.append(el("div", "hint", "Drag the sun handle in the viewport to aim it."));
                }
                else {
                    const dirMode = l.inSky;
                    box.append(slider(dirMode ? "dir x" : "pos x", l.position.x, -120, 120, 0.5, (v) => (l.position.x = v)));
                    box.append(slider(dirMode ? "dir y" : "pos y", l.position.y, 0, 200, 0.5, (v) => (l.position.y = v)));
                    box.append(slider(dirMode ? "dir z" : "pos z", l.position.z, -120, 120, 0.5, (v) => (l.position.z = v)));
                    box.append(slider("body radius", l.bodyRadius, 0, 40, 0.5, (v) => (l.bodyRadius = v)));
                    box.append(el("div", "hint", "Body radius > 0 shows a glowing planet that casts light."));
                    if (l.bodyRadius > 0) {
                        const row = el("div", "row");
                        const lab = el("label", "lab", "in sky (distant)");
                        const input = el("input");
                        input.type = "checkbox";
                        input.checked = l.inSky;
                        input.addEventListener("change", () => {
                            l.inSky = input.checked;
                            scene.touch();
                            hooks.onEdit();
                            refresh();
                        });
                        row.append(lab, input);
                        box.append(row);
                        box.append(el("div", "hint", l.inSky
                            ? "Pushed far away like the sun — never collides; sliders aim its sky direction and it casts directional light."
                            : "Spawn it into the skybox: far away like the sun so it can never collide with the scene."));
                    }
                }
            }
            else {
                box.append(slider("intensity", l.intensity, 0, l.type === LightType.Directional ? 8 : 4000, l.type === LightType.Directional ? 0.05 : 10, (v) => (l.intensity = v)));
                if (l.type === LightType.Point) {
                    box.append(slider("range", l.range, 10, 400, 1, (v) => (l.range = v)));
                }
                box.append(slider("softness", l.softness, 0, 0.3, 0.005, (v) => (l.softness = v)));
                box.append(el("div", "hint", l.bodyRadius > 0 ? "planet / light color" : "color"));
                box.append(colorPicker(l.color, () => { }));
                // Planet rings (point lights with a visible body).
                if (l.type === LightType.Point && l.bodyRadius > 0) {
                    box.append(el("div", "h", "Rings"));
                    box.append(slider("opacity", l.ringOpacity, 0, 1, 0.02, (v) => (l.ringOpacity = v)));
                    box.append(slider("inner", l.ringInner, 1.05, 4, 0.05, (v) => (l.ringInner = v)));
                    box.append(slider("outer", l.ringOuter, 1.1, 6, 0.05, (v) => (l.ringOuter = v)));
                    box.append(slider("tilt", l.ringTilt, -1.57, 1.57, 0.02, (v) => (l.ringTilt = v)));
                    box.append(el("div", "hint", "ring color"));
                    box.append(colorPicker(l.ringColor, () => { }));
                    box.append(el("div", "hint", "Opacity: Saturn ≈ 0.9, Jupiter ≈ 0.25."));
                }
            }
        }
        const footer = el("div", "insp-footer");
        const mirrorRow = el("div", "btns");
        mirrorRow.append(el("label", "lab", "mirror"));
        ["X", "Y", "Z"].forEach((label, a) => mirrorRow.append(button(label, () => hooks.onMirror(a))));
        footer.append(mirrorRow);
        const actions = el("div", "btns");
        actions.append(button(selection.length > 1 ? `Duplicate ${selection.length}` : "Duplicate", () => hooks.onDuplicate()), button(selection.length > 1 ? `Delete ${selection.length}` : "Delete", () => {
            for (const it of selection)
                scene.remove(it);
            const fallback = scene.prims[0] ?? scene.lights[0] ?? null;
            selection = fallback ? [fallback] : [];
            hooks.onCommit();
            refresh();
        }));
        footer.append(actions);
        box.append(footer);
        return box;
    }
    // --- object list ---
    function buildList() {
        const list = el("div", "list");
        const items = [...scene.prims, ...scene.instances, ...scene.lights];
        for (const item of items) {
            let tag = item instanceof Primitive
                ? PRIM_LABELS[item.type]
                : item instanceof MeshInstance
                    ? "Model"
                    : "Light";
            if (item instanceof Primitive || item instanceof MeshInstance) {
                if (item.subtractive)
                    tag += " ⊖";
                if (item.group > 0)
                    tag += ` g${item.group}`;
            }
            const row = el("div", "item" + (selection.includes(item) ? " sel" : ""));
            row.append(el("span", "name", item.name), el("span", "kind", tag));
            row.addEventListener("click", (e) => {
                if (e.shiftKey || e.ctrlKey) {
                    const i = selection.indexOf(item);
                    if (i >= 0)
                        selection.splice(i, 1);
                    else
                        selection.push(item);
                }
                else {
                    selection = [item];
                }
                refresh();
            });
            list.append(row);
        }
        return list;
    }
    // --- the procedural "labs": terrain / sky / water / render ---
    // Rendered into the bottom-left labsPanel; which lab shows is driven by the
    // Labs menu (activeLab). Caller guarantees activeLab is non-null here.
    function buildLabs() {
        const w = scene.world;
        labsPanel.append(el("div", "h", `${activeLab[0].toUpperCase() + activeLab.slice(1)} lab`));
        const box = el("div", "inspector");
        if (activeLab === "terrain") {
            box.append(toggle("enabled", w.terrainEnabled, (v) => (w.terrainEnabled = v)));
            box.append(el("div", "hint", "Recipe"));
            const recipes = el("div", "btns");
            LANDSCAPE_TYPES.forEach((rk) => recipes.append(button(LANDSCAPE_LABELS[rk] + (w.terrainRecipe === rk ? " •" : ""), () => {
                scene.applyTerrainRecipe(rk);
                hooks.onCommit();
                refresh();
            })));
            box.append(recipes);
            const seedRow = el("div", "btns");
            seedRow.append(button("Re-roll seed", () => { scene.randomizeTerrain(); hooks.onCommit(); }));
            box.append(seedRow);
            box.append(worldSlider("amplitude", w.terrainAmp, 0, 60, 0.5, (v) => (w.terrainAmp = v)));
            box.append(worldSlider("frequency", w.terrainFreq, 0.002, 0.05, 0.001, (v) => (w.terrainFreq = v)));
            box.append(worldSlider("ridges", w.terrainRidge, 0.5, 3, 0.05, (v) => (w.terrainRidge = v)));
            box.append(worldSlider("detail", w.terrainOctaves, 1, 8, 1, (v) => (w.terrainOctaves = v)));
            box.append(worldSlider("warp", w.terrainWarp, 0, 40, 0.5, (v) => (w.terrainWarp = v)));
            box.append(worldSlider("sea level", w.terrainOffset, -20, 10, 0.5, (v) => (w.terrainOffset = v)));
            box.append(group("Advanced", (g) => {
                const BASES = [[0, "Value"], [1, "Gradient"], [2, "Worley"]];
                const FRACTALS = [[0, "fbm"], [1, "Ridged"], [2, "Billow"]];
                g.append(worldDropdown("basis", w.terrainBasis, BASES, (v) => (w.terrainBasis = v)));
                g.append(worldDropdown("fractal", w.terrainFractal, FRACTALS, (v) => (w.terrainFractal = v)));
                g.append(worldSlider("terrace steps", w.terrainTerraceSteps, 0, 8, 1, (v) => (w.terrainTerraceSteps = v)));
                g.append(worldSlider("terrace sharp", w.terrainTerraceSharp, 0, 1, 0.05, (v) => (w.terrainTerraceSharp = v)));
                g.append(worldSlider("warp freq", w.terrainWarpFreq, 0.1, 2, 0.05, (v) => (w.terrainWarpFreq = v)));
                g.append(worldSlider("snow line", w.terrainSnowLine, 0, 1.1, 0.02, (v) => (w.terrainSnowLine = v)));
                g.append(worldSlider("slope rock", w.terrainSlopeRock, 0, 1, 0.05, (v) => (w.terrainSlopeRock = v)));
                g.append(el("div", "hint", "second layer"));
                g.append(worldSlider("mix", w.terrainWeight2, 0, 1, 0.05, (v) => (w.terrainWeight2 = v)));
                g.append(worldDropdown("basis 2", w.terrainBasis2, BASES, (v) => (w.terrainBasis2 = v)));
                g.append(worldDropdown("fractal 2", w.terrainFractal2, FRACTALS, (v) => (w.terrainFractal2 = v)));
                g.append(worldSlider("freq 2", w.terrainFreq2, 0.25, 4, 0.05, (v) => (w.terrainFreq2 = v)));
                g.append(worldSlider("detail 2", w.terrainOctaves2, 1, 8, 1, (v) => (w.terrainOctaves2 = v)));
            }));
            box.append(el("div", "hint", "Biome"));
            const biomes = el("div", "btns");
            ["alpine", "desert", "tundra", "badlands"].forEach((bm) => biomes.append(button(bm[0].toUpperCase() + bm.slice(1), () => { scene.applyBiome(bm); hooks.onCommit(); refresh(); })));
            box.append(biomes);
            box.append(el("div", "hint", "ground"));
            box.append(colorPicker(w.terrainLow, () => scene.touchWorld()));
            box.append(el("div", "hint", "slopes"));
            box.append(colorPicker(w.terrainRock, () => scene.touchWorld()));
            box.append(el("div", "hint", "peaks"));
            box.append(colorPicker(w.terrainHigh, () => scene.touchWorld()));
        }
        else if (activeLab === "sky") {
            // Presets
            box.append(group("Presets", (g) => {
                const presets = el("div", "btns");
                ["day", "sunset", "night", "overcast"].forEach((p) => presets.append(button(p[0].toUpperCase() + p.slice(1), () => {
                    scene.applySkyPreset(p);
                    hooks.onCommit();
                    refresh();
                })));
                g.append(presets);
            }));
            // Coloration — zenith & horizon
            box.append(group("Coloration", (g) => {
                g.append(el("div", "hint", "Zenith"));
                g.append(colorPicker(w.zenith, () => scene.touchWorld()));
                g.append(el("div", "hint", "Horizon"));
                g.append(colorPicker(w.horizon, () => scene.touchWorld()));
            }));
            // Clouds — stacked layers
            box.append(group("Clouds", (g) => {
                g.append(toggle("enabled", w.cloudsEnabled, (v) => (w.cloudsEnabled = v)));
                const cloudOpts = Object.keys(CLOUD_LABELS).map((k) => [Number(k), CLOUD_LABELS[k]]);
                w.cloudLayers.forEach((layer, idx) => {
                    g.append(el("div", "hint", `Layer ${idx + 1}`));
                    g.append(dropdown("type", layer.type, cloudOpts, (v) => {
                        layer.type = v;
                        scene.touchWorld();
                        refresh();
                    }));
                    g.append(worldSlider("coverage", layer.coverage, 0, 1, 0.02, (v) => (layer.coverage = v)));
                    g.append(worldSlider("density", layer.density, 0, 1, 0.02, (v) => (layer.density = v)));
                    g.append(worldSlider("thickness", layer.thickness, 0, 1, 0.02, (v) => (layer.thickness = v)));
                    g.append(worldSlider("base light", layer.darkness, 0, 1, 0.02, (v) => (layer.darkness = v)));
                    g.append(worldSlider("altitude", layer.altitude, 0, 1, 0.02, (v) => (layer.altitude = v)));
                    g.append(worldSlider("scale", layer.scale, 0.2, 4, 0.05, (v) => (layer.scale = v)));
                    g.append(el("div", "hint", "cloud color"));
                    g.append(colorPicker(layer.color, () => scene.touchWorld()));
                    const lrow = el("div", "btns");
                    lrow.append(button("Randomize", () => { scene.randomizeCloudLayer(layer); refresh(); }));
                    lrow.append(button("Remove", () => { scene.removeCloudLayer(layer); refresh(); }));
                    g.append(lrow);
                });
                if (w.cloudLayers.length < MAX_CLOUD_LAYERS) {
                    g.append(el("div", "hint", "Add layer"));
                    const addRow = el("div", "btns");
                    const sel = document.createElement("select");
                    for (const [val, text] of cloudOpts) {
                        const o = document.createElement("option");
                        o.value = String(val);
                        o.textContent = text;
                        if (val === addCloudType)
                            o.selected = true;
                        sel.append(o);
                    }
                    sel.style.flex = "1 1 auto";
                    sel.addEventListener("change", () => { addCloudType = Number(sel.value); });
                    const addBtn = button("Add", () => { scene.addCloudLayer(addCloudType); refresh(); });
                    addBtn.style.flex = "0 0 auto";
                    addRow.append(sel, addBtn);
                    g.append(addRow);
                }
            }));
            // Lighting — time of day + sun
            box.append(group("Lighting", (g) => {
                g.append(slider("time", w.timeOfDay, 0, 24, 0.1, (v) => scene.applyTimeOfDay(v)));
                const jumps = el("div", "btns");
                [["Dawn", 6.5], ["Noon", 12], ["Dusk", 18], ["Night", 22]].forEach(([lbl, h]) => jumps.append(button(lbl, () => { scene.applyTimeOfDay(h); hooks.onCommit(); refresh(); })));
                g.append(jumps);
                g.append(worldSlider("sun size", w.sunSize, 0, 1, 0.02, (v) => (w.sunSize = v)));
                g.append(el("div", "hint", "Time sweeps the sun's arc, colour & sky tint together. Presets still apply full looks."));
            }));
            box.append(group("Stars", (g) => {
                g.append(toggle("enabled", w.starsEnabled, (v) => (w.starsEnabled = v)));
                g.append(worldSlider("density", w.starDensity, 0, 1, 0.02, (v) => (w.starDensity = v)));
                g.append(worldSlider("brightness", w.starBrightness, 0, 3, 0.05, (v) => (w.starBrightness = v)));
                g.append(worldSlider("size", w.starSize, 0, 1, 0.02, (v) => (w.starSize = v)));
                g.append(worldSlider("twinkle", w.starTwinkle, 0, 1, 0.02, (v) => (w.starTwinkle = v)));
                g.append(worldSlider("clustering", w.starClustering, 0, 1, 0.02, (v) => (w.starClustering = v)));
                g.append(worldSlider("color spread", w.starVariation, 0, 1, 0.02, (v) => (w.starVariation = v)));
                g.append(el("div", "hint", "base star color"));
                g.append(colorPicker(w.starColor, () => scene.touchWorld()));
                g.append(button("Regenerate", () => { scene.randomizeStars(); }));
            }));
        }
        else if (activeLab === "water") {
            box.append(toggle("enabled", w.waterEnabled, (v) => (w.waterEnabled = v)));
            box.append(colorPicker(w.waterColor, () => scene.touchWorld()));
            box.append(worldSlider("reflectivity", w.waterReflectivity, 0, 0.3, 0.005, (v) => (w.waterReflectivity = v)));
        }
        else if (activeLab === "render") {
            box.append(worldSlider("exposure", w.exposure, 0.2, 3, 0.02, (v) => (w.exposure = v)));
            box.append(worldSlider("warmth", w.warmth, 0, 1, 0.02, (v) => (w.warmth = v)));
            box.append(worldSlider("haze", w.hazeDensity, 0, 0.01, 0.0002, (v) => (w.hazeDensity = v)));
            box.append(group("Depth of field", (g) => {
                g.append(worldSlider("aperture", w.aperture, 0, 4, 0.02, (v) => (w.aperture = v)));
                g.append(worldSlider("focus dist", w.focusDistance, 2, 320, 0.5, (v) => (w.focusDistance = v)));
                const fb = el("div", "btns");
                fb.append(button(hooks.isFocusPick() ? "Click in view to focus… (cancel)" : "Focus on point", () => { hooks.onToggleFocusPick(); refresh(); }));
                g.append(fb);
                g.append(el("div", "hint", "Aperture 0 = all sharp; higher blurs everything off the focal plane."));
            }));
            box.append(group("Turntable", (g) => {
                g.append(plainSlider("seconds", turntable.seconds, 2, 30, 1, (v) => (turntable.seconds = Math.round(v))));
                g.append(plainSlider("fps", turntable.fps, 12, 60, 1, (v) => (turntable.fps = Math.round(v))));
                g.append(plainSlider("samples/frame", turntable.samples, 8, 256, 8, (v) => (turntable.samples = Math.round(v))));
                g.append(plainSlider("width", turntable.width, 480, 1920, 80, (v) => (turntable.width = Math.round(v))));
                const tb = el("div", "btns");
                tb.append(button("Render turntable (WebM)", () => hooks.onRenderTurntable({ ...turntable })));
                g.append(tb);
                g.append(el("div", "hint", `${turntable.seconds}s · ${turntable.seconds * turntable.fps} frames · orbits the target once. Higher samples = cleaner but slower.`));
            }));
        }
        labsPanel.append(box);
    }
    // --- top menu bar (Add / Tools / View / Scene / Labs + render toggle) ---
    function buildMenubar() {
        const tool = hooks.getTool();
        const gm = hooks.getGizmoMode();
        const snap = hooks.getSnap();
        const bm = hooks.getBookmarks();
        const addPrim = (t) => () => { selection = [scene.addPrim(t, getSpawn())]; hooks.onCommit(); };
        const addMenu = menu("add", "Add", (pop) => {
            pop.append(el("div", "sec", "Basic"));
            pop.append(menuItem(PRIM_LABELS[PrimType.Sphere], addPrim(PrimType.Sphere)));
            pop.append(menuItem(PRIM_LABELS[PrimType.Box], addPrim(PrimType.Box)));
            pop.append(menuItem(PRIM_LABELS[PrimType.RoundedBox], addPrim(PrimType.RoundedBox)));
            pop.append(el("div", "sec", "Tubular"));
            pop.append(menuItem(PRIM_LABELS[PrimType.Cylinder], addPrim(PrimType.Cylinder)));
            pop.append(menuItem(PRIM_LABELS[PrimType.Cone], addPrim(PrimType.Cone)));
            pop.append(menuItem(PRIM_LABELS[PrimType.Capsule], addPrim(PrimType.Capsule)));
            pop.append(menuItem(PRIM_LABELS[PrimType.Torus], addPrim(PrimType.Torus)));
            pop.append(el("div", "sec", "Polyhedra"));
            pop.append(menuItem(PRIM_LABELS[PrimType.Octahedron], addPrim(PrimType.Octahedron)));
            pop.append(menuItem(PRIM_LABELS[PrimType.Pyramid], addPrim(PrimType.Pyramid)));
            pop.append(el("div", "sec", "Landscape"));
            LANDSCAPE_TYPES.forEach((t) => pop.append(menuItem(LANDSCAPE_LABELS[t], () => hooks.onAddLandscape(t))));
            pop.append(el("div", "sec", "Light"));
            pop.append(menuItem("Sun", () => { selection = [scene.addLight(LightType.Directional)]; hooks.onCommit(); }));
            pop.append(menuItem("Point", () => { selection = [scene.addLight(LightType.Point, getSpawn())]; hooks.onCommit(); }));
            pop.append(menuItem("Planet", () => hooks.onAddPlanet()));
        });
        const toolsMenu = menu("tools", "Tools", (pop) => {
            pop.append(el("div", "sec", "Tool"));
            pop.append(menuItem("Move (1)", () => hooks.onSetTool("move"), { dot: tool === "move", keep: true }));
            pop.append(menuItem("Select (2)", () => hooks.onSetTool("select"), { dot: tool === "select", keep: true }));
            pop.append(el("div", "sec", "Gizmo"));
            pop.append(menuItem("Move (W)", () => hooks.onSetGizmoMode("translate"), { dot: gm === "translate", keep: true }));
            pop.append(menuItem("Rotate (E)", () => hooks.onSetGizmoMode("rotate"), { dot: gm === "rotate", keep: true }));
            pop.append(menuItem("Scale (R)", () => hooks.onSetGizmoMode("scale"), { dot: gm === "scale", keep: true }));
        });
        const viewMenu = menu("view", "View", (pop) => {
            pop.append(menuItem(`Carver ghosts: ${hooks.isGhostCarvers() ? "on" : "off"}`, () => hooks.onToggleGhostCarvers(), { dot: hooks.isGhostCarvers(), keep: true }));
            pop.append(menuItem(`Snap: ${snap.enabled ? snap.grid : "off"}`, () => hooks.onToggleSnap(), { dot: snap.enabled, keep: true }));
            if (snap.enabled)
                pop.append(plainSlider("grid", snap.grid, 0.25, 10, 0.25, (v) => hooks.onSetGrid(v)));
            pop.append(el("div", "menu-sep"));
            pop.append(el("div", "sec", "Save view"));
            const saveRow = el("div", "btns");
            for (let i = 0; i < bm.length; i++)
                saveRow.append(button(String(i + 1), () => { hooks.onSaveBookmark(i); refresh(); }));
            pop.append(saveRow);
            pop.append(el("div", "sec", "Go to view"));
            const goRow = el("div", "btns");
            for (let i = 0; i < bm.length; i++) {
                const b = button(`${i + 1}${bm[i] ? " •" : ""}`, () => hooks.onRecallBookmark(i));
                if (!bm[i])
                    b.disabled = true;
                goRow.append(b);
            }
            pop.append(goRow);
            pop.append(el("div", "menu-sep"));
            pop.append(menuItem("Save PNG…", () => hooks.onRenderImage()));
        });
        const sceneMenu = menu("scene", "Scene", (pop) => {
            pop.append(menuItem("Gallery…", () => hooks.onOpenGallery()));
            pop.append(menuItem("Save to gallery…", () => hooks.onSaveToGallery()));
            pop.append(menuItem("Import to gallery…", () => galleryFileInput.click()));
            pop.append(menuItem("New scene…", () => hooks.onNewScene()));
            pop.append(el("div", "menu-sep"));
            pop.append(menuItem("Save scene…", () => hooks.onSaveScene()));
            pop.append(menuItem("Load scene…", () => sceneFileInput.click()));
            pop.append(el("div", "menu-sep"));
            pop.append(menuItem("Import model…", () => modelFileInput.click()));
            const accept = acceptAttribute();
            pop.append(el("div", "hint", `${accept.replace(/\./g, "").replace(/,/g, ", ")} · drag-drop too`));
            if (scene.instances.length)
                pop.append(el("div", "hint", `${scene.instances.length} model(s) loaded`));
        });
        const labsMenu = menu("labs", "Labs", (pop) => {
            ["terrain", "sky", "water", "render"].forEach((t) => pop.append(menuItem(t[0].toUpperCase() + t.slice(1), () => { activeLab = activeLab === t ? null : t; }, { dot: activeLab === t, keep: true })));
            if (activeLab) {
                pop.append(el("div", "menu-sep"));
                pop.append(menuItem("Hide labs", () => { activeLab = null; }, { keep: true }));
            }
        });
        const helpMenu = menu("help", "?", (pop) => {
            pop.append(el("div", "sec", "Shortcuts"));
            pop.append(el("pre", "shortcuts", SHORTCUTS));
        }, "right");
        const renderBtn = el("button", "render-btn", hooks.isPreview() ? "■ Render" : "▶ Preview");
        renderBtn.addEventListener("click", () => { hooks.onToggleMode(); refresh(); });
        const csBtn = el("button", "menu-btn", "Cutscene");
        if (hooks.cutscene.active())
            csBtn.classList.add("active");
        csBtn.addEventListener("click", (e) => { e.stopPropagation(); hooks.cutscene.toggle(); });
        menubar.replaceChildren(el("div", "brand", "Aerie"), addMenu, toolsMenu, viewMenu, sceneMenu, labsMenu, csBtn, el("div", "menu-spacer"), renderBtn, helpMenu);
    }
    function refresh() {
        buildMenubar(); // reflect active tool / menu / mode state
        // Right panel: the inspector for the current selection.
        panel.replaceChildren();
        panel.append(el("div", "h", "Inspector"));
        panel.append(buildInspector());
        // Left-top panel: the object list.
        scenePanel.replaceChildren();
        scenePanel.append(el("div", "h", "Scene"));
        scenePanel.append(buildList());
        // Left-bottom panel: the active procedural lab (hidden when none).
        labsPanel.replaceChildren();
        if (activeLab) {
            labsPanel.style.display = "";
            buildLabs();
        }
        else {
            labsPanel.style.display = "none";
        }
        // Bottom-center: the material library dock + cutscene timeline.
        buildLibraryDock();
        buildCutsceneDock();
    }
    // Clicking anywhere outside an open menu dismisses it.
    document.addEventListener("click", () => {
        if (openMenu !== null) {
            openMenu = null;
            refresh();
        }
    });
    refresh();
    return {
        select(item) {
            selection = item ? [item] : [];
            refresh();
        },
        getSelected: () => primary(),
        getSelection: () => selection.slice(),
        setSelection(items) {
            selection = items.slice();
            refresh();
        },
        refresh,
        refreshCutscene: buildCutsceneDock,
    };
}
