import raytraceWGSL from "../shaders/raytrace.wgsl?raw";
import presentWGSL from "../shaders/present.wgsl?raw";
import { MAX_PRIMS, MAX_LIGHTS, MAX_CLOUD_LAYERS, LIGHT_FLOATS, LightType } from "../scene/scene";
import { buildTLAS } from "../mesh/bvh";
import { Vector3, Matrix4 } from "three";
// Uniform buffer is 10 rows of vec4 = 160 bytes. Keep in sync with the WGSL structs.
const UNIFORM_FLOATS = 40;
const UNIFORM_BYTES = UNIFORM_FLOATS * 4;
const PRIM_BYTES = MAX_PRIMS * 36 * 4;
const LIGHT_BYTES = MAX_LIGHTS * LIGHT_FLOATS * 4;
export class Renderer {
    device;
    context;
    accumBuffer;
    uniformBuffer;
    primBuffer;
    lightBuffer;
    triBuffer;
    bvhBuffer;
    matBuffer;
    instBuffer;
    tlasBuffer; // TLAS nodes (BVH over instance world AABBs)
    tlasOrderBuffer; // instance indices in TLAS leaf order
    meshTexture;
    meshNormalTex;
    primTexture;
    meshSampler;
    instCount = 0;
    triBase = []; // per-BLAS triangle offset into the global pool
    nodeBase = []; // per-BLAS node offset
    pickBuffer;
    pickRead;
    pickX = -1;
    pickY = -1;
    worldBuffer;
    exposure = 1.05;
    warmth = 0.5;
    aperture = 0;
    focusDistance = 60;
    // 6 base + 3 star + 3 terrain-color + 3 recipe vec4 + MAX_CLOUD_LAYERS × 3 vec4.
    worldData = new Float32Array((15 + MAX_CLOUD_LAYERS * 3) * 4);
    computePipeline;
    presentPipeline;
    presentModule;
    computeBind;
    presentBind;
    uniformData = new Float32Array(UNIFORM_FLOATS);
    /** Accumulated samples for the current camera pose. */
    frame = 0;
    width = 1;
    height = 1;
    sunDir = new Vector3(0.5, 0.55, 0.3).normalize();
    sunColorV = new Vector3(1, 0.96, 0.9);
    starsFlag = 0;
    sunFlag = 1; // 1 when a directional light (sun) exists, else 0
    sunBright = 3; // sun-disk brightness, driven by the sun light's intensity
    primCount = 0;
    lightCount = 0;
    startTime = performance.now();
    constructor(device, context, format) {
        this.device = device;
        this.context = context;
        this.uniformBuffer = device.createBuffer({
            size: UNIFORM_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.primBuffer = device.createBuffer({
            size: PRIM_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.lightBuffer = device.createBuffer({
            size: LIGHT_BYTES,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // Mesh buffers start as minimal placeholders (always bound, even with no mesh).
        const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        this.triBuffer = device.createBuffer({ size: 128, usage: storage });
        this.bvhBuffer = device.createBuffer({ size: 32, usage: storage });
        this.matBuffer = device.createBuffer({ size: 48, usage: storage });
        this.instBuffer = device.createBuffer({ size: 80, usage: storage });
        this.tlasBuffer = device.createBuffer({ size: 32, usage: storage }); // 1 placeholder node
        this.tlasOrderBuffer = device.createBuffer({ size: 4, usage: storage });
        this.meshTexture = this.makeTexture(1, 1);
        this.meshNormalTex = this.makeTexture(1, 1);
        this.primTexture = this.makeTexture(1, 1);
        this.meshSampler = device.createSampler({
            magFilter: "linear",
            minFilter: "linear",
            addressModeU: "repeat",
            addressModeV: "repeat",
        });
        this.pickBuffer = device.createBuffer({
            size: 16, // 4 × i32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this.pickRead = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.worldBuffer = device.createBuffer({
            size: (15 + MAX_CLOUD_LAYERS * 3) * 16, // 6 base + 3 star + 3 terrain-color + 3 recipe + 4 clouds × 3
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const computeModule = device.createShaderModule({ code: raytraceWGSL });
        this.computePipeline = device.createComputePipeline({
            layout: "auto",
            compute: { module: computeModule, entryPoint: "main" },
        });
        const presentModule = device.createShaderModule({ code: presentWGSL });
        this.presentModule = presentModule;
        this.presentPipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: { module: presentModule, entryPoint: "vs" },
            fragment: {
                module: presentModule,
                entryPoint: "fs",
                targets: [{ format }],
            },
            primitive: { topology: "triangle-list" },
        });
    }
    /** (Re)allocate the accumulation buffer and bind groups for a new size. */
    resize(width, height) {
        this.width = Math.max(1, width);
        this.height = Math.max(1, height);
        this.accumBuffer?.destroy();
        this.accumBuffer = this.device.createBuffer({
            size: this.width * this.height * 4 * 4, // vec4<f32> per pixel
            usage: GPUBufferUsage.STORAGE,
        });
        this.rebuildComputeBind();
        this.presentBind = this.device.createBindGroup({
            layout: this.presentPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: this.accumBuffer } },
            ],
        });
        this.frame = 0;
    }
    makeTexture(size, layers) {
        return this.device.createTexture({
            size: { width: size, height: size, depthOrArrayLayers: layers },
            format: "rgba8unorm",
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
    }
    computeEntries(accum) {
        return [
            { binding: 0, resource: { buffer: this.uniformBuffer } },
            { binding: 1, resource: { buffer: accum } },
            { binding: 2, resource: { buffer: this.primBuffer } },
            { binding: 3, resource: { buffer: this.lightBuffer } },
            { binding: 4, resource: { buffer: this.triBuffer } },
            { binding: 5, resource: { buffer: this.bvhBuffer } },
            { binding: 6, resource: { buffer: this.matBuffer } },
            { binding: 7, resource: this.meshTexture.createView({ dimension: "2d-array" }) },
            { binding: 8, resource: this.meshSampler },
            { binding: 9, resource: { buffer: this.pickBuffer } },
            { binding: 10, resource: { buffer: this.instBuffer } },
            { binding: 11, resource: this.meshNormalTex.createView({ dimension: "2d-array" }) },
            { binding: 12, resource: { buffer: this.worldBuffer } },
            { binding: 13, resource: this.primTexture.createView({ dimension: "2d-array" }) },
            { binding: 14, resource: { buffer: this.tlasBuffer } },
            { binding: 15, resource: { buffer: this.tlasOrderBuffer } },
        ];
    }
    rebuildComputeBind() {
        this.computeBind = this.device.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: this.computeEntries(this.accumBuffer),
        });
    }
    /** Offline high-resolution render to a PNG blob (no UI/overlays). */
    /** Render one fully-accumulated frame off-screen at an arbitrary size and
     *  return its RGBA pixels (used by PNG export and the turntable recorder). */
    async renderToPixels(cam, w, h, samples) {
        const dev = this.device;
        const accum = dev.createBuffer({ size: w * h * 16, usage: GPUBufferUsage.STORAGE });
        const tex = dev.createTexture({
            size: { width: w, height: h },
            format: "rgba8unorm",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        });
        const presentPipe = dev.createRenderPipeline({
            layout: "auto",
            vertex: { module: this.presentModule, entryPoint: "vs" },
            fragment: { module: this.presentModule, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
            primitive: { topology: "triangle-list" },
        });
        const computeBind = dev.createBindGroup({
            layout: this.computePipeline.getBindGroupLayout(0),
            entries: this.computeEntries(accum),
        });
        const presentBind = dev.createBindGroup({
            layout: presentPipe.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } },
                { binding: 1, resource: { buffer: accum } },
            ],
        });
        // Render at the requested size by temporarily overriding the dimensions.
        const sw = this.width, sh = this.height, sf = this.frame;
        this.width = w;
        this.height = h;
        for (let i = 0; i < samples; i++) {
            this.frame = i;
            this.writeUniforms(cam); // queue-ordered with the dispatch below
            const enc = dev.createCommandEncoder();
            const cp = enc.beginComputePass();
            cp.setPipeline(this.computePipeline);
            cp.setBindGroup(0, computeBind);
            cp.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
            cp.end();
            dev.queue.submit([enc.finish()]);
        }
        const bytesPerRow = Math.ceil((w * 4) / 256) * 256;
        const readBuf = dev.createBuffer({
            size: bytesPerRow * h,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        this.frame = samples - 1; // sampleCount = samples for the present average
        this.writeUniforms(cam);
        const enc = dev.createCommandEncoder();
        const rp = enc.beginRenderPass({
            colorAttachments: [
                { view: tex.createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
            ],
        });
        rp.setPipeline(presentPipe);
        rp.setBindGroup(0, presentBind);
        rp.draw(3);
        rp.end();
        enc.copyTextureToBuffer({ texture: tex }, { buffer: readBuf, bytesPerRow, rowsPerImage: h }, { width: w, height: h });
        dev.queue.submit([enc.finish()]);
        await readBuf.mapAsync(GPUMapMode.READ);
        const src = new Uint8Array(readBuf.getMappedRange());
        const img = new Uint8ClampedArray(w * h * 4);
        for (let y = 0; y < h; y++)
            img.set(src.subarray(y * bytesPerRow, y * bytesPerRow + w * 4), y * w * 4);
        readBuf.unmap();
        accum.destroy();
        tex.destroy();
        readBuf.destroy();
        this.width = sw;
        this.height = sh;
        this.frame = sf;
        this.resetAccumulation();
        return img;
    }
    /** Render one accumulated frame and encode it as a PNG blob. */
    async renderToImage(cam, w, h, samples) {
        const img = await this.renderToPixels(cam, w, h, samples);
        const cnv = document.createElement("canvas");
        cnv.width = w;
        cnv.height = h;
        cnv.getContext("2d").putImageData(new ImageData(img, w, h), 0, 0);
        return await new Promise((res, rej) => cnv.toBlob((b) => (b ? res(b) : rej(new Error("toBlob failed"))), "image/png"));
    }
    /** Upload the primitive image-texture array (triplanar-mapped patterns). */
    uploadPrimTextures(scene) {
        this.primTexture = this.uploadTexArray(scene.primImageLayers, scene.primImageSize, this.primTexture);
        this.rebuildComputeBind();
        this.resetAccumulation();
    }
    /** Upload the procedural-world (terrain/sky/water/render) parameters. */
    uploadWorld(scene) {
        const w = scene.world;
        const d = this.worldData;
        const layers = w.cloudLayers.slice(0, MAX_CLOUD_LAYERS);
        d[0] = w.terrainAmp;
        d[1] = w.terrainFreq;
        d[2] = w.terrainRidge;
        d[3] = w.terrainOffset;
        d[4] = w.terrainEnabled ? 1 : 0;
        d[5] = w.waterEnabled ? 1 : 0;
        d[6] = w.cloudsEnabled && layers.length > 0 ? 1 : 0;
        d[7] = layers.length;
        d[8] = w.zenith[0];
        d[9] = w.zenith[1];
        d[10] = w.zenith[2];
        d[11] = 6000 + (300 - 6000) * w.sunSize; // sun disk exponent (bigger size → smaller exp)
        d[12] = w.horizon[0];
        d[13] = w.horizon[1];
        d[14] = w.horizon[2];
        d[15] = 0;
        d[16] = w.hazeDensity;
        d[17] = w.terrainOctaves;
        d[18] = w.terrainWarp;
        d[19] = w.terrainSeed;
        d[20] = w.waterColor[0];
        d[21] = w.waterColor[1];
        d[22] = w.waterColor[2];
        d[23] = w.waterReflectivity;
        // star0: density, brightness, size, twinkle
        d[24] = w.starDensity;
        d[25] = w.starBrightness;
        d[26] = w.starSize;
        d[27] = w.starTwinkle;
        // star1: color.rgb, variation
        d[28] = w.starColor[0];
        d[29] = w.starColor[1];
        d[30] = w.starColor[2];
        d[31] = w.starVariation;
        // star2: clustering, seed
        d[32] = w.starClustering;
        d[33] = w.starSeed;
        d[34] = 0;
        d[35] = 0;
        // terrain palette: low / rock / high
        d[36] = w.terrainLow[0];
        d[37] = w.terrainLow[1];
        d[38] = w.terrainLow[2];
        d[39] = 0;
        d[40] = w.terrainRock[0];
        d[41] = w.terrainRock[1];
        d[42] = w.terrainRock[2];
        d[43] = 0;
        d[44] = w.terrainHigh[0];
        d[45] = w.terrainHigh[1];
        d[46] = w.terrainHigh[2];
        d[47] = 0;
        // recipe block: rec0 (layer bases/fractals), rec1 (layer-1 + terrace), rec2 (terrace/warp/palette thresholds)
        d[48] = w.terrainBasis;
        d[49] = w.terrainFractal;
        d[50] = w.terrainBasis2;
        d[51] = w.terrainFractal2;
        d[52] = w.terrainFreq2;
        d[53] = w.terrainOctaves2;
        d[54] = w.terrainWeight2;
        d[55] = w.terrainTerraceSteps;
        d[56] = w.terrainTerraceSharp;
        d[57] = w.terrainWarpFreq;
        d[58] = w.terrainSnowLine;
        d[59] = w.terrainSlopeRock;
        // Cloud layers: each is 3 vec4 — (type, coverage, density, scale),
        // (altitude, seed, thickness, darkness), (color.rgb, _).
        for (let i = 0; i < MAX_CLOUD_LAYERS; i++) {
            const o = 60 + i * 12;
            const L = layers[i];
            if (L) {
                d[o] = L.type;
                d[o + 1] = L.coverage;
                d[o + 2] = L.density;
                d[o + 3] = L.scale;
                d[o + 4] = L.altitude;
                d[o + 5] = L.seed;
                d[o + 6] = L.thickness;
                d[o + 7] = L.darkness;
                d[o + 8] = L.color[0];
                d[o + 9] = L.color[1];
                d[o + 10] = L.color[2];
                d[o + 11] = 0;
            }
            else {
                for (let k = 0; k < 12; k++)
                    d[o + k] = 0;
                d[o + 3] = 1;
            }
        }
        this.device.queue.writeBuffer(this.worldBuffer, 0, d);
        this.exposure = w.exposure;
        this.warmth = w.warmth;
        this.aperture = w.aperture;
        this.focusDistance = w.focusDistance;
        this.starsFlag = w.starsEnabled ? 1 : 0;
        this.resetAccumulation();
    }
    uploadTexArray(layers, size, old) {
        old.destroy();
        if (layers.length === 0)
            return this.makeTexture(1, 1);
        const tex = this.makeTexture(size, layers.length);
        for (let l = 0; l < layers.length; l++) {
            this.device.queue.writeTexture({ texture: tex, origin: { x: 0, y: 0, z: l } }, layers[l], { bytesPerRow: size * 4, rowsPerImage: size }, { width: size, height: size, depthOrArrayLayers: 1 });
        }
        return tex;
    }
    /** Trace one ray through a pixel and return the hit { kind, index }.
     *  kind: 0 miss, 1 terrain, 2 water, 4 primitive, 5 mesh. */
    async pick(cam, px, py) {
        this.pickX = px;
        this.pickY = py;
        this.writeUniforms(cam);
        this.pickX = -1;
        this.pickY = -1;
        const encoder = this.device.createCommandEncoder();
        const cpass = encoder.beginComputePass();
        cpass.setPipeline(this.computePipeline);
        cpass.setBindGroup(0, this.computeBind);
        cpass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        cpass.end();
        encoder.copyBufferToBuffer(this.pickBuffer, 0, this.pickRead, 0, 16);
        this.device.queue.submit([encoder.finish()]);
        await this.pickRead.mapAsync(GPUMapMode.READ);
        const buf = this.pickRead.getMappedRange().slice(0);
        const ri = new Int32Array(buf);
        const rf = new Float32Array(buf); // pickResult[2] is a bit-cast f32 (focal distance)
        this.pickRead.unmap();
        return { kind: ri[0], index: ri[1], dist: rf[2] };
    }
    /** Reset accumulation — call whenever the camera or scene changes. */
    resetAccumulation() {
        this.frame = 0;
    }
    /** Upload the global mesh pools: concatenated BLAS triangles + nodes,
     *  materials, and the base-color texture array. Call on import/remove. */
    uploadMeshPools(scene) {
        const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        const blases = scene.blases;
        // Compute per-BLAS offsets and total sizes.
        this.triBase = [];
        this.nodeBase = [];
        let triFloats = 0;
        let nodeFloats = 0;
        for (const b of blases) {
            this.triBase.push(triFloats / 32);
            this.nodeBase.push(nodeFloats / 8);
            triFloats += b.tris.length;
            nodeFloats += b.nodes.length;
        }
        const triData = new Float32Array(Math.max(32, triFloats));
        const nodeData = new Float32Array(Math.max(8, nodeFloats));
        let to = 0;
        let no = 0;
        for (const b of blases) {
            triData.set(b.tris, to);
            to += b.tris.length;
            nodeData.set(b.nodes, no);
            no += b.nodes.length;
        }
        this.triBuffer.destroy();
        this.triBuffer = this.device.createBuffer({ size: triData.byteLength, usage: storage });
        this.device.queue.writeBuffer(this.triBuffer, 0, triData);
        this.bvhBuffer.destroy();
        this.bvhBuffer = this.device.createBuffer({ size: nodeData.byteLength, usage: storage });
        this.device.queue.writeBuffer(this.bvhBuffer, 0, nodeData);
        const matData = new Float32Array(Math.max(12, scene.meshMaterials.length));
        matData.set(scene.meshMaterials);
        this.matBuffer.destroy();
        this.matBuffer = this.device.createBuffer({ size: matData.byteLength, usage: storage });
        this.device.queue.writeBuffer(this.matBuffer, 0, matData);
        const size = scene.meshTexSize;
        this.meshTexture = this.uploadTexArray(scene.meshTexLayers, size, this.meshTexture);
        this.meshNormalTex = this.uploadTexArray(scene.meshNormalLayers, size, this.meshNormalTex);
        this.uploadInstances(scene); // also rebuilds the bind group
    }
    /** Rewrite just the material buffer (cheap — call on a PBR override edit). */
    uploadMeshMaterials(scene) {
        const matData = new Float32Array(Math.max(12, scene.meshMaterials.length));
        matData.set(scene.meshMaterials);
        if (matData.byteLength <= this.matBuffer.size) {
            this.device.queue.writeBuffer(this.matBuffer, 0, matData);
            this.resetAccumulation();
        }
        else {
            this.uploadMeshPools(scene); // material count grew — full rebuild
        }
    }
    /** Upload instance transforms only (cheap — call on add/remove/drag). Also
     *  rebuilds the TLAS (a BVH over instance world AABBs) so ray/shadow queries
     *  scale with log(instances) rather than linearly. */
    uploadInstances(scene) {
        const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
        const insts = scene.instances;
        const data = new Float32Array(Math.max(20, insts.length * 20));
        const mins = new Float32Array(Math.max(3, insts.length * 3));
        const maxs = new Float32Array(Math.max(3, insts.length * 3));
        const model = new Matrix4();
        const corner = new Vector3();
        for (let i = 0; i < insts.length; i++) {
            const inst = insts[i];
            const o = i * 20;
            inst.writeInvModel(data, o);
            data[o + 16] = this.nodeBase[inst.blasIndex] ?? 0;
            data[o + 17] = this.triBase[inst.blasIndex] ?? 0;
            // World AABB: the BLAS local-root AABB (nodes[0..6]) transformed by the
            // model matrix. Transforming the 8 corners bounds any rotation/scale.
            const n = scene.blases[inst.blasIndex].nodes;
            const lx = n[0], ly = n[1], lz = n[2], hx = n[4], hy = n[5], hz = n[6];
            inst.modelMatrix(model);
            let miX = Infinity, miY = Infinity, miZ = Infinity;
            let maX = -Infinity, maY = -Infinity, maZ = -Infinity;
            for (let c = 0; c < 8; c++) {
                corner.set(c & 1 ? hx : lx, c & 2 ? hy : ly, c & 4 ? hz : lz).applyMatrix4(model);
                miX = Math.min(miX, corner.x);
                miY = Math.min(miY, corner.y);
                miZ = Math.min(miZ, corner.z);
                maX = Math.max(maX, corner.x);
                maY = Math.max(maY, corner.y);
                maZ = Math.max(maZ, corner.z);
            }
            mins[i * 3] = miX;
            mins[i * 3 + 1] = miY;
            mins[i * 3 + 2] = miZ;
            maxs[i * 3] = maX;
            maxs[i * 3 + 1] = maY;
            maxs[i * 3 + 2] = maZ;
        }
        if (data.byteLength > this.instBuffer.size) {
            this.instBuffer.destroy();
            this.instBuffer = this.device.createBuffer({ size: data.byteLength, usage: storage });
        }
        this.device.queue.writeBuffer(this.instBuffer, 0, data);
        this.instCount = insts.length;
        // Build + upload the TLAS. Buffers are always non-empty (the shader guards
        // traversal on instCount, so the placeholder tree is never walked).
        const tlas = buildTLAS(mins, maxs, insts.length);
        if (tlas.nodes.byteLength > this.tlasBuffer.size) {
            this.tlasBuffer.destroy();
            this.tlasBuffer = this.device.createBuffer({ size: tlas.nodes.byteLength, usage: storage });
        }
        this.device.queue.writeBuffer(this.tlasBuffer, 0, tlas.nodes);
        if (tlas.order.byteLength > this.tlasOrderBuffer.size) {
            this.tlasOrderBuffer.destroy();
            this.tlasOrderBuffer = this.device.createBuffer({ size: tlas.order.byteLength, usage: storage });
        }
        this.device.queue.writeBuffer(this.tlasOrderBuffer, 0, tlas.order);
        this.rebuildComputeBind();
        this.resetAccumulation();
    }
    /** Push the current scene (primitives + lights) to the GPU. */
    uploadScene(scene) {
        scene.pack();
        this.device.queue.writeBuffer(this.primBuffer, 0, scene.primData);
        this.device.queue.writeBuffer(this.lightBuffer, 0, scene.lightData);
        this.primCount = scene.prims.length;
        this.lightCount = scene.lights.length;
        const sun = scene.lights.find((l) => l.type === LightType.Directional);
        this.sunFlag = sun ? 1 : 0;
        this.sunBright = sun ? sun.intensity : 0;
        this.sunDir.copy(scene.sunDir);
        const sc = scene.sunColor;
        this.sunColorV.set(sc[0], sc[1], sc[2]);
        this.resetAccumulation();
    }
    writeUniforms(cam) {
        const d = this.uniformData;
        const t = (performance.now() - this.startTime) / 1000;
        // row 0: camPos.xyz, fovScale
        d[0] = cam.position.x;
        d[1] = cam.position.y;
        d[2] = cam.position.z;
        d[3] = cam.fovScale;
        // row 1: forward.xyz, aspect
        d[4] = cam.forward.x;
        d[5] = cam.forward.y;
        d[6] = cam.forward.z;
        d[7] = this.width / this.height;
        // row 2: right.xyz, frame
        d[8] = cam.right.x;
        d[9] = cam.right.y;
        d[10] = cam.right.z;
        d[11] = this.frame;
        // row 3: up.xyz, seed
        d[12] = cam.up.x;
        d[13] = cam.up.y;
        d[14] = cam.up.z;
        d[15] = Math.random();
        // row 4: sunDir.xyz, time
        d[16] = this.sunDir.x;
        d[17] = this.sunDir.y;
        d[18] = this.sunDir.z;
        d[19] = t;
        // row 5: res.xy, sampleCount, pad
        d[20] = this.width;
        d[21] = this.height;
        d[22] = this.frame + 1;
        d[23] = this.sunBright;
        // row 6: primCount, lightCount, instCount, sunFlag
        d[24] = this.primCount;
        d[25] = this.lightCount;
        d[26] = this.instCount;
        d[27] = this.sunFlag;
        // row 7: pickX, pickY, exposure, warmth
        d[28] = this.pickX;
        d[29] = this.pickY;
        d[30] = this.exposure;
        d[31] = this.warmth;
        // row 8: sunColor, starsFlag
        d[32] = this.sunColorV.x;
        d[33] = this.sunColorV.y;
        d[34] = this.sunColorV.z;
        d[35] = this.starsFlag;
        // row 9: aperture, focusDistance, pad, pad (thin-lens depth of field)
        d[36] = this.aperture;
        d[37] = this.focusDistance;
        d[38] = 0;
        d[39] = 0;
        this.device.queue.writeBuffer(this.uniformBuffer, 0, d);
    }
    /** Trace one new sample per pixel and present the running average. */
    renderSample(cam) {
        this.writeUniforms(cam);
        const encoder = this.device.createCommandEncoder();
        const cpass = encoder.beginComputePass();
        cpass.setPipeline(this.computePipeline);
        cpass.setBindGroup(0, this.computeBind);
        cpass.dispatchWorkgroups(Math.ceil(this.width / 8), Math.ceil(this.height / 8));
        cpass.end();
        const rpass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: this.context.getCurrentTexture().createView(),
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });
        rpass.setPipeline(this.presentPipeline);
        rpass.setBindGroup(0, this.presentBind);
        rpass.draw(3);
        rpass.end();
        this.device.queue.submit([encoder.finish()]);
        this.frame++;
    }
}
