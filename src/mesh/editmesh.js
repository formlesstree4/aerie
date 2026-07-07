import { buildBVH } from "./bvh";
import { computeFaceTangents, TRI_STRIDE } from "./modelImport";
export function buildEditMesh(blas) {
    const tris = blas.tris;
    const triCount = blas.triCount;
    const verts = [];
    const map = new Map();
    const cornerVert = new Uint32Array(triCount * 3);
    const q = (n) => Math.round(n * 1e4) / 1e4; // weld tolerance
    for (let t = 0; t < triCount; t++) {
        for (let v = 0; v < 3; v++) {
            const o = t * TRI_STRIDE + v * 4;
            const x = tris[o], y = tris[o + 1], z = tris[o + 2];
            const key = `${q(x)},${q(y)},${q(z)}`;
            let idx = map.get(key);
            if (idx === undefined) {
                idx = verts.length / 3;
                verts.push(x, y, z);
                map.set(key, idx);
            }
            cornerVert[t * 3 + v] = idx;
        }
    }
    return { verts: new Float32Array(verts), cornerVert, triCount, src: tris.slice() };
}
/** Rebuild a BLAS from edited vertices: smooth normals, preserved UVs/material,
 *  recomputed tangents, fresh BVH. */
export function rebuildBlas(edit) {
    const { verts, cornerVert, triCount, src } = edit;
    const nVerts = verts.length / 3;
    // Area-weighted smooth vertex normals.
    const vn = new Float32Array(nVerts * 3);
    for (let t = 0; t < triCount; t++) {
        const i0 = cornerVert[3 * t], i1 = cornerVert[3 * t + 1], i2 = cornerVert[3 * t + 2];
        const ax = verts[3 * i0], ay = verts[3 * i0 + 1], az = verts[3 * i0 + 2];
        const e1x = verts[3 * i1] - ax, e1y = verts[3 * i1 + 1] - ay, e1z = verts[3 * i1 + 2] - az;
        const e2x = verts[3 * i2] - ax, e2y = verts[3 * i2 + 1] - ay, e2z = verts[3 * i2 + 2] - az;
        const nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
        for (const i of [i0, i1, i2]) {
            vn[3 * i] += nx;
            vn[3 * i + 1] += ny;
            vn[3 * i + 2] += nz;
        }
    }
    for (let i = 0; i < nVerts; i++) {
        const x = vn[3 * i], y = vn[3 * i + 1], z = vn[3 * i + 2];
        const len = Math.hypot(x, y, z) || 1;
        vn[3 * i] = x / len;
        vn[3 * i + 1] = y / len;
        vn[3 * i + 2] = z / len;
    }
    const out = new Float32Array(triCount * TRI_STRIDE);
    for (let t = 0; t < triCount; t++) {
        const base = t * TRI_STRIDE;
        for (let v = 0; v < 3; v++) {
            const i = cornerVert[3 * t + v];
            const po = base + v * 4;
            out[po] = verts[3 * i];
            out[po + 1] = verts[3 * i + 1];
            out[po + 2] = verts[3 * i + 2];
            const no = base + 12 + v * 4;
            out[no] = vn[3 * i];
            out[no + 1] = vn[3 * i + 1];
            out[no + 2] = vn[3 * i + 2];
        }
        for (let k = 24; k < 32; k++)
            out[base + k] = src[base + k]; // uv01, uv2m (uv + matId + vcolor2)
        out[base + 19] = src[base + 19]; // preserve packed vertex colors 0/1 (n1.w, n2.w)
        out[base + 23] = src[base + 23];
    }
    computeFaceTangents(out);
    const { nodes, tris, nodeCount } = buildBVH(out, TRI_STRIDE);
    return { tris, triCount: tris.length / TRI_STRIDE, nodes, nodeCount };
}
