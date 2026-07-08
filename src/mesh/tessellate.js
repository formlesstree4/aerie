import { BufferGeometry, Float32BufferAttribute, Uint32BufferAttribute, Mesh, MeshStandardMaterial, Group, Matrix4, Quaternion, Vector3, } from "three";
import { PrimType } from "../scene/scene";
import { buildBVH } from "./bvh";
import { computeFaceTangents, TRI_STRIDE } from "./modelImport";
class Builder {
    pos = [];
    nor = [];
    idx = [];
    vert(p, n) {
        this.pos.push(p[0], p[1], p[2]);
        this.nor.push(n[0], n[1], n[2]);
        return this.pos.length / 3 - 1;
    }
    tri(a, b, c) {
        this.idx.push(a, b, c);
    }
    quad(a, b, c, d) {
        this.tri(a, b, c);
        this.tri(a, c, d);
    }
    /** A grid of (uSeg+1)×(vSeg+1) vertices from a parametric surface, stitched
     *  into quads. Seams are duplicated; the edit mesh welds them by position. */
    grid(uSeg, vSeg, fn) {
        const base = this.pos.length / 3;
        const rw = vSeg + 1;
        for (let i = 0; i <= uSeg; i++) {
            for (let j = 0; j <= vSeg; j++) {
                const { p, n } = fn(i, j);
                this.vert(p, n);
            }
        }
        for (let i = 0; i < uSeg; i++) {
            for (let j = 0; j < vSeg; j++) {
                const a = base + i * rw + j;
                this.quad(a, a + rw, a + rw + 1, a + 1);
            }
        }
    }
    /** A triangle fan around `center` (with normal `n`) over a ring of points. */
    fan(center, n, ring, flip) {
        const c = this.vert(center, n);
        const start = this.pos.length / 3;
        for (const p of ring)
            this.vert(p, n);
        for (let i = 0; i < ring.length; i++) {
            const a = start + i;
            const b = start + ((i + 1) % ring.length);
            if (flip)
                this.tri(c, b, a);
            else
                this.tri(c, a, b);
        }
    }
}
const norm = (x, y, z) => {
    const l = Math.hypot(x, y, z) || 1;
    return [x / l, y / l, z / l];
};
const SEG = 28; // radial segments
const RINGS = 16; // latitude / tube segments
function sphereMesh(r) {
    const b = new Builder();
    b.grid(SEG, RINGS, (i, j) => {
        const phi = (i / SEG) * Math.PI * 2;
        const theta = (j / RINGS) * Math.PI;
        const st = Math.sin(theta);
        const n = [st * Math.cos(phi), Math.cos(theta), st * Math.sin(phi)];
        return { p: [n[0] * r, n[1] * r, n[2] * r], n };
    });
    return b;
}
function boxMesh(a, bb, c) {
    const b = new Builder();
    const faces = [
        [[1, 0, 0], [a, -bb, -c], [a, bb, -c], [a, bb, c], [a, -bb, c]],
        [[-1, 0, 0], [-a, -bb, c], [-a, bb, c], [-a, bb, -c], [-a, -bb, -c]],
        [[0, 1, 0], [-a, bb, -c], [-a, bb, c], [a, bb, c], [a, bb, -c]],
        [[0, -1, 0], [-a, -bb, c], [-a, -bb, -c], [a, -bb, -c], [a, -bb, c]],
        [[0, 0, 1], [-a, -bb, c], [a, -bb, c], [a, bb, c], [-a, bb, c]],
        [[0, 0, -1], [a, -bb, -c], [-a, -bb, -c], [-a, bb, -c], [a, bb, -c]],
    ];
    for (const [n, p0, p1, p2, p3] of faces) {
        const i0 = b.vert(p0, n), i1 = b.vert(p1, n), i2 = b.vert(p2, n), i3 = b.vert(p3, n);
        b.quad(i0, i1, i2, i3);
    }
    return b;
}
function torusMesh(R, r) {
    const b = new Builder();
    b.grid(SEG, RINGS, (i, j) => {
        const phi = (i / SEG) * Math.PI * 2;
        const theta = (j / RINGS) * Math.PI * 2;
        const ct = Math.cos(theta);
        const n = [ct * Math.cos(phi), Math.sin(theta), ct * Math.sin(phi)];
        const p = [(R + r * ct) * Math.cos(phi), r * Math.sin(theta), (R + r * ct) * Math.sin(phi)];
        return { p, n };
    });
    return b;
}
function cylinderMesh(r, h) {
    const b = new Builder();
    b.grid(SEG, 1, (i, j) => {
        const phi = (i / SEG) * Math.PI * 2;
        const n = [Math.cos(phi), 0, Math.sin(phi)];
        return { p: [r * Math.cos(phi), -h + j * 2 * h, r * Math.sin(phi)], n };
    });
    const top = [], bot = [];
    for (let i = 0; i < SEG; i++) {
        const phi = (i / SEG) * Math.PI * 2;
        top.push([r * Math.cos(phi), h, r * Math.sin(phi)]);
        bot.push([r * Math.cos(phi), -h, r * Math.sin(phi)]);
    }
    b.fan([0, h, 0], [0, 1, 0], top, false);
    b.fan([0, -h, 0], [0, -1, 0], bot, true);
    return b;
}
function coneMesh(r1, h) {
    const r2 = 0.05; // matches the SDF's capped tip
    const b = new Builder();
    const nr = 2 * h, ny = r1 - r2; // outward slope normal components
    b.grid(SEG, 1, (i, j) => {
        const phi = (i / SEG) * Math.PI * 2;
        const rad = r1 + (r2 - r1) * j;
        const n = norm(nr * Math.cos(phi), ny, nr * Math.sin(phi));
        return { p: [rad * Math.cos(phi), -h + j * 2 * h, rad * Math.sin(phi)], n };
    });
    const bot = [];
    for (let i = 0; i < SEG; i++) {
        const phi = (i / SEG) * Math.PI * 2;
        bot.push([r1 * Math.cos(phi), -h, r1 * Math.sin(phi)]);
    }
    b.fan([0, -h, 0], [0, -1, 0], bot, true);
    return b;
}
function capsuleMesh(r, h) {
    const b = new Builder();
    // Cylindrical shaft.
    b.grid(SEG, 1, (i, j) => {
        const phi = (i / SEG) * Math.PI * 2;
        const n = [Math.cos(phi), 0, Math.sin(phi)];
        return { p: [r * Math.cos(phi), -h + j * 2 * h, r * Math.sin(phi)], n };
    });
    const half = Math.max(2, Math.floor(RINGS / 2));
    // Top hemisphere (theta 0 → π/2), centered at +h.
    b.grid(SEG, half, (i, j) => {
        const phi = (i / SEG) * Math.PI * 2;
        const theta = (j / half) * (Math.PI / 2);
        const st = Math.sin(theta);
        const n = [st * Math.cos(phi), Math.cos(theta), st * Math.sin(phi)];
        return { p: [r * n[0], h + r * n[1], r * n[2]], n };
    });
    // Bottom hemisphere (theta π/2 → π), centered at -h.
    b.grid(SEG, half, (i, j) => {
        const phi = (i / SEG) * Math.PI * 2;
        const theta = Math.PI / 2 + (j / half) * (Math.PI / 2);
        const st = Math.sin(theta);
        const n = [st * Math.cos(phi), Math.cos(theta), st * Math.sin(phi)];
        return { p: [r * n[0], -h + r * n[1], r * n[2]], n };
    });
    return b;
}
function pyramidMesh(a, h) {
    const b = new Builder();
    const apex = [0, h, 0];
    const base = [[-a, -h, -a], [a, -h, -a], [a, -h, a], [-a, -h, a]];
    for (let i = 0; i < 4; i++) {
        const p1 = base[i], p2 = base[(i + 1) % 4];
        const n = norm((p1[1] - apex[1]) * (p2[2] - apex[2]) - (p1[2] - apex[2]) * (p2[1] - apex[1]), (p1[2] - apex[2]) * (p2[0] - apex[0]) - (p1[0] - apex[0]) * (p2[2] - apex[2]), (p1[0] - apex[0]) * (p2[1] - apex[1]) - (p1[1] - apex[1]) * (p2[0] - apex[0]));
        b.tri(b.vert(apex, n), b.vert(p1, n), b.vert(p2, n));
    }
    const dn = [0, -1, 0];
    b.quad(b.vert(base[3], dn), b.vert(base[2], dn), b.vert(base[1], dn), b.vert(base[0], dn));
    return b;
}
function octahedronMesh(s) {
    const b = new Builder();
    const v = [[s, 0, 0], [-s, 0, 0], [0, s, 0], [0, -s, 0], [0, 0, s], [0, 0, -s]];
    const faces = [
        [0, 2, 4], [4, 2, 1], [1, 2, 5], [5, 2, 0],
        [0, 4, 3], [4, 1, 3], [1, 5, 3], [5, 0, 3],
    ];
    for (const [a, c, d] of faces) {
        const n = norm(v[a][0] + v[c][0] + v[d][0], v[a][1] + v[c][1] + v[d][1], v[a][2] + v[c][2] + v[d][2]);
        b.tri(b.vert(v[a], n), b.vert(v[c], n), b.vert(v[d], n));
    }
    return b;
}
/** A primitive's tessellated surface as a LOCAL-space BufferGeometry — the caller
 *  applies the primitive's transform on the mesh, so it can move/rotate without
 *  re-tessellating (used by the raster preview). */
export function primitiveLocalGeometry(prim) {
    const m = meshFor(prim);
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(m.pos, 3));
    g.setAttribute("normal", new Float32BufferAttribute(m.nor, 3));
    g.setIndex(new Uint32BufferAttribute(m.idx, 1));
    return g;
}
/** A primitive's tessellated surface as a world-space BufferGeometry (for CSG). */
export function primitiveGeometry(prim) {
    const m = meshFor(prim);
    const g = new BufferGeometry();
    g.setAttribute("position", new Float32BufferAttribute(m.pos, 3));
    g.setAttribute("normal", new Float32BufferAttribute(m.nor, 3));
    g.setIndex(new Uint32BufferAttribute(m.idx, 1));
    const mat = new Matrix4().compose(prim.position, new Quaternion().setFromEuler(prim.rotation), new Vector3(1, 1, 1));
    g.applyMatrix4(mat);
    return g;
}
function meshFor(prim) {
    switch (prim.type) {
        case PrimType.Sphere: return sphereMesh(prim.a);
        case PrimType.Box: return boxMesh(prim.a, prim.b, prim.c);
        case PrimType.Torus: return torusMesh(prim.a, prim.b);
        case PrimType.Cylinder: return cylinderMesh(prim.a, prim.b);
        case PrimType.Cone: return coneMesh(prim.a, prim.b);
        case PrimType.Octahedron: return octahedronMesh(prim.a);
        case PrimType.Capsule: return capsuleMesh(prim.a, prim.b);
        case PrimType.Pyramid: return pyramidMesh(prim.a, prim.b);
        case PrimType.RoundedBox: return boxMesh(prim.a, prim.b, prim.c); // rounding approximated
    }
}
/** De-index a PrimMesh into the renderer's 32-float triangle layout and build a
 *  BVH. Local material id 0 (the single material added by the caller). */
function buildBlas(m) {
    const triCount = m.idx.length / 3;
    const raw = new Float32Array(triCount * TRI_STRIDE);
    for (let t = 0; t < triCount; t++) {
        const o = t * TRI_STRIDE;
        for (let v = 0; v < 3; v++) {
            const vi = m.idx[t * 3 + v];
            raw[o + v * 4] = m.pos[vi * 3];
            raw[o + v * 4 + 1] = m.pos[vi * 3 + 1];
            raw[o + v * 4 + 2] = m.pos[vi * 3 + 2];
            raw[o + 12 + v * 4] = m.nor[vi * 3];
            raw[o + 12 + v * 4 + 1] = m.nor[vi * 3 + 1];
            raw[o + 12 + v * 4 + 2] = m.nor[vi * 3 + 2];
        }
        // uv01 / uv2 stay 0; matId (offset 30) stays 0.
        raw[o + 19] = 0xffffff; // n1.w / n2.w / uv2m.w = white vertex color (no tint)
        raw[o + 23] = 0xffffff;
        raw[o + 31] = 0xffffff;
    }
    computeFaceTangents(raw);
    const { nodes, tris, nodeCount } = buildBVH(raw, TRI_STRIDE);
    return { tris, triCount: tris.length / TRI_STRIDE, nodes, nodeCount };
}
/**
 * Tessellate a procedural primitive into an editable mesh model + a raster
 * preview group. The shape's color and reflectivity carry over (mapped to a
 * metal/rough PBR material); procedural patterns, size sliders and boolean
 * carve do not survive the conversion.
 */
export function convertPrimitive(prim) {
    const mesh = meshFor(prim);
    const blas = buildBlas(mesh);
    const metalness = prim.reflectivity;
    const roughness = Math.max(0.04, 1 - prim.reflectivity);
    const materials = [
        prim.color[0], prim.color[1], prim.color[2], 1,
        0, 0, 0, 0,
        metalness, roughness, -1, -1,
    ];
    const model = {
        name: `${prim.name} (mesh)`,
        blas,
        materials,
        materialNames: [prim.name],
        texLayers: [],
        normalLayers: [],
        texSize: 256,
        halfHeight: 0,
    };
    // Preview group mirrors the importer's structure: outer group → inner mesh.
    const geom = new BufferGeometry();
    geom.setAttribute("position", new Float32BufferAttribute(mesh.pos, 3));
    geom.setAttribute("normal", new Float32BufferAttribute(mesh.nor, 3));
    geom.setIndex(new Uint32BufferAttribute(mesh.idx, 1));
    const previewMat = new MeshStandardMaterial({ metalness, roughness });
    previewMat.color.setRGB(prim.color[0], prim.color[1], prim.color[2]);
    const previewMesh = new Mesh(geom, previewMat);
    const group = new Group();
    group.add(previewMesh);
    return { model, group };
}
