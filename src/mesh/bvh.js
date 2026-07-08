// A compact binary BVH built on the CPU and flattened for GPU traversal.
//
// Input is a triangle buffer (AoS, `stride` floats per triangle, positions in
// the first three vec4 slots). Output is a flat node array plus the triangles
// reordered into BVH leaf order, so a leaf can reference a contiguous range
// directly — no indirection buffer needed on the GPU.
//
// Node layout (8 floats):
//   [minX, minY, minZ, A,  maxX, maxY, maxZ, B]
//   leaf     → B = count  > 0, A = first triangle index
//   internal → B = -(rightChild + 1) (< 0), A = left child node index
//   (right child index is stored explicitly — the left subtree is allocated
//    between this node and its right child, so right ≠ left + 1.)
const LEAF_SIZE = 4;
export function buildBVH(triData, stride) {
    const triCount = Math.floor(triData.length / stride);
    if (triCount === 0) {
        return { nodes: new Float32Array(8), tris: new Float32Array(0), nodeCount: 1 };
    }
    const idx = new Uint32Array(triCount);
    const cent = new Float32Array(triCount * 3);
    const tmin = new Float32Array(triCount * 3);
    const tmax = new Float32Array(triCount * 3);
    for (let t = 0; t < triCount; t++) {
        idx[t] = t;
        const o = t * stride;
        let lx = Infinity, ly = Infinity, lz = Infinity;
        let hx = -Infinity, hy = -Infinity, hz = -Infinity;
        for (let v = 0; v < 3; v++) {
            const p = o + v * 4; // p0,p1,p2 live in vec4 slots 0,1,2
            const x = triData[p], y = triData[p + 1], z = triData[p + 2];
            lx = Math.min(lx, x);
            ly = Math.min(ly, y);
            lz = Math.min(lz, z);
            hx = Math.max(hx, x);
            hy = Math.max(hy, y);
            hz = Math.max(hz, z);
        }
        tmin[t * 3] = lx;
        tmin[t * 3 + 1] = ly;
        tmin[t * 3 + 2] = lz;
        tmax[t * 3] = hx;
        tmax[t * 3 + 1] = hy;
        tmax[t * 3 + 2] = hz;
        cent[t * 3] = (lx + hx) * 0.5;
        cent[t * 3 + 1] = (ly + hy) * 0.5;
        cent[t * 3 + 2] = (lz + hz) * 0.5;
    }
    const nodes = [];
    function bounds(start, end) {
        let lx = Infinity, ly = Infinity, lz = Infinity;
        let hx = -Infinity, hy = -Infinity, hz = -Infinity;
        for (let i = start; i < end; i++) {
            const t = idx[i];
            lx = Math.min(lx, tmin[t * 3]);
            ly = Math.min(ly, tmin[t * 3 + 1]);
            lz = Math.min(lz, tmin[t * 3 + 2]);
            hx = Math.max(hx, tmax[t * 3]);
            hy = Math.max(hy, tmax[t * 3 + 1]);
            hz = Math.max(hz, tmax[t * 3 + 2]);
        }
        return { lx, ly, lz, hx, hy, hz };
    }
    function build(start, end) {
        const ni = nodes.length / 8;
        for (let k = 0; k < 8; k++)
            nodes.push(0);
        const b = bounds(start, end);
        const count = end - start;
        const writeAABB = (leftFirst, cnt) => {
            const o = ni * 8;
            nodes[o] = b.lx;
            nodes[o + 1] = b.ly;
            nodes[o + 2] = b.lz;
            nodes[o + 3] = leftFirst;
            nodes[o + 4] = b.hx;
            nodes[o + 5] = b.hy;
            nodes[o + 6] = b.hz;
            nodes[o + 7] = cnt;
        };
        if (count <= LEAF_SIZE) {
            writeAABB(start, count);
            return ni;
        }
        // Split on the longest axis of the centroid bounds, at its midpoint.
        let cl = [Infinity, Infinity, Infinity];
        let ch = [-Infinity, -Infinity, -Infinity];
        for (let i = start; i < end; i++) {
            const t = idx[i];
            for (let a = 0; a < 3; a++) {
                cl[a] = Math.min(cl[a], cent[t * 3 + a]);
                ch[a] = Math.max(ch[a], cent[t * 3 + a]);
            }
        }
        let axis = 0;
        if (ch[1] - cl[1] > ch[axis] - cl[axis])
            axis = 1;
        if (ch[2] - cl[2] > ch[axis] - cl[axis])
            axis = 2;
        // Median (object) split: sort the range by centroid on the longest axis and
        // split at the midpoint. This guarantees a balanced tree (~log2 n depth) so
        // GPU traversal never overflows its fixed stack — a spatial-midpoint split
        // can degenerate to O(n) depth and drop geometry.
        const sub = Array.from(idx.subarray(start, end));
        sub.sort((a, b) => cent[a * 3 + axis] - cent[b * 3 + axis]);
        idx.set(sub, start);
        const mid = (start + end) >> 1;
        const left = build(start, mid);
        const right = build(mid, end);
        writeAABB(left, -(right + 1)); // internal: store both child indices
        return ni;
    }
    build(0, triCount);
    // Reorder triangles into leaf order.
    const tris = new Float32Array(triCount * stride);
    for (let i = 0; i < triCount; i++) {
        tris.set(triData.subarray(idx[i] * stride, idx[i] * stride + stride), i * stride);
    }
    return { nodes: Float32Array.from(nodes), tris, nodeCount: nodes.length / 8 };
}
// Small leaves: each mesh instance still costs a full BLAS descent, so keep
// leaves tight to maximise culling. 2 halves the node count vs. 1 at little
// traversal cost.
const TLAS_LEAF_SIZE = 2;
// Build the top-level acceleration structure: a BVH over per-instance world-space
// AABBs. Node encoding is identical to the BLAS (see buildBVH), but a leaf's
// `first` indexes the returned `order` array (instance indices), not triangles.
export function buildTLAS(mins, maxs, count) {
    if (count === 0) {
        return { nodes: new Float32Array(8), order: new Uint32Array(1), nodeCount: 1 };
    }
    const idx = new Uint32Array(count);
    const cent = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        idx[i] = i;
        for (let a = 0; a < 3; a++)
            cent[i * 3 + a] = (mins[i * 3 + a] + maxs[i * 3 + a]) * 0.5;
    }
    const nodes = [];
    function bounds(start, end) {
        let lx = Infinity, ly = Infinity, lz = Infinity;
        let hx = -Infinity, hy = -Infinity, hz = -Infinity;
        for (let i = start; i < end; i++) {
            const t = idx[i];
            lx = Math.min(lx, mins[t * 3]);
            ly = Math.min(ly, mins[t * 3 + 1]);
            lz = Math.min(lz, mins[t * 3 + 2]);
            hx = Math.max(hx, maxs[t * 3]);
            hy = Math.max(hy, maxs[t * 3 + 1]);
            hz = Math.max(hz, maxs[t * 3 + 2]);
        }
        return { lx, ly, lz, hx, hy, hz };
    }
    function build(start, end) {
        const ni = nodes.length / 8;
        for (let k = 0; k < 8; k++)
            nodes.push(0);
        const b = bounds(start, end);
        const count = end - start;
        const writeAABB = (leftFirst, cnt) => {
            const o = ni * 8;
            nodes[o] = b.lx;
            nodes[o + 1] = b.ly;
            nodes[o + 2] = b.lz;
            nodes[o + 3] = leftFirst;
            nodes[o + 4] = b.hx;
            nodes[o + 5] = b.hy;
            nodes[o + 6] = b.hz;
            nodes[o + 7] = cnt;
        };
        if (count <= TLAS_LEAF_SIZE) {
            writeAABB(start, count);
            return ni;
        }
        // Split on the longest axis of the centroid bounds (median object split, as
        // in buildBVH — guarantees ~log2 n depth so the GPU stack can't overflow).
        let cl = [Infinity, Infinity, Infinity];
        let ch = [-Infinity, -Infinity, -Infinity];
        for (let i = start; i < end; i++) {
            const t = idx[i];
            for (let a = 0; a < 3; a++) {
                cl[a] = Math.min(cl[a], cent[t * 3 + a]);
                ch[a] = Math.max(ch[a], cent[t * 3 + a]);
            }
        }
        let axis = 0;
        if (ch[1] - cl[1] > ch[axis] - cl[axis])
            axis = 1;
        if (ch[2] - cl[2] > ch[axis] - cl[axis])
            axis = 2;
        const sub = Array.from(idx.subarray(start, end));
        sub.sort((a, b) => cent[a * 3 + axis] - cent[b * 3 + axis]);
        idx.set(sub, start);
        const mid = (start + end) >> 1;
        const left = build(start, mid);
        const right = build(mid, end);
        writeAABB(left, -(right + 1));
        return ni;
    }
    build(0, count);
    return { nodes: Float32Array.from(nodes), order: idx, nodeCount: nodes.length / 8 };
}
// Recompute a TLAS's node AABBs in place from fresh instance AABBs, keeping the
// existing tree topology (leaf membership + child links). Much cheaper than a
// rebuild — no sorting, no allocation — so it can run every drag frame.
//
// buildTLAS emits each parent before its children (parent index < child index),
// so iterating from the last node down to the root visits children before their
// parent, letting internal nodes union their already-refreshed children.
export function refitTLAS(nodes, order, nodeCount, mins, maxs) {
    for (let ni = nodeCount - 1; ni >= 0; ni--) {
        const o = ni * 8;
        const b = nodes[o + 7]; // leaf: count > 0; internal: -(rightChild + 1) < 0
        let lx = Infinity, ly = Infinity, lz = Infinity;
        let hx = -Infinity, hy = -Infinity, hz = -Infinity;
        if (b > 0) {
            const first = nodes[o + 3];
            for (let k = 0; k < b; k++) {
                const ii = order[first + k];
                lx = Math.min(lx, mins[ii * 3]);
                ly = Math.min(ly, mins[ii * 3 + 1]);
                lz = Math.min(lz, mins[ii * 3 + 2]);
                hx = Math.max(hx, maxs[ii * 3]);
                hy = Math.max(hy, maxs[ii * 3 + 1]);
                hz = Math.max(hz, maxs[ii * 3 + 2]);
            }
        }
        else {
            const lo = nodes[o + 3] * 8; // left child node
            const ro = (-b - 1) * 8; // right child node
            lx = Math.min(nodes[lo], nodes[ro]);
            ly = Math.min(nodes[lo + 1], nodes[ro + 1]);
            lz = Math.min(nodes[lo + 2], nodes[ro + 2]);
            hx = Math.max(nodes[lo + 4], nodes[ro + 4]);
            hy = Math.max(nodes[lo + 5], nodes[ro + 5]);
            hz = Math.max(nodes[lo + 6], nodes[ro + 6]);
        }
        nodes[o] = lx;
        nodes[o + 1] = ly;
        nodes[o + 2] = lz;
        nodes[o + 4] = hx;
        nodes[o + 5] = hy;
        nodes[o + 6] = hz;
        // Child links (o+3, o+7) are left untouched — topology is preserved.
    }
}
