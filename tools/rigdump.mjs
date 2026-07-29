/**
 * Extract just the skeleton from a .glb and report how the poser will treat it.
 *
 *   node tools/rigdump.mjs path/to/model.glb [out.json]
 *
 * Reads only the glTF JSON chunk plus the skin weight accessors — no textures, no
 * geometry — so a 120 MB character turns into a few hundred KB of rig. Prints a
 * summary and writes the full dump for offline analysis of chain selection.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { Bone, Vector3, Matrix4 } from "three";
import { ikChain, boneChildCount } from "../src/mesh/poseChain.ts";

const [file, outPath = "rig.json"] = process.argv.slice(2);
if (!file) {
  console.error("usage: node tools/rigdump.mjs path/to/model.glb [out.json]");
  process.exit(2);
}

// ---- GLB container: 12-byte header, then chunks (JSON first, BIN second) ----
const buf = readFileSync(file);
const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
if (dv.getUint32(0, true) !== 0x46546c67) throw new Error("not a GLB (bad magic)");
let off = 12, json = null, bin = null;
while (off < buf.byteLength) {
  const len = dv.getUint32(off, true), type = dv.getUint32(off + 4, true);
  const body = buf.subarray(off + 8, off + 8 + len);
  if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(body));
  else if (type === 0x004e4942) bin = body;
  off += 8 + len + ((4 - (len % 4)) % 4);
}
if (!json) throw new Error("no JSON chunk");

// ---- accessor reader (enough of the spec for JOINTS_0 / WEIGHTS_0) ----
// Read through a DataView rather than typed-array views: accessor offsets aren't
// guaranteed to be aligned for the element type once the file's own offset is
// added, and a misaligned `new Uint16Array(buffer, offset)` throws outright.
const COMP = {
  5120: [1, (d, o) => d.getInt8(o), 127],
  5121: [1, (d, o) => d.getUint8(o), 255],
  5122: [2, (d, o) => d.getInt16(o, true), 32767],
  5123: [2, (d, o) => d.getUint16(o, true), 65535],
  5125: [4, (d, o) => d.getUint32(o, true), 0],
  5126: [4, (d, o) => d.getFloat32(o, true), 0],
};
const NUM = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
function readAccessor(i) {
  const acc = json.accessors[i];
  const n = NUM[acc.type];
  const [size, get, max] = COMP[acc.componentType];
  const out = new Float32Array(acc.count * n);
  if (acc.bufferView === undefined || !bin) return out; // sparse / zero-filled
  const view = json.bufferViews[acc.bufferView];
  const base = (view.byteOffset ?? 0) + (acc.byteOffset ?? 0);
  const stride = view.byteStride || n * size;
  const bd = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  const scale = acc.normalized && max ? max : 0;
  for (let e = 0; e < acc.count; e++) {
    for (let k = 0; k < n; k++) {
      const v = get(bd, base + e * stride + k * size);
      out[e * n + k] = scale ? v / scale : v;
    }
  }
  return out;
}

// ---- build the bone hierarchy the app would pose ----
const skin = json.skins?.[0];
if (!skin) throw new Error("no skin — this model isn't rigged");
const joints = skin.joints;
const jointSet = new Set(joints);
const parentOf = new Map();
json.nodes.forEach((nd, i) => (nd.children ?? []).forEach((c) => parentOf.set(c, i)));

const bones = joints.map((ni) => {
  const nd = json.nodes[ni];
  const b = new Bone();
  b.name = nd.name || `node${ni}`;
  if (nd.matrix) new Matrix4().fromArray(nd.matrix).decompose(b.position, b.quaternion, b.scale);
  else {
    if (nd.translation) b.position.fromArray(nd.translation);
    if (nd.rotation) b.quaternion.fromArray(nd.rotation);
    if (nd.scale) b.scale.fromArray(nd.scale);
  }
  return b;
});
const byNode = new Map(joints.map((ni, i) => [ni, bones[i]]));
joints.forEach((ni, i) => {
  const p = parentOf.get(ni);
  if (p !== undefined && byNode.has(p)) byNode.get(p).add(bones[i]);
});
const roots = bones.filter((b) => !b.parent);
roots.forEach((r) => r.updateWorldMatrix(true, true));

// ---- skin influence per joint, summed over every primitive using this skin ----
const influence = new Float32Array(joints.length);
let vertsScanned = 0;
for (const mesh of json.meshes ?? []) {
  for (const prim of mesh.primitives ?? []) {
    const ja = prim.attributes?.JOINTS_0, wa = prim.attributes?.WEIGHTS_0;
    if (ja === undefined || wa === undefined || !bin) continue;
    const ji = readAccessor(ja), wi = readAccessor(wa);
    vertsScanned += ji.length / 4;
    for (let v = 0; v < ji.length; v += 4) {
      for (let k = 0; k < 4; k++) {
        const w = wi[v + k];
        if (w > 0 && ji[v + k] < influence.length) influence[ji[v + k]] += w;
      }
    }
  }
}

// ---- bind-pose scale sanity: mirrored (negative) or non-uniform scale breaks
// quaternion-only IK, and rigs mirror their right side surprisingly often ----
const ibm = skin.inverseBindMatrices !== undefined ? readAccessor(skin.inverseBindMatrices) : null;
const m4 = new Matrix4();
const scaleFlags = bones.map((b, i) => {
  const s = b.scale;
  const nonUniform = Math.abs(s.x - s.y) > 1e-4 || Math.abs(s.y - s.z) > 1e-4;
  const negLocal = s.x < 0 || s.y < 0 || s.z < 0;
  let negBind = false;
  if (ibm) {
    m4.fromArray(ibm.subarray(i * 16, i * 16 + 16));
    negBind = m4.determinant() < 0;
  }
  return { nonUniform, negLocal, negBind };
});

// ---- what the poser will do with each joint ----
const DEFORM_MIN = 1.0;
const set = new Set(bones);
const wp = new Vector3(), wp2 = new Vector3();
const deforms = bones.map((_, i) => influence[i] >= DEFORM_MIN);
const rows = bones.map((b, i) => {
  const chain = ikChain(bones, i);
  const par = b.parent && set.has(b.parent) ? b.parent : null;
  return {
    i, name: b.name,
    influence: +influence[i].toFixed(2),
    deforms: deforms[i],
    children: b.children.filter((c) => set.has(c)).length,
    forks: boneChildCount(b, set),
    segment: par ? +b.getWorldPosition(wp).distanceTo(par.getWorldPosition(wp2)).toFixed(4) : 0,
    chain: chain.map((c) => c.name),
    rotating: Math.max(0, chain.length - 1),
    ...scaleFlags[i],
  };
});

const deformCount = deforms.filter(Boolean).length;
const say = (...a) => console.log(...a);
say(`\n${file}`);
say(`joints: ${bones.length} · roots: ${roots.length} · verts scanned: ${vertsScanned}`);
say(`carry skin weight (>= ${DEFORM_MIN}): ${deformCount} · helpers: ${bones.length - deformCount}`);
say(`mirrored bind (negative determinant): ${scaleFlags.filter((f) => f.negBind).length}`);
say(`negative local scale: ${scaleFlags.filter((f) => f.negLocal).length} · non-uniform scale: ${scaleFlags.filter((f) => f.nonUniform).length}`);

// Things most likely to feel "goofy" while posing.
const deforming = rows.filter((r) => r.deforms);
const stuck = deforming.filter((r) => r.rotating <= 1 && r.children > 0);
const long = deforming.filter((r) => r.rotating > 4);
const helperDriven = deforming.filter((r) => r.chain.slice(0, -1).some((n) => {
  const j = rows.find((x) => x.name === n);
  return j && !j.deforms;
}));
say(`\nsuspicious:`);
say(`  only one bone rotates (drag barely moves anything): ${stuck.length}`);
say(`  long CCD chains (>4 rotating): ${long.length}`);
say(`  chain rotates a ZERO-WEIGHT helper bone: ${helperDriven.length}`);
const show = (label, list) => {
  if (!list.length) return;
  say(`\n${label}`);
  for (const r of list.slice(0, 12)) say(`  ${r.name}  [w ${r.influence}, forks ${r.forks}]  → ${r.chain.join(" → ")}`);
  if (list.length > 12) say(`  … and ${list.length - 12} more`);
};
show("one-bone chains:", stuck);
show("long chains:", long);
show("helper-driven chains:", helperDriven);

// Landmarks, so chain selection can be eyeballed against anatomy.
const LANDMARK = /(fore ?arm|elbow|hand$|wrist|shoulder|upper ?arm|clavicle|thigh|shin|knee|calf|foot|ankle|head$|neck$|spine|hip)/i;
show("landmarks:", deforming.filter((r) => LANDMARK.test(r.name)));

writeFileSync(outPath, JSON.stringify({
  file, joints: bones.length, deformCount, vertsScanned,
  scaleSummary: {
    negBind: scaleFlags.filter((f) => f.negBind).length,
    negLocal: scaleFlags.filter((f) => f.negLocal).length,
    nonUniform: scaleFlags.filter((f) => f.nonUniform).length,
  },
  bones: rows,
}, null, 1));
say(`\nwrote ${outPath}`);
