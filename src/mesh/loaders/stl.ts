import { Mesh, MeshStandardMaterial } from "three";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { registerLoader } from "../modelImport";

const loader = new STLLoader();

// STL is geometry only (no UVs, materials, or color) — common for CAD/3D-print.
registerLoader({
  label: "STL",
  exts: ["stl"],
  load: async (file) => {
    const geom = loader.parse(await file.arrayBuffer());
    if (!geom.getAttribute("normal")) geom.computeVertexNormals();
    const mesh = new Mesh(geom, new MeshStandardMaterial({ color: 0xb6bcc6, metalness: 0, roughness: 0.7 }));
    return { root: mesh };
  },
});
