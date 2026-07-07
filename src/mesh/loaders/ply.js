import { Mesh, MeshStandardMaterial } from "three";
import { PLYLoader } from "three/examples/jsm/loaders/PLYLoader.js";
import { registerLoader } from "../modelImport";
const loader = new PLYLoader();
// PLY is common for scans/photogrammetry and frequently carries per-vertex
// colors (which our shader honors via the COLOR_0 path).
registerLoader({
    label: "PLY",
    exts: ["ply"],
    load: async (file) => {
        const geom = loader.parse(await file.arrayBuffer());
        if (!geom.getAttribute("normal"))
            geom.computeVertexNormals();
        const vertexColors = !!geom.getAttribute("color");
        const mesh = new Mesh(geom, new MeshStandardMaterial({ color: 0xffffff, vertexColors, metalness: 0, roughness: 0.85 }));
        return { root: mesh };
    },
});
