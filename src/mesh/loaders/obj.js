import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { registerLoader } from "../modelImport";
const loader = new OBJLoader();
// OBJ is one of the most common interchange formats. Materials and textures
// live in a sibling .mtl file, so a single-file drop imports geometry with a
// default material (full .mtl + texture pairing would need multi-file import).
registerLoader({
    label: "OBJ",
    exts: ["obj"],
    load: async (file) => {
        const root = loader.parse(await file.text());
        return { root };
    },
});
