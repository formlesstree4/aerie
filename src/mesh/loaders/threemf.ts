import { ThreeMFLoader } from "three/examples/jsm/loaders/3MFLoader.js";
import { registerLoader } from "../modelImport";

const loader = new ThreeMFLoader();

// 3MF is the modern 3D-printing container (a zip): geometry plus per-object
// colors/materials. parse() takes the raw archive bytes and returns a Group.
registerLoader({
  label: "3MF",
  exts: ["3mf"],
  load: async (file) => {
    const root = loader.parse(await file.arrayBuffer());
    return { root };
  },
});
