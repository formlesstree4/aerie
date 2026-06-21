import { ColladaLoader } from "three/examples/jsm/loaders/ColladaLoader.js";
import { registerLoader } from "../modelImport";

const loader = new ColladaLoader();

// Collada (.dae) is a widely-exported XML scene format carrying meshes,
// materials, and (sometimes) animation. parse() returns a wrapper whose `scene`
// is the root; clips hang off scene.animations.
registerLoader({
  label: "Collada",
  exts: ["dae"],
  load: async (file) => {
    const result = loader.parse(await file.text(), "");
    return { root: result.scene, animations: result.scene.animations };
  },
});
