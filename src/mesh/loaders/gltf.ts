import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { registerLoader } from "../modelImport";

const loader = new GLTFLoader();

registerLoader({
  label: "glTF",
  exts: ["glb", "gltf"],
  load: async (file) => {
    const gltf = await loader.parseAsync(await file.arrayBuffer(), "");
    return { root: gltf.scene, animations: gltf.animations };
  },
});
