import { FBXLoader } from "three/examples/jsm/loaders/FBXLoader.js";
import { registerLoader } from "../modelImport";

const loader = new FBXLoader();

// FBX is the standard rigged/animated character format (e.g. Mixamo exports).
// FBXLoader returns a Group with embedded materials, skeletons, and clips.
registerLoader({
  label: "FBX",
  exts: ["fbx"],
  load: async (file) => {
    const root = loader.parse(await file.arrayBuffer(), "");
    return { root, animations: root.animations };
  },
});
