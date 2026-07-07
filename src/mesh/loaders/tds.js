import { TDSLoader } from "three/examples/jsm/loaders/TDSLoader.js";
import { registerLoader } from "../modelImport";
const loader = new TDSLoader();
// 3DS is a legacy Autodesk format still common in asset libraries. parse() takes
// the raw bytes plus a base path for resolving texture references (none here).
registerLoader({
    label: "3DS",
    exts: ["3ds"],
    load: async (file) => {
        const root = loader.parse(await file.arrayBuffer(), "");
        return { root };
    },
});
