// Importing this barrel registers every supported format loader (side effects).
// Add new formats here, in tier order.
import "./gltf"; // .glb / .gltf
import "./fbx"; //  .fbx  (rigged/animated characters)
import "./obj"; //  .obj  (ubiquitous interchange; geometry + default material)
import "./collada"; // .dae (XML scene: meshes, materials, animation)
import "./threemf"; // .3mf (3D-printing container with colors)
import "./tds"; //  .3ds  (legacy Autodesk)
import "./stl"; //  .stl  (static CAD/print geometry)
import "./ply"; //  .ply  (scans, often vertex-colored)
