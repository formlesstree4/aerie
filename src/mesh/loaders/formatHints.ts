// Known 3D formats Aerie can't load directly, each with a tailored "way forward".
// Matched by extension only — we trust the file name, do no content sniffing — so
// an unsupported drop becomes actionable guidance instead of a dead end.
//
// Most advice funnels toward glTF 2.0 (.glb), OBJ or STL, since those are the
// formats the registered loaders in ./index already handle well.

interface Hint {
  name: string; // origin tool / format family, e.g. "Blender"
  advice: string; // how to get this into Aerie
}

/** Expand a shared hint across every extension that maps to it. */
function group(exts: string[], name: string, advice: string): [string, Hint][] {
  return exts.map((e) => [e, { name, advice }]);
}

const HINTS = new Map<string, Hint>([
  // ── Native DCC scene files: re-export as glTF/FBX ──────────────────────────
  ...group(["blend", "blend1"], "Blender",
    "in Blender use File ▸ Export ▸ glTF 2.0 (.glb) and import that."),
  ...group(["max"], "3ds Max",
    "in 3ds Max export glTF (.glb) or FBX and import the result."),
  ...group(["ma", "mb"], "Maya",
    "in Maya use the glTF/GLB or FBX exporter and import that instead."),
  ...group(["c4d"], "Cinema 4D",
    "in Cinema 4D use File ▸ Export ▸ glTF (.glb) or FBX and import that."),
  ...group(["skp"], "SketchUp",
    "in SketchUp export a glTF (.glb), Collada (.dae) or OBJ and import that."),
  ...group(["lxo", "lxl"], "Modo",
    "in Modo export as glTF (.glb) or FBX and import that."),
  ...group(["hip", "hipnc"], "Houdini",
    "in Houdini export the geometry as glTF (.glb) or FBX and import that."),
  ...group(["ztl", "zpr"], "ZBrush",
    "in ZBrush export an OBJ or FBX (decimate first if it's high-poly) and import that."),

  // ── CAD solids: boundary-rep, must be tessellated to a mesh first ──────────
  ...group(["step", "stp", "iges", "igs"], "STEP/IGES CAD",
    "this is a CAD solid — convert it to glTF, OBJ or STL (e.g. FreeCAD, Fusion 360) and import that."),
  ...group(["sldprt", "sldasm"], "SolidWorks",
    "export it as STL, glTF or OBJ from SolidWorks and import that."),
  ...group(["ipt", "iam"], "Autodesk Inventor",
    "export it as STL, glTF or OBJ from Inventor and import that."),
  ...group(["catpart", "catproduct"], "CATIA",
    "export it as STEP, then convert to glTF/OBJ/STL (e.g. FreeCAD) and import that."),
  ...group(["f3d"], "Fusion 360",
    "in Fusion 360 use Export ▸ Mesh (STL/OBJ) or glTF and import that."),
  ...group(["3dm"], "Rhino",
    "in Rhino export a glTF (.glb), OBJ or STL and import that."),
  ...group(["x_t", "x_b", "xmt_txt"], "Parasolid",
    "convert it to STEP, then to glTF/OBJ/STL (e.g. FreeCAD) and import that."),
  ...group(["dwg", "dxf"], "AutoCAD",
    "these are mostly 2D/CAD — export any 3D geometry as OBJ, STL or glTF and import that."),

  // ── Scene interchange ─────────────────────────────────────────────────────
  ...group(["usd", "usda", "usdc", "usdz"], "USD",
    "convert it to glTF with usd tools or Blender (Import USD, then Export glTF) and import that."),
  ...group(["abc"], "Alembic",
    "re-export it as glTF (.glb) or FBX from your DCC tool and import that."),

  // ── Voxel / misc mesh formats ─────────────────────────────────────────────
  ...group(["vox"], "MagicaVoxel",
    "in MagicaVoxel export an OBJ and import that."),
  ...group(["x"], "DirectX (.x)",
    "convert it to FBX or OBJ (e.g. Blender, Assimp) and import that."),
  ...group(["wrl", "vrml"], "VRML",
    "convert it to OBJ or glTF (e.g. Blender, MeshLab) and import that."),
]);

/** If `filename`'s extension is a recognized-but-unsupported 3D format, return a
 *  human-readable message explaining how to get it into Aerie; else undefined. */
export function formatHint(filename: string): string | undefined {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  const hint = HINTS.get(ext);
  if (!hint) return undefined;
  return `${hint.name} files (.${ext}) can't be imported directly — ${hint.advice}`;
}
