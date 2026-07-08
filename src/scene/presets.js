// Curated starter scenes for the gallery. Each preset composes the existing,
// already-tuned terrain recipes / biomes / sky presets with a few props so a new
// user lands on something finished-looking in one click. Props are grounded on
// the terrain surface via the same height field the ray tracer uses, so contact
// shadows read correctly.
import { Scene, PrimType, LightType } from "./scene";
import { worldTerrainHeight } from "../gen/terrainField";
/** A fresh scene with a sun; presets layer terrain/sky/props on top. */
function withSun() {
    const s = new Scene();
    s.addLight(LightType.Directional);
    return s;
}
/** Sit a primitive on the terrain surface at (x, z) — or on the water when the
 *  ground there is submerged — so its base rests on the ground and grounds with
 *  a contact shadow. Call after the shape's size (`a`) is set. */
function ground(s, p, x, z) {
    const surface = worldTerrainHeight(x, z, s.world);
    const base = s.world.waterEnabled ? Math.max(surface, 0) : surface;
    p.position.set(x, base + p.a, z);
}
export const SCENE_PRESETS = [
    {
        id: "alpine-dawn",
        name: "Alpine Dawn",
        description: "Snow-capped peaks over a still lake at first light.",
        camera: { target: [0, 6, 0], distance: 95, yaw: -0.7, pitch: 0.16 },
        build: () => {
            const s = withSun();
            s.applyTerrainRecipe("mountain");
            s.applyBiome("alpine");
            s.world.waterEnabled = true;
            s.applyTimeOfDay(6.6); // low warm sun → golden peaks + long shadows
            s.touch();
            return s;
        },
    },
    {
        id: "desert-dusk",
        name: "Desert Mesa, Dusk",
        description: "Terraced mesas glowing under a sinking sun.",
        camera: { target: [0, 5, 0], distance: 78, yaw: 0.5, pitch: 0.2 },
        build: () => {
            const s = withSun();
            s.applyTerrainRecipe("mesa");
            s.applyBiome("desert");
            s.applyTimeOfDay(17.6);
            const r1 = s.addPrim(PrimType.Octahedron);
            r1.a = 4;
            r1.color = [0.5, 0.34, 0.22];
            r1.reflectivity = 0.02;
            r1.name = "Boulder";
            ground(s, r1, 20, 12);
            const r2 = s.addPrim(PrimType.Box);
            r2.a = 3;
            r2.b = 4;
            r2.c = 3;
            r2.rotation.y = 0.7;
            r2.color = [0.55, 0.4, 0.28];
            r2.reflectivity = 0.02;
            r2.name = "Slab";
            ground(s, r2, -16, 22);
            s.touch();
            return s;
        },
    },
    {
        id: "canyon-river",
        name: "Canyon River",
        description: "Deep water-carved canyons under a bright midday sky.",
        camera: { target: [0, 2, 0], distance: 72, yaw: -0.4, pitch: 0.14 },
        build: () => {
            const s = withSun();
            s.applyTerrainRecipe("canyon");
            s.applyBiome("badlands");
            s.world.waterEnabled = true;
            s.applyTimeOfDay(12);
            s.touch();
            return s;
        },
    },
    {
        id: "ringed-giant",
        name: "Ringed Giant",
        description: "A ringed gas giant hangs over moonlit dunes and stars.",
        camera: { target: [0, 8, 0], distance: 82, yaw: -1.0, pitch: 0.34 },
        build: () => {
            const s = withSun();
            s.applyTerrainRecipe("dunes");
            s.applyBiome("desert");
            s.applyTimeOfDay(22.5); // night → stars on, moonlight floor
            const planet = s.addLight(LightType.Point);
            planet.inSky = true;
            planet.position.set(-0.4, 0.5, -1).normalize().multiplyScalar(100); // sky direction
            planet.bodyRadius = 8;
            planet.color = [0.86, 0.72, 0.5];
            planet.intensity = 2.4;
            planet.ringOpacity = 0.85;
            planet.ringColor = [0.85, 0.78, 0.62];
            planet.name = "Gas Giant";
            s.touch();
            return s;
        },
    },
    {
        id: "chrome-garden",
        name: "Chrome Garden",
        description: "Polished shapes on rolling green hills — reflections galore.",
        camera: { target: [0, 6, 0], distance: 56, yaw: -0.6, pitch: 0.22 },
        build: () => {
            const s = withSun();
            s.applyTerrainRecipe("hills");
            s.applyBiome("alpine");
            s.world.waterEnabled = true;
            s.applyTimeOfDay(9.5);
            const a = s.addPrim(PrimType.Sphere);
            a.a = 6;
            a.color = [0.95, 0.96, 0.98];
            a.reflectivity = 0.92;
            a.name = "Chrome Sphere";
            ground(s, a, 12, 8);
            const b = s.addPrim(PrimType.Torus);
            b.a = 5;
            b.b = 1.8;
            b.rotation.x = 0.6;
            b.color = [0.8, 0.3, 0.25];
            b.reflectivity = 0.4;
            b.name = "Ring";
            ground(s, b, -12, 14);
            const c = s.addPrim(PrimType.Octahedron);
            c.a = 5;
            c.color = [0.3, 0.7, 0.85];
            c.reflectivity = 0.6;
            c.name = "Gem";
            ground(s, c, 0, -14);
            s.touch();
            return s;
        },
    },
    {
        id: "taiga-mist",
        name: "Taiga Mist",
        description: "Boreal ridgelines fading into a hazy overcast.",
        camera: { target: [0, 6, 0], distance: 90, yaw: -0.5, pitch: 0.18 },
        build: () => {
            const s = withSun();
            s.applyTerrainRecipe("taiga");
            s.applyBiome("tundra");
            s.world.waterEnabled = true;
            s.applySkyPreset("overcast");
            s.world.hazeDensity = 0.004; // thicker aerial perspective
            s.touch();
            return s;
        },
    },
];
