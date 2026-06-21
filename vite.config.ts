import { defineConfig } from "vite";

// WGSL shaders are imported as raw strings via the `?raw` suffix,
// so no custom plugin is needed.
export default defineConfig({
  server: { port: 5173 },
  build: { target: "esnext" },
});
