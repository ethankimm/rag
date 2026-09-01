import { defineConfig } from "astro/config";
import react from "@astrojs/react";
import { loadEnv } from "vite";

const mode = process.env.NODE_ENV === "production" ? "production" : "development";
const env = loadEnv(mode, process.cwd(), "");

export default defineConfig({
  site: env.SITE_URL ?? "http://localhost:4321",
  integrations: [react()],
  server: {
    host: "127.0.0.1",
    port: 4321,
  },
  vite: {
    server: {
      proxy: {
        "/api": {
          target: env.RAG_API_URL ?? "http://127.0.0.1:8000",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
        },
      },
    },
  },
});
