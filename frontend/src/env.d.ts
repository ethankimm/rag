/// <reference types="astro/client" />

interface ImportMetaEnv {
  readonly PUBLIC_RAG_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
