# 0002. Use Next.js and React for the Frontend

## Status

Superseded

Superseded by
[0006. Use a Single Astro Frontend](0006-evaluate-single-astro-vite-frontend-vs-wikijs-split.md).

Retained as historical context for earlier Next.js chat app choice. Current accepted frontend is Astro.

## Context

**Split frontend model** decided (ADR 0001): Wiki.js for write/browse, custom Next.js chat for RAG with citations. Next choice: frontend foundation for **chat application**.

Frontend needs interactive workflows: search input, filters, result cards, document previews, selected sources, citations, eventually RAG assistant panel. Visualize locally before committing to backend, auth, search, or RAG details.

## Historical Decision

Use Next.js with React for frontend.

## Reasons

### React Gives Reusable UI Components

React = reusable components. Fits expected product shapes:

- search bars
- document or topic filters
- source cards
- document preview panels
- selected-source lists
- chat and RAG assistant panels
- citation displays

Less custom browser code vs plain HTML, CSS, JavaScript.

### Next.js Provides Application Structure

Next.js structures React into full web app: routing, dev server, build tooling, deploy conventions, env vars, future auth/API integration points.

Start with local visual prototype, grow to production without changing foundation immediately.

### The Ecosystem Has Common Design Primitives

React + Next.js mature ecosystem. shadcn/ui, Radix UI, Tailwind CSS for buttons, cards, dialogs, tabs, inputs, badges, layouts, sidebars.

Don't design every common UI element from scratch. Focus on product-specific experience: searching company docs, inspecting sources, asking grounded questions.

### It Supports A Slow Build

Stack supports incremental approach. First step: static or mocked local visualization. Later: real APIs, auth, hybrid search, RAG when architecture decisions ready.

## Alternatives Considered

### Plain HTML, CSS, and JavaScript

Minimizes dependencies, but enough interactive state makes raw browser code harder to maintain.

### Vite and React

Lighter React setup, good for standalone SPA. Chose Next.js for more application structure: routing, future auth, production growth.

### SvelteKit

Strong framework, clean DX. Not chosen — React ecosystem has broader enterprise familiarity and mature component libraries for internal tools.

### Vue or Nuxt

Reasonable for app frontends. Not chosen — React + Next.js more common default for component ecosystem and RAG-style app examples we likely build from.
