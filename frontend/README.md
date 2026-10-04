# RestoreLab frontend

Vite, React, TypeScript, Tailwind, and React Router interface for the RestoreLab API.

```sh
npm install
npm run dev
```

Vite proxies `/api` requests to `http://localhost:8000` during development. Run the FastAPI service on that address or change `server.proxy` in `vite.config.ts`. For a production deployment, serve the frontend and API behind the same origin so the relative `/api/...` routes remain available.

Replace the placeholder W&B project URLs in `src/config.ts` with the team's project links.
