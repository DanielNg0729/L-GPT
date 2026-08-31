import { defineConfig } from 'vite'

// The browser talks to Vite; Vite forwards /api to the Python agent. That keeps the
// frontend origin-clean and means no CORS handling in the page itself.
export default defineConfig({
  server: {
    port: 5173,
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true } },
  },
})
