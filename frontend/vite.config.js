import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { dirname } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  root: __dirname,
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy /api/* requests to the FastAPI backend in development
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          markdown: ['react-markdown', 'remark-gfm'],
          icons: ['lucide-react'],
        },
      },
    },
  },
})
