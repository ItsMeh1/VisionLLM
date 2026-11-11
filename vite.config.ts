import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// VisionLLM frontend. The training engine is a separate local process on :8765.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '127.0.0.1',
    proxy: {
      '/engine': {
        target: 'http://127.0.0.1:8765',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/engine/, ''),
      },
    },
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  build: {
    target: 'es2022',
  },
})
