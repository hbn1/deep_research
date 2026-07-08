import { fileURLToPath, URL } from 'node:url'

import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueDevTools from 'vite-plugin-vue-devtools'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const projectRoot = fileURLToPath(new URL('../..', import.meta.url))
  const env = {
    ...loadEnv(mode, projectRoot, ''),
    ...loadEnv(mode, process.cwd(), ''),
    ...process.env,
  }
  const apiTarget = env.VITE_API_TARGET || 'http://127.0.0.1:8000'
  const proxyHeaders: Record<string, string> = {}
  if (env.API_AUTH_KEY) proxyHeaders['X-API-Key'] = env.API_AUTH_KEY
  if (env.ADMIN_API_KEY) proxyHeaders['X-Admin-Key'] = env.ADMIN_API_KEY

  return {
    plugins: [
      vue(),
      vueDevTools(),
    ],
    server: {
      host: '0.0.0.0',
      port: 5173,
      proxy: {
        '/api': {
          target: apiTarget,
          changeOrigin: true,
          ...(Object.keys(proxyHeaders).length ? { headers: proxyHeaders } : {}),
        },
        '/health': {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      },
    },
  }
})
