# ---- Build stage ----
ARG NODE_IMAGE=node:22-alpine
ARG NGINX_IMAGE=nginx:1.27-alpine

FROM ${NODE_IMAGE} AS builder
WORKDIR /app

ARG NPM_CONFIG_REGISTRY=https://registry.npmmirror.com
ARG VITE_ENABLE_ADMIN_TOOLS=false
ENV NPM_CONFIG_REGISTRY=${NPM_CONFIG_REGISTRY}
ENV VITE_ENABLE_ADMIN_TOOLS=${VITE_ENABLE_ADMIN_TOOLS}

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---- Runtime stage ----
FROM ${NGINX_IMAGE} AS runtime

ENV CLIENT_MAX_BODY_SIZE=25m
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx/default.conf.template /etc/nginx/templates/default.conf.template

EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1/health >/dev/null || exit 1
