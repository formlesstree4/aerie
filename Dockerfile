# syntax=docker/dockerfile:1

# ---- Build stage ----
FROM node:22-alpine AS build
WORKDIR /app

# Install dependencies against the lockfile for reproducible builds.
COPY package.json package-lock.json ./
RUN npm ci

# Compile TypeScript and produce the static Vite bundle in /app/dist.
COPY . .
RUN npm run build

# ---- Runtime stage ----
FROM nginx:1.27-alpine AS runtime

# openssl CLI for the self-signed certificate generated at first start.
RUN apk add --no-cache openssl

# SPA-friendly nginx config (falls back to index.html), HTTP + HTTPS listeners.
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY nginx-app.conf /etc/nginx/snippets/aerie-app.conf

# Mints /etc/nginx/certs/aerie.{crt,key} before nginx starts, if absent.
COPY --chmod=755 docker/40-aerie-cert.sh /docker-entrypoint.d/40-aerie-cert.sh

# Serve the compiled static assets.
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80 443

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -q --spider http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
