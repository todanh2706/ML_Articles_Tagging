FROM node:22-alpine

WORKDIR /usr/src/app

COPY package*.json ./
COPY tailwind.config.js postcss.config.js ./

RUN npm ci

COPY server.js ./server.js
COPY config ./config
COPY routes ./routes
COPY views ./views
COPY public ./public

RUN npm run build:css

ENV PORT=3000
EXPOSE 3000

CMD ["node", "server.js"]
